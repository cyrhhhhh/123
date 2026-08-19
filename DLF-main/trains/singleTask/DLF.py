import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss


logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def _ccdr_losses(self, output):
        zero = output['output_logit'].new_zeros(())
        if 'conflict_energy' not in output:
            return zero, zero, zero, zero, zero, zero

        reconstruction_loss = sum(
            self.MSE(reconstructed, target)
            for reconstructed, target in zip(
                output['ccdr_reconstructed'],
                output['ccdr_targets'],
            )
        )

        def covariance_penalty(left, right):
            left = left.reshape(-1, left.size(-1))
            right = right.reshape(-1, right.size(-1))
            left = left - left.mean(dim=0, keepdim=True)
            right = right - right.mean(dim=0, keepdim=True)
            covariance = left.transpose(0, 1).matmul(right)
            covariance = covariance / max(left.size(0) - 1, 1)
            return covariance.pow(2).mean()

        decorrelation_loss = zero
        for consensus, support, conflict in zip(
            output['consensus_parts'],
            output['support_parts'],
            output['conflict_parts'],
        ):
            decorrelation_loss = decorrelation_loss + (
                covariance_penalty(consensus, support)
                + covariance_penalty(consensus, conflict)
                + covariance_penalty(support, conflict)
            )

        modality_logits = torch.cat([
            output['logits_l_shared'],
            output['logits_a_shared'],
            output['logits_v_shared'],
        ], dim=1).detach()
        prediction_spread = modality_logits.std(dim=1, unbiased=False)
        prediction_signs = torch.sign(modality_logits)
        sign_disagreement = (
            prediction_signs.max(dim=1).values
            - prediction_signs.min(dim=1).values
        ).abs() * 0.5
        weak_conflict = prediction_spread + sign_disagreement

        conflict_energy = output['conflict_energy'].mean(dim=1)
        target_difference = weak_conflict[:, None] - weak_conflict[None, :]
        energy_difference = conflict_energy[:, None] - conflict_energy[None, :]
        valid_pairs = target_difference.abs() > 1e-4
        if valid_pairs.any():
            ranking_loss = torch.relu(
                0.1 - target_difference.sign() * energy_difference
            )[valid_pairs].mean()
        else:
            ranking_loss = zero

        low_conflict_weight = torch.exp(-weak_conflict)
        delta_loss = (
            low_conflict_weight * output['conflict_delta'].view(-1).abs()
        ).mean()

        conflict_ratio = torch.stack([
            (1.0 - gate).mean(dim=(0, 2))
            for gate in output['route_gates']
        ], dim=1).mean(dim=1)
        routing_target = weak_conflict / (1.0 + weak_conflict)
        routing_loss = self.MSE(conflict_ratio, routing_target)

        consensus_logit = output['consensus_logit'].detach().view(-1)
        consensus_direction = torch.sign(consensus_logit)
        valid_direction = consensus_logit.abs() > 0.1
        if valid_direction.any():
            support_effect_loss = torch.relu(
                -consensus_direction * output['support_delta'].view(-1)
            )[valid_direction].mean()
            conflict_effect_loss = (
                routing_target * torch.relu(
                    consensus_direction * output['conflict_delta'].view(-1)
                )
            )[valid_direction].mean()
            effect_loss = support_effect_loss + conflict_effect_loss
        else:
            effect_loss = zero
        return (
            reconstruction_loss,
            decorrelation_loss,
            ranking_loss,
            delta_loss,
            routing_loss,
            effect_loss,
        )

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters() 

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)    
        model = net
        
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()
              

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)


                   
                    output = model[0](text, audio, vision) 

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)  
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task_shared = (
                        self.criterion(output['logits_l_shared'], labels)
                        + self.criterion(output['logits_v_shared'], labels)
                        + self.criterion(output['logits_a_shared'], labels)
                    )
                    shared_task_weight = (
                        getattr(self.args, 'ccdr_shared_task_weight', 0.5)
                        if 'conflict_energy' in output else 0.0
                    )
                    consensus_task_loss = (
                        self.criterion(output['consensus_logit'], labels)
                        if 'consensus_logit' in output else loss_task_all.new_zeros(())
                    )
                    
                    # total MSA loss L_msa
                    loss_task = (
                        loss_task_all
                        + loss_task_c
                        + 3 * loss_task_l_hetero
                        + loss_task_v_hetero
                        + loss_task_a_hetero
                        + shared_task_weight * loss_task_shared
                        + getattr(self.args, 'cred_consensus_task_weight', 0.5) * consensus_task_loss
                    )
                    
                    # reconstruction loss L_r
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s 
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10

                    s_l_flat = output['s_l'].reshape(-1, num)
                    s_v_flat = output['s_v'].reshape(-1, num)
                    s_a_flat = output['s_a'].reshape(-1, num)
                    cosine_similarity_s_c_l = self.cosine(s_l_flat, output['c_l'].reshape(-1, num), -torch.ones(s_l_flat.size(0), device=s_l_flat.device))
                    cosine_similarity_s_c_v = self.cosine(s_v_flat, output['c_v'].reshape(-1, num), -torch.ones(s_v_flat.size(0), device=s_v_flat.device))
                    cosine_similarity_s_c_a = self.cosine(s_a_flat, output['c_a'].reshape(-1, num), -torch.ones(s_a_flat.size(0), device=s_a_flat.device))
                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # triplet margin loss L_m
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    (
                        loss_ccdr_recon,
                        loss_ccdr_decor,
                        loss_ccdr_rank,
                        loss_ccdr_delta,
                        loss_cred_routing,
                        loss_cred_effect,
                    ) = self._ccdr_losses(output)
                    loss_ccdr = (
                        getattr(self.args, 'ccdr_reconstruction_weight', 0.1) * loss_ccdr_recon
                        + getattr(self.args, 'ccdr_decorrelation_weight', 0.01) * loss_ccdr_decor
                        + getattr(self.args, 'ccdr_ranking_weight', 0.1) * loss_ccdr_rank
                        + getattr(self.args, 'ccdr_delta_weight', 0.05) * loss_ccdr_delta
                        + getattr(self.args, 'cred_routing_weight', 0.1) * loss_cred_routing
                        + getattr(self.args, 'cred_effect_weight', 0.05) * loss_cred_effect
                    )

                    #overall loss L_DLF
                    combined_loss = loss_task + (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1 + loss_ccdr
                
                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())  
                                 
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/DLF' + str(self.args.dataset_name)+'.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results