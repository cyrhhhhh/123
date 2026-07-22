"""DLF-style trainer adapted to the DMRL output/loss interface."""

import logging

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics_dmrl import MetricsTop, dict_to_str


logger = logging.getLogger("DMRL")


def build_masks(text, audio, vision, use_bert):
    if use_bert:
        text_mask = text[:, 1, :].long() > 0
    else:
        text_mask = text.abs().sum(dim=-1) > 0
    audio_mask = audio.abs().sum(dim=-1) > 0
    vision_mask = vision.abs().sum(dim=-1) > 0

    # Avoid all-masked rows, which make Transformer attention undefined.
    for mask in (text_mask, audio_mask, vision_mask):
        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty, 0] = True
    return text_mask, audio_mask, vision_mask


class DMRLTrainer:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def _forward(self, model, batch_data, with_labels):
        vision = batch_data["vision"].to(self.args.device)
        audio = batch_data["audio"].to(self.args.device)
        text = batch_data["text"].to(self.args.device)
        labels = batch_data["labels"]["M"].to(self.args.device).view(-1)

        text_mask, audio_mask, vision_mask = build_masks(
            text, audio, vision, bool(self.args.effective_use_bert)
        )
        output = model(
            text=text,
            audio=audio,
            video=vision,
            text_mask=text_mask,
            audio_mask=audio_mask,
            video_mask=vision_mask,
            labels=labels if with_labels else None,
        )
        return output, labels

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.args.patience,
        )

        epoch_results = {"train": [], "valid": [], "test": []} if return_epoch_results else None
        min_or_max = "min" if self.args.KeyEval in ("Loss", "MAE") else "max"
        best_valid = float("inf") if min_or_max == "min" else -float("inf")
        best_epoch = 0
        epochs = 0

        while True:
            epochs += 1
            model.train()
            optimizer.zero_grad()

            train_total = 0.0
            y_pred, y_true = [], []
            accumulation = max(int(self.args.update_epochs), 1)

            for step, batch_data in enumerate(dataloader["train"], start=1):
                output, labels = self._forward(model, batch_data, with_labels=True)
                total_loss = output["losses"]["total_loss"]
                (total_loss / accumulation).backward()

                is_update = step % accumulation == 0 or step == len(dataloader["train"])
                if is_update:
                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                train_total += float(total_loss.detach().item())
                y_pred.append(output["output_logit"].detach().cpu())
                y_true.append(labels.detach().cpu())

            train_loss = train_total / max(len(dataloader["train"]), 1)
            train_results = self.metrics(torch.cat(y_pred), torch.cat(y_true))
            train_results["Loss"] = round(train_loss, 4)

            logger.info(
                ">> Epoch: %d TRAIN-(%s) [%d/%d/%s] >> %s",
                epochs,
                self.args.model_name,
                epochs - best_epoch,
                epochs,
                self.args.cur_seed,
                dict_to_str(train_results),
            )

            val_results = self.do_test(model, dataloader["valid"], mode="VAL")
            test_results = self.do_test(model, dataloader["test"], mode="TEST")
            scheduler.step(val_results["Loss"])

            cur_valid = val_results[self.args.KeyEval]
            is_better = (
                cur_valid <= best_valid - 1e-6
                if min_or_max == "min"
                else cur_valid >= best_valid + 1e-6
            )
            if is_better:
                best_valid = cur_valid
                best_epoch = epochs
                torch.save(model.state_dict(), self.args.model_save_path)

            if return_epoch_results:
                epoch_results["train"].append(train_results)
                epoch_results["valid"].append(val_results)
                epoch_results["test"].append(test_results)

            if epochs - best_epoch >= self.args.early_stop:
                break
            if self.args.max_epochs > 0 and epochs >= self.args.max_epochs:
                break

        return epoch_results

    @torch.no_grad()
    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        for batch_data in dataloader:
            output, labels = self._forward(model, batch_data, with_labels=False)
            task_loss = self.criterion(output["output_logit"].view(-1), labels)
            eval_loss += float(task_loss.detach().item())
            y_pred.append(output["output_logit"].detach().cpu())
            y_true.append(labels.detach().cpu())

        pred = torch.cat(y_pred)
        true = torch.cat(y_true)
        results = self.metrics(pred, true)
        results["Loss"] = round(eval_loss / max(len(dataloader), 1), 4)
        logger.info("%s-(%s) >> %s", mode, self.args.model_name, dict_to_str(results))
        return results
