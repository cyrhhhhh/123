"""DLF-style experiment runner for DMRL."""

import gc
import logging
import random
from pathlib import Path

import numpy as np
import torch

from config_dmrl_dlf import get_config_regression
from data_loader_dmrl import MMDataLoader
from DMRL import DMRL
from trainer_dmrl_dlf import DMRLTrainer


logger = logging.getLogger("DMRL")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(Path(log_dir) / f"{model_name}-{dataset_name}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] - %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel({0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}[verbose_level])
    stream_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    logger.addHandler(stream_handler)
    return logger


def DMRL_run(
    model_name="DMRL",
    dataset_name="mosi",
    config=None,
    config_file="",
    seeds=None,
    model_save_dir="./pt",
    log_dir="./log",
    gpu_id=0,
    num_workers=0,
    verbose_level=1,
):
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()
    seeds = list(seeds) if seeds else [1111, 1112, 1113, 1114, 1115]

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    logger_obj = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    args = get_config_regression(model_name, dataset_name, config_file)
    if config:
        args.update(config)

    if bool(args.get("cpu", False)) or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{gpu_id}")
    args.train_mode = "regression"

    all_results = []
    for index, seed in enumerate(seeds):
        setup_seed(int(seed))
        args.cur_seed = index + 1
        args.model_save_path = str(Path(model_save_dir) / f"{model_name}-{dataset_name}-{seed}.pth")
        logger_obj.info("Running %s on %s, seed=%s, device=%s", model_name, dataset_name, seed, args.device)

        dataloader = MMDataLoader(args, num_workers=num_workers)
        # The pkl decides whether text_bert is actually available. Keep model
        # construction synchronized with the DataLoader's inferred choice.
        args.use_bert = bool(args.effective_use_bert)
        model = DMRL(args).to(args.device)
        trainer = DMRLTrainer(args)
        trainer.do_train(model, dataloader)

        # Test the validation-selected checkpoint, matching DLF's post-train flow.
        model.load_state_dict(torch.load(args.model_save_path, map_location=args.device))
        result = trainer.do_test(model, dataloader["test"], mode="TEST-BEST")
        all_results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(all_results) > 1:
        logger_obj.info("===== averaged over seeds =====")
        for key in all_results[0]:
            values = [r[key] for r in all_results]
            logger_obj.info("%s: %.4f +/- %.4f", key, np.mean(values), np.std(values))

    return all_results
