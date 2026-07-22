"""DLF-style hierarchical configuration loader for DMRL."""

import copy
import json
import os
from pathlib import Path


class AttrDict(dict):
    """Dictionary with both item and attribute access, like DLF EasyDict."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_config_regression(model_name, dataset_name, config_file=""):
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()
    config_path = Path(config_file) if config_file else Path(__file__).parent / "configer" / "dmrl_dlf.json"

    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config_all = json.load(f)

    model_common = copy.deepcopy(config_all[model_name]["commonParams"])
    model_dataset = copy.deepcopy(config_all[model_name]["datasetParams"][dataset_name])
    dataset_common = config_all["datasetCommonParams"]
    dataset_args = copy.deepcopy(dataset_common[dataset_name])

    aligned_key = "aligned" if model_common.get("need_data_aligned", True) else "unaligned"
    if aligned_key in dataset_args:
        dataset_args = copy.deepcopy(dataset_args[aligned_key])

    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
    }
    config.update(dataset_args)
    config.update(model_common)
    config.update(model_dataset)

    if dataset_name != "synthetic":
        root = dataset_common.get("dataset_root_dir", "./dataset")
        feature_path = config["featurePath"]
        if not os.path.isabs(feature_path):
            config["featurePath"] = os.path.normpath(os.path.join(root, feature_path))

    return AttrDict(config)
