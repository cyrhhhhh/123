"""Training entry for DMRL, mirroring DLF's thin train.py."""

import argparse

from run_dmrl_dlf import DMRL_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mosi", choices=["mosi", "mosei", "synthetic"])
    parser.add_argument("--config", default="./configer/dmrl_dlf.json")
    parser.add_argument("--seed", type=int, nargs="+", default=[1111])
    parser.add_argument("--feature_path", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.feature_path:
        overrides["featurePath"] = args.feature_path
    if args.cpu:
        overrides["cpu"] = True

    DMRL_run(
        model_name="DMRL",
        dataset_name=args.dataset,
        config=overrides,
        config_file=args.config,
        seeds=args.seed,
        model_save_dir="./pt",
        log_dir="./log",
        num_workers=0,
    )


if __name__ == "__main__":
    main()
