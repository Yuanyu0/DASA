# Isolate backbone imports and resolve repository paths.
import sys as _dasa_sys
from pathlib import Path as _DASAPath
_dasa_root = next(p for p in _DASAPath(__file__).resolve().parents if (p / 'dasa' / 'runtime.py').is_file())
if str(_dasa_root) not in _dasa_sys.path:
    _dasa_sys.path.insert(0, str(_dasa_root))
from dasa.runtime import prepare_runtime, load_config, config_path
prepare_runtime('longlive')

# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from omegaconf import OmegaConf
import wandb
import torch.distributed as dist

from trainer import ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None, help="Run this many additional optimizer steps")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")

    args = parser.parse_args()

    config = load_config(args.config_path, "longlive")
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max_steps must be positive")
        config.max_steps = args.max_steps
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume
    config.use_one_logger = not args.no_one_logger

    try:
        if config.trainer == "score_distillation":
            trainer = ScoreDistillationTrainer(config)
        trainer.train()

        wandb.finish()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
