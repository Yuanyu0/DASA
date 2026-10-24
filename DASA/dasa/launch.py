"""Launch framework scripts in isolated processes; --dry-run needs only Python."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dasa.runtime import ROOT, FRAMEWORKS, config_path, framework_dir

ENTRIES = {
    "self_forcing": {
        "inference": ("inference.py", "inference.yaml", False),
        "finetune": ("train.py", "finetune.yaml", True),
        "train_dmd": ("train.py", "dmd.yaml", True),
        "train_sid": ("train.py", "sid.yaml", True),
        "demo": ("demo.py", "inference.yaml", False),
        "generate_ode_pairs": ("scripts/generate_ode_pairs.py", None, True),
        "create_lmdb_iterative": ("scripts/create_lmdb_iterative.py", None, False),
        "create_lmdb_14b_shards": ("scripts/create_lmdb_14b_shards.py", None, False),
        "vae_torch2trt": ("demo_utils/vae_torch2trt.py", None, False),
    },
    "longlive": {
        "inference": ("inference.py", "inference.yaml", False),
        "finetune": ("train.py", "finetune.yaml", True),
        "train_init": ("train.py", "train_init.yaml", True),
        "train_long": ("train.py", "train_long.yaml", True),
        "interactive": ("interactive_inference.py", "interactive_inference.yaml", False),
        "infinity": ("inference.py", "inference_infinity.yaml", False),
    },
}


def build_command(framework, action, extra, nproc=1, master_port=29500):
    if action not in ENTRIES[framework]:
        raise ValueError(f"Unknown {framework} action {action!r}; choose from {', '.join(ENTRIES[framework])}")
    filename, config, training = ENTRIES[framework][action]
    entry = framework_dir(framework) / filename
    forwarded = list(extra)
    if config and not any(a == "--config_path" or a.startswith("--config_path=") for a in forwarded):
        forwarded[:0] = ["--config_path", str(config_path(framework, config))]
    if filename == "train.py":
        if not any(a == "--logdir" or a.startswith("--logdir=") for a in forwarded):
            forwarded += ["--logdir", str(ROOT / "outputs" / framework / action)]
        if "--disable-wandb" not in forwarded:
            forwarded += ["--disable-wandb"]
        if framework == "longlive" and "--no-one-logger" not in forwarded:
            forwarded += ["--no-one-logger"]
    if training or nproc > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
                   f"--nproc_per_node={nproc}", f"--master_port={master_port}", str(entry)]
    else:
        command = [sys.executable, str(entry)]
    environment = os.environ.copy()
    environment["DASA_ACTIVE_FRAMEWORK"] = framework
    paths = [str(framework_dir(framework)), str(ROOT)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return command + forwarded, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("action", help="inference, finetune, or a preserved advanced entry")
    parser.add_argument("--nproc_per_node", type=int, default=int(os.environ.get("DASA_NPROC_PER_NODE", "1")))
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("DASA_MASTER_PORT", "29500")))
    parser.add_argument("--dry-run", action="store_true", help="Print the command without importing ML libraries")
    args, extra = parser.parse_known_args(argv)
    if args.nproc_per_node < 1:
        parser.error("--nproc_per_node must be positive")
    try:
        command, environment = build_command(args.framework, args.action, extra, args.nproc_per_node, args.master_port)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"cwd": str(ROOT), "command": command, "pythonpath": environment["PYTHONPATH"]}, indent=2))
        return 0
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
