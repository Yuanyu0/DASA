"""Pack ODE trajectory shards into an LMDB dataset."""
# Isolate backbone imports and resolve repository paths.
import sys as _dasa_sys
from pathlib import Path as _DASAPath
_dasa_root = next(p for p in _DASAPath(__file__).resolve().parents if (p / 'dasa' / 'runtime.py').is_file())
if str(_dasa_root) not in _dasa_sys.path:
    _dasa_sys.path.insert(0, str(_dasa_root))
from dasa.runtime import prepare_runtime, load_config, config_path
prepare_runtime('self_forcing')

from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os

from utils.lmdb import store_arrays_to_lmdb, process_data_dict


def main():
    """Pack per-video .pt ODE trajectory pairs into LMDB."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to ode pairs")
    parser.add_argument("--lmdb_path", type=str,
                        required=True, help="path to lmdb")
    parser.add_argument("--num_shards", type=int,
                        default=16, help="num_shards")

    args = parser.parse_args()

    all_dirs = sorted(os.listdir(args.data_path))

    map_size = int(1e12)  # 1 TB LMDB capacity.
    os.makedirs(args.lmdb_path, exist_ok=True)
    envs = []
    num_shards = args.num_shards
    for shard_id in range(num_shards):
        print("shard_id ", shard_id)
        path = os.path.join(args.lmdb_path, f"shard_{shard_id}")
        env = lmdb.open(path,
                        map_size=map_size,
                        subdir=True,       # set to True if you want a directory per env
                        readonly=False,
                        metasync=True,
                        sync=True,
                        lock=True,
                        readahead=False,
                        meminit=False)
        envs.append(env)

    counters = [0] * num_shards
    seen_prompts = set()  # for deduplication
    total_samples = 0
    all_files = []

    for part_dir in all_dirs:
        all_files += sorted(glob.glob(os.path.join(args.data_path, part_dir, "*.pt")))

    for idx, file in tqdm(enumerate(all_files)):
        try:
            data_dict = torch.load(file)
            data_dict = process_data_dict(data_dict, seen_prompts)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

        if data_dict["latents"].shape != (1, 21, 16, 60, 104):
            continue

        shard_id = idx % num_shards
        store_arrays_to_lmdb(envs[shard_id], data_dict, start_index=counters[shard_id])
        counters[shard_id] += len(data_dict['prompts'])
        data_shape = data_dict["latents"].shape

    total_samples += len(all_files)

    print(len(seen_prompts))

    for shard_id, env in enumerate(envs):
        with env.begin(write=True) as txn:
            for key, val in (data_dict.items()):
                assert len(data_shape) == 5
                array_shape = np.array(data_shape)
                array_shape[0] = counters[shard_id]
                shape_key = f"{key}_shape".encode()
                print(shape_key, array_shape)
                shape_str = " ".join(map(str, array_shape))
                txn.put(shape_key, shape_str.encode())

    print(f"Finished writing {total_samples} examples into {num_shards} shards under {args.lmdb_path}")


if __name__ == "__main__":
    main()
