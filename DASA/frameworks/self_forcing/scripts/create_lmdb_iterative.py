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

    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_path, "*.pt")))

    total_array_size = 5000000000000  # 5 TB LMDB capacity.

    env = lmdb.open(args.lmdb_path, map_size=total_array_size * 2)

    counter = 0

    seen_prompts = set()  # for deduplication

    for index, file in tqdm(enumerate(all_files)):
        data_dict = torch.load(file)

        data_dict = process_data_dict(data_dict, seen_prompts)

        store_arrays_to_lmdb(env, data_dict, start_index=counter)
        counter += len(data_dict['prompts'])

    with env.begin(write=True) as txn:
        for key, val in data_dict.items():
            print(key, val)
            array_shape = np.array(val.shape)
            array_shape[0] = counter

            shape_key = f"{key}_shape".encode()
            shape_str = " ".join(map(str, array_shape))
            txn.put(shape_key, shape_str.encode())


if __name__ == "__main__":
    main()
