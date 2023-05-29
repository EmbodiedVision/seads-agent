"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pickle
from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from seads import EXPERIMENT_DIR


def load_skill_coverage(run_prefix, envs, variants, run_seeds):
    checkpoints_to_load_list = []
    for env, variant_name, seed in list(product(envs, variants, run_seeds)):
        if env in ["lightsout", "tileswap"]:
            surrogate_env = {
                "lightsout": "lightsout_cursor",
                "tileswap": "tileswap_cursor",
            }[env]
        else:
            surrogate_env = env

        run_id = f"{run_prefix}_{env}_{variant_name}_s{seed}"
        run_dir = EXPERIMENT_DIR.joinpath("seads", run_id)

        if not run_dir.joinpath("skill_eval").is_dir():
            print(f"{run_dir} has no eval")
            continue

        for checkpoint_file in run_dir.joinpath("skill_eval").iterdir():
            checkpoint_step = int(checkpoint_file.stem.split("_")[2][len("ckpt") :])
            meta_data = {
                "run_stem": (run_prefix + "_" + surrogate_env + "_" + variant_name),
                "run_seed": seed,
                "checkpoint": checkpoint_step,
            }
            checkpoints_to_load_list.append((checkpoint_file, meta_data))

    evaluation_list = Parallel(n_jobs=24)(
        delayed(_load_checkpoint)(f, m) for f, m in tqdm(checkpoints_to_load_list)
    )
    evaluation_list = [e for e in evaluation_list if e is not None]
    df = pd.DataFrame(evaluation_list)
    return df


def _load_checkpoint(checkpoint_file, meta_data):
    eval_data = None
    try:
        with open(checkpoint_file, "rb") as handle:
            eval_data = pickle.load(handle)
            eval_data["run_stem"] = meta_data["run_stem"]
            eval_data["run_seed"] = meta_data["run_seed"]
            eval_data["checkpoint"] = meta_data["checkpoint"]
    except Exception as e:
        print(e)
        print(checkpoint_file)
        print(meta_data)
    return eval_data
