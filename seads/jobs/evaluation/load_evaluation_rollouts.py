"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

from seads import EXPERIMENT_DIR
from seads.utils import pickle_load


def load_evaluation_rollouts(
    run_stem_list,
    n_seeds,
    n_rollouts_per_depth,
    keep_episodes=False,
    env_seed_gen_seed=4343,
):
    run_base_dir = EXPERIMENT_DIR.joinpath("seads")
    run_seed_list = list(range(1, n_seeds + 1))

    seed_gen = np.random.RandomState(env_seed_gen_seed)
    env_seed_whitelist = [
        seed_gen.randint(0, int(1e8)) for _ in range(n_rollouts_per_depth)
    ]

    def _load_rollout_worker(rollout_filename):
        if not rollout_filename.is_file():
            return None
        with open(rollout_filename, "rb") as handle:
            try:
                rollout_data = pickle_load(handle)
            except EOFError:
                print(f"EOFError for {rollout_filename}")
                rollout_data = None
        return rollout_data

    def load_rollouts(run_dir):
        results_dir = run_dir.joinpath("eval_rollouts")
        rollouts = []
        if results_dir.is_dir():
            rollout_filenames = list(results_dir.iterdir())
            rollouts = Parallel(n_jobs=8)(
                tqdm([delayed(_load_rollout_worker)(f) for f in rollout_filenames])
            )
        else:
            print(f"Results dir not found at {results_dir}")
        return rollouts

    run_info_list = []
    for run_stem, run_seed in product(run_stem_list, run_seed_list):
        run_dir = run_base_dir.joinpath(f"{run_stem}_s{run_seed}")
        run_info_list.append((run_stem, run_seed, run_dir))

    all_rollouts = []
    for run_stem, run_seed, run_dir in tqdm(run_info_list):
        rollouts = load_rollouts(run_dir)
        for r in rollouts:
            if r is None:
                continue
            if int(r["env_seed"]) not in env_seed_whitelist:
                continue
            if "episodes" in r:
                # compute episode lengths
                episode_len = [len(e) for e in r["episodes"]]
                r["episode_len"] = episode_len
                r["total_len"] = sum(episode_len)
                if not keep_episodes:
                    del r["episodes"]
            r["run_stem"] = run_stem
            r["run_seed"] = run_seed
            all_rollouts.append(r)
    df_all = pd.DataFrame(all_rollouts)
    # replace NaN by 'False'
    df_all["success_all"] = df_all["success_all"] == True
    df_all["success_lowlevel"] = df_all["success_lowlevel"] == True

    if "timeout" in df_all:
        df_all["timeout"] = df_all["timeout"] == True
    else:
        df_all["timeout"] = False

    if "planning_failed" in df_all:
        df_all["planning_failed"] = df_all["planning_failed"] == True
    else:
        df_all["planning_failed"] = False

    df_all["env_steps"] = (
        df_all["env_interactions_lpfm"] + df_all["env_interactions_sac"]
    )
    mean_success = (
        df_all.groupby(["run_stem", "run_seed", "replanning", "ckpt_step"])[
            ["env_steps", "success_all"]
        ]
        .mean()
        .reset_index()
    )
    mean_success["mean_val_success_rate"] = mean_success["success_all"]
    return df_all, mean_success
