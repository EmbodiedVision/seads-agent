"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import os
from itertools import product
from json import JSONDecodeError
from pathlib import Path

from seads import EXPERIMENT_DIR
from seads.utils.sacred_utils import load_json

this_dir = Path(__file__).parent.resolve()

ENV_SET = [
    ("lightsout_cursor", "cursor"),
    ("tileswap_cursor", "cursor"),
    ("lightsout_reacher", "reacher"),
    ("tileswap_reacher", "reacher"),
    ("lightsout_jaco", "jaco"),
    ("tileswap_jaco", "jaco"),
    ("lightsout_jaco_stairs", "jaco"),
]

BASE_CONFIG = {
    "relabel_sac": 0.5,
    "relabel_skm": 1.0,
    "rollout_datasource": "both",
    "has_novelty_bonus": True,
    "reward_normalization": "neg_log_q_k_baseline",
    "num_skills": "default",
    "skill_model": "skill_forward_model",
    "ext_timelimit_obs": "'full'",
}

UPDATES_DEFAULT = {
    "default": {},
}

UPDATES = UPDATES_DEFAULT

SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

command_list_init_dict = {}
command_list_restart_dict = {}

for (env, env_set), seed in product(ENV_SET, SEED):

    for update_name, update_dict in UPDATES.items():

        if ("vic" in update_name) and ("cursor" not in env):
            # VIC variants only on Cursor envs
            continue

        if update_name != "default" and env == "lightsout_jaco_stairs":
            # For LightsoutJacoStairs, only run the default variant
            continue

        config = dict(**BASE_CONFIG)
        for k, v in update_dict.items():
            config[k] = v

        run_id = f"corl22_{env}_{update_name}_s{seed}"

        reward_components = ["log_q_k", config["reward_normalization"]]

        if config["has_novelty_bonus"]:
            reward_components.append("log_novb")

        if config["num_skills"] == "more":
            if "lightsout" in env:
                num_skills = 30
            elif "tileswap" in env:
                num_skills = 15
            else:
                raise ValueError
        elif config["num_skills"] == "default":
            if "lightsout" in env:
                num_skills = 25
            elif "tileswap" in env:
                num_skills = 12
            else:
                raise ValueError
        else:
            raise ValueError

        reward_components = [f"'{c}'" for c in reward_components]

        command_init = (
            "python -m seads.train_seads "
            "-p "
            f"--force-id={run_id} "
            f"with "
            f"{env}_env "
            f"{config['skill_model']} "
            f"rollout_datasource={config['rollout_datasource']} "
            f"relabel_sac={config['relabel_sac']} "
            f"relabel_skm={config['relabel_skm']} "
            f"\"reward_options.reward_components=[{','.join(reward_components)}]\" "
            f"save_all_episodes={(('cursor' in env) and (update_name == 'default'))} "  # for train/test overlap
            f"seed={112233 + seed} " + "\n"
        )

        command_restart = f"python -m seads.train_seads restart_base={run_id} \n"

        if env_set not in command_list_init_dict:
            command_list_init_dict[env_set] = []
        command_list_init = command_list_init_dict[env_set]

        if env_set not in command_list_restart_dict:
            command_list_restart_dict[env_set] = []
        command_list_restart = command_list_restart_dict[env_set]

        run_base_dir = EXPERIMENT_DIR.joinpath("seads")
        if run_base_dir.joinpath(run_id, "run.json").is_file():
            try:
                json_data = load_json(run_base_dir.joinpath(run_id, "run.json"))
                if json_data["status"] not in [
                    "COMPLETED",
                ]:
                    command_list_restart.append(command_restart)
            except JSONDecodeError:
                print(f"Run {run_id} broken")
        else:
            if run_base_dir.joinpath(run_id).is_dir():
                os.rmdir(run_base_dir.joinpath(run_id))
            command_list_init.append(command_init)
            command_list_restart.append(command_restart)


# write commands to job file
for env_set in command_list_init_dict.keys():

    with open(this_dir.joinpath(f"job_list_init_{env_set}.txt"), "w") as handle:
        handle.writelines(command_list_init_dict[env_set])
    with open(this_dir.joinpath(f"job_list_restart_{env_set}.txt"), "w") as handle:
        handle.writelines(command_list_restart_dict[env_set])
