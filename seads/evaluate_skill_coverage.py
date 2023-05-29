"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from seads_envs import load_env
from tqdm import tqdm

from seads.common import collect_episode
from seads.loader import load_agent, load_config


def compute_skill_coverage(env, hybrid_agent):
    coverage_list = []
    for seed in tqdm(range(444555, 444555 + 20)):
        executed_action_list = []
        for symbolic_action in range(hybrid_agent.num_skills):
            env.seed(seed)
            episode = collect_episode(
                env,
                hybrid_agent,
                symbolic_action,
                random_actions=False,
                evaluate=True,
                reset=True,
                render=False,
            )
            executed_action = episode[-1]["info"].get("groundtruth_skill", -1)
            if executed_action is None:
                executed_action = -1
            executed_action_list.append(executed_action)
        unique_actions = np.unique(np.array(executed_action_list))
        if -1 in unique_actions:
            n_unique_actions = len(unique_actions) - 1
        else:
            n_unique_actions = len(unique_actions)
        coverage_list.append(n_unique_actions)
    return coverage_list


def compute_skill_metrics(env, hybrid_agent):
    coverage_list = compute_skill_coverage(env, hybrid_agent)
    skill_metrics = {
        "coverage_list": coverage_list,
        "covered_mean": np.mean(coverage_list),
    }
    return skill_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=False)
    parser.add_argument("--ckpt_step", type=int, default=200, required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False)
    parser.add_argument("--print_only", action="store_true")

    args = parser.parse_args()

    device = args.device

    run_directory = Path(args.run_dir)
    config = load_config(run_directory)
    env_kwargs = dict(**config["env_kwargs"])
    env_name = config["env_name"]
    reset_split = "test"
    env = load_env(
        env_name,
        reset_split,
        env_kwargs,
        time_limit=True,
        ext_timelimit_obs=config["ext_timelimit_obs"],
        prototype=False,
    )

    skill_eval_dir = run_directory.joinpath("skill_eval")
    os.makedirs(skill_eval_dir, exist_ok=True)

    checkpoint_set = set()
    if args.ckpt is None:
        assert args.ckpt_step is not None
        checkpoint_dir = run_directory.joinpath("checkpoints")
        checkpoint_steps = [int(f.stem.split("_")[1]) for f in checkpoint_dir.iterdir()]
        checkpoint_set.add(min(checkpoint_steps))
        checkpoint_set.add(max(checkpoint_steps))
        checkpoint_set.update(
            range(
                args.ckpt_step, max(checkpoint_steps) + args.ckpt_step, args.ckpt_step
            )
        )
    else:
        checkpoint_set = {args.ckpt}

    for checkpoint in tqdm(list(checkpoint_set)):

        filename = skill_eval_dir.joinpath(f"skill_coverage_ckpt{checkpoint}.pkl")
        if filename.exists():
            if not args.print_only:
                print(f"{filename} exists, skipping")
                continue

        checkpoint_filename = run_directory.joinpath(
            "checkpoints", f"step_{checkpoint}.pkl"
        )
        if not checkpoint_filename.is_file():
            print(f"Checkpoint {checkpoint_filename} does not exists")
            continue

        hybrid_agent, _, checkpoint_data = load_agent(
            run_directory,
            checkpoint,
            env_name,
            env,
            device=device,
            return_checkpoint_data=True,
        )
        skill_metrics = compute_skill_metrics(env, hybrid_agent)
        skill_metrics["env_interactions_lpfm"] = checkpoint_data[
            "env_interactions_lpfm"
        ]
        skill_metrics["env_interactions_sac"] = checkpoint_data["env_interactions_sac"]

        if not args.print_only:
            with open(filename, "wb") as handle:
                pickle.dump(skill_metrics, handle)
        else:
            print(skill_metrics)


if __name__ == "__main__":
    main()
