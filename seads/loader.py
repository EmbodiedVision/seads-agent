"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import torch
from seads_envs import load_env

from seads.agent import HybridAgent
from seads.utils import torch_load
from seads.utils.sacred_utils import load_json

__all__ = ["load_config", "load_agent", "load_agent_and_env"]


def load_config(run_directory):
    config_file = run_directory.joinpath("config.json")
    config = load_json(config_file)
    return config


def load_agent(
    run_directory,
    checkpoint_step,
    env_name,
    env,
    device=None,
    return_checkpoint_data=False,
):

    config = load_config(run_directory)
    if device is None:
        device = "cuda" if config["sac_args"]["cuda"] else "cpu"
    config["sac_args"]["cuda"] = device == "cuda"

    if "skill_model_kwargs" in config:
        skill_model_kwargs = config["skill_model_kwargs"]
    else:
        if config["skill_model_type"] == "forward_model":
            skill_model_kwargs = config["lpfm_kwargs"]
            if "pfm_kwargs" not in skill_model_kwargs:
                skill_model_kwargs["pfm_kwargs"] = config["pfm_kwargs"]
        elif config["skill_model_type"] == "discriminator":
            skill_model_kwargs = config["skill_discriminator_kwargs"]
        else:
            raise ValueError

    # Hybrid agent
    hybrid_agent = HybridAgent(
        env,
        num_skills=config["num_skills"],
        skill_model_type=config["skill_model_type"],
        use_groundtruth_phi=config["use_groundtruth_phi"],
        sac_args=config["sac_args"],
        train_sac=config["train_sac"],
        loss_n_z_samples=config["loss_n_z_samples"],
        reward_n_z_samples=config["reward_n_z_samples"],
        reward_type=config["reward_type"],
        reward_options=config["reward_options"],
        device=device,
        skill_model_kwargs=skill_model_kwargs,
    )

    if checkpoint_step == "last":
        checkpoint_dir = run_directory.joinpath("checkpoints")
        checkpoint_steps = [int(f.stem.split("_")[1]) for f in checkpoint_dir.iterdir()]
        if len(checkpoint_steps) == 0:
            raise RuntimeError("No checkpoints available")
        else:
            checkpoint_step = max(checkpoint_steps)
            print(f"Loading checkpoint at {checkpoint_step}")

    checkpoint_filename = run_directory.joinpath(
        "checkpoints", f"step_{checkpoint_step}.pkl"
    )
    checkpoint_data = torch_load(checkpoint_filename, map_location=torch.device(device))
    hybrid_agent.load_checkpoint(checkpoint_data, with_optim=False)
    if return_checkpoint_data:
        return hybrid_agent, config, checkpoint_data
    else:
        return hybrid_agent, config


def load_agent_and_env(
    run_directory, checkpoint_step, env_reset_split, env_kwargs_update=None
):
    device = "cuda"
    config = load_config(run_directory)
    if env_kwargs_update is None:
        env_kwargs_update = {}
    env_kwargs = dict(**config["env_kwargs"], **env_kwargs_update)
    env_name = config["env_name"]
    reset_split = env_reset_split
    proto_env = load_env(
        env_name,
        reset_split,
        env_kwargs,
        ext_timelimit_obs=config["ext_timelimit_obs"],
        time_limit=True,
        prototype=False,
    )
    hybrid_agent, config = load_agent(
        run_directory, checkpoint_step, env_name, proto_env, device=device
    )
    return hybrid_agent, proto_env
