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
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from seads_envs import load_env

from seads.common import collect_episode
from seads.loader import load_agent, load_config
from seads.planning.bfs_planning import (
    FringeSizeExceededError,
    PlanningFailedError,
    bfs_search,
)

MAX_NUM_REPLANS = 10
MAX_FRINGE_SIZE = 14 * 1024 * 1024 * 1024  # 14GB


def apply_action_sequence(env, hybrid_agent, action_sequence, render):
    episodes = []
    success = True
    for symbolic_action in action_sequence:
        if hasattr(env, "reset_step_counter"):
            env.reset_step_counter()
        episode = collect_episode(
            env,
            hybrid_agent,
            symbolic_action,
            random_actions=False,
            evaluate=True,
            reset=False,
            render=render,
        )
        episodes.append(episode)
        # if 'done' is not set in last transition, episode execution failed
        if not episode[-1]["done"]:
            success = False
            break

    return episodes, success


def bfs_search_timeout(*args, **kwargs):
    timeout_s = kwargs.pop("timeout_s")
    result = func_timeout(timeout_s, bfs_search, args=args, kwargs=kwargs)
    return result


def predict_trajectory(forward_model, action_plan, state, device="cuda"):
    states = []
    forward_model = forward_model.to(device)
    last_state = torch.from_numpy(state).to(device).unsqueeze(0)
    for action in action_plan:
        next_state = forward_model.predict(
            last_state, torch.tensor([action]).to(device)
        ).mode
        states.append(next_state[0].detach().cpu().numpy())
        last_state = next_state
    return states


def solve_board_replanning_timeout(*args, **kwargs):
    try:
        result = solve_board_replanning(*args, **kwargs)
    except PlanningFailedError:
        result = {"planning_failed": True, "fail_cause": "planning_failed"}
    except FringeSizeExceededError:
        result = {"planning_failed": True, "fail_cause": "fringe_size_exceeded"}
    except FunctionTimedOut:
        result = {"planning_failed": True, "timeout": True, "fail_cause": "timeout"}
    return result


def solve_board_replanning(hybrid_agent, env, render, heuristics, timeout_s):
    state = env.get_current_observation()
    forward_model = hybrid_agent.skill_model
    n_bfs_steps = 20
    bfs_batch_size = 1024
    action_sequence = []
    episodes = []
    target_state = env.binary_symbolic_target_state.astype(bool)
    t0 = time.time()

    time_step = 0
    num_replans = 0
    plans = []
    predicted_trajectories = []
    planning_points = []

    while num_replans < MAX_NUM_REPLANS:
        planning_initial_state = env.get_binary_symbolic_state(state)
        plan = bfs_search_timeout(
            initial_state=planning_initial_state,
            target_state=target_state,
            forward_model=forward_model,
            n_steps=n_bfs_steps,
            batch_size=bfs_batch_size,
            device=hybrid_agent._device,
            fringe_size_limit=MAX_FRINGE_SIZE,
            heuristics=heuristics,
            timeout_s=timeout_s,
        )
        plans.append(plan)
        planning_points.append(time_step)

        num_replans += 1
        predicted_trajectory = predict_trajectory(
            forward_model, plan, planning_initial_state, device=hybrid_agent._device
        )
        predicted_trajectories.append(predicted_trajectory)

        for symbolic_action, pred_next_state in zip(plan, predicted_trajectory):
            if hasattr(env, "reset_step_counter"):
                env.reset_step_counter()
            episode = collect_episode(
                env,
                hybrid_agent,
                symbolic_action,
                random_actions=False,
                evaluate=True,
                reset=False,
                render=render,
            )
            state = episode[-1]["next_state"]
            time_step += 1

            action_sequence.append(symbolic_action)
            episodes.append(episode)

            if env.is_solved():
                return {
                    "action_sequence": action_sequence,
                    "success_lowlevel": True,
                    "success_all": True,
                    "episodes": episodes,
                    "plans": plans,
                    "planning_points": planning_points,
                    "predicted_trajectories": predicted_trajectories,
                    "planning_time": time.time() - t0,
                    "n_replans": num_replans - 1,
                }

            if not np.array_equal(
                env.get_binary_symbolic_state(state), pred_next_state
            ):
                break

    result = {"planning_failed": True, "fail_cause": "replanning_failed"}
    return result


def solve_board(hybrid_agent, env, render, heuristics, timeout_s):
    state = env.get_current_observation()

    forward_model = hybrid_agent.skill_model
    n_bfs_steps = 20
    bfs_batch_size = 1024

    t0 = time.time()
    try:
        initial_state = env.get_binary_symbolic_state(state)
        target_state = env.binary_symbolic_target_state
        action_sequence = bfs_search_timeout(
            initial_state=initial_state,
            target_state=target_state,
            forward_model=forward_model,
            n_steps=n_bfs_steps,
            batch_size=bfs_batch_size,
            device=hybrid_agent._device,
            fringe_size_limit=MAX_FRINGE_SIZE,
            heuristics=heuristics,
            timeout_s=timeout_s,
        )

        if action_sequence is None:
            raise PlanningFailedError

        planning_time = time.time() - t0
        print(f"Symbolic action sequence: {action_sequence}, time {planning_time}")
        episodes, success_lowlevel = apply_action_sequence(
            env, hybrid_agent, action_sequence, render
        )
        # check if target state is reached
        success_all = env.is_solved()

        result = {
            "action_sequence": action_sequence,
            "success_lowlevel": success_lowlevel,
            "success_all": success_all,
            "episodes": episodes,
            "planning_time": planning_time,
        }

    except PlanningFailedError:
        result = {"planning_failed": True, "fail_cause": "planning_failed"}
    except FringeSizeExceededError:
        result = {"planning_failed": True, "fail_cause": "fringe_size_exceeded"}
    except FunctionTimedOut:
        result = {"planning_failed": True, "timeout": True, "fail_cause": "timeout"}
    return result


def solve_seeded_board(
    hybrid_agent,
    results_dir,
    filename_stem,
    env,
    do_replanning,
    render,
    heuristics,
    timeout_s,
    info=None,
):

    results_file = results_dir.joinpath(filename_stem + ".pkl")
    render_dir = results_dir.joinpath(filename_stem)

    done = True

    if not results_file.is_file():
        done = False

    if render and not render_dir.is_dir():
        done = False

    if done:
        print("All evaluations already done")
        return

    if info is None:
        info = {}

    solve_fcn = solve_board_replanning_timeout if do_replanning else solve_board
    result = solve_fcn(
        hybrid_agent=hybrid_agent,
        env=env,
        render=render,
        heuristics=heuristics,
        timeout_s=timeout_s,
    )
    result.update(info)
    print(result)
    with open(results_file, "wb") as handle:
        pickle.dump(result, handle)

    if "episodes" in result and render:
        os.makedirs(render_dir, exist_ok=True)
        save_renderings(result["episodes"], render_dir)


def save_renderings(episodes, render_dir):
    for episode_idx, episode in enumerate(episodes):
        for step_idx, transition in enumerate(episode):
            rendering = transition["rendering"]
            im = Image.fromarray(rendering)
            im.save(render_dir.joinpath(f"ep{episode_idx:03}_s{step_idx:03}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=False)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--envseed", nargs="+", type=int, required=True)
    parser.add_argument("--binary_state", nargs="+", type=str, required=False)
    parser.add_argument("--maxsoldepth", nargs="+", type=int, required=False)
    parser.add_argument("--replan", action="store_true")
    parser.add_argument("--envnoise", type=str, default="none")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument("--heuristics", nargs="+", type=str)

    args = parser.parse_args()

    run_directory = Path(args.run_dir)

    checkpoint_step = args.ckpt
    if checkpoint_step == "last":
        checkpoint_dir = run_directory.joinpath("checkpoints")
        checkpoint_steps = [int(f.stem.split("_")[1]) for f in checkpoint_dir.iterdir()]
        checkpoint_step = max(checkpoint_steps)
        print(f"Using checkpoint at {checkpoint_step}")
    else:
        checkpoint_step = int(checkpoint_step)

    results_dir = run_directory.joinpath("eval_rollouts")
    os.makedirs(results_dir, exist_ok=True)
    print("Results are at {}".format(results_dir))

    device = args.device
    config = load_config(run_directory)
    env_kwargs = dict(**config["env_kwargs"])

    if args.render and "jaco" in config["env_name"].lower():
        env_kwargs["rendering_enabled"] = True

    if args.envnoise == "lowlevel":
        noise_info = {"noise_cursor_std": 0.05, "noise_boardstate_ber": 0}
    elif args.envnoise == "none":
        noise_info = {}
    else:
        raise NotImplementedError
    env_kwargs.update(noise_info)
    noise_info["envnoise"] = args.envnoise

    if "ext_timelimit_obs" not in config:
        config["ext_timelimit_obs"] = True

    env_name = config["env_name"]
    reset_split = "test"
    proto_env = load_env(
        env_name,
        reset_split,
        env_kwargs,
        time_limit=True,
        ext_timelimit_obs=config["ext_timelimit_obs"],
        prototype=True,
    )

    hybrid_agent, _, checkpoint_data = load_agent(
        run_directory,
        checkpoint_step,
        env_name,
        proto_env,
        device=device,
        return_checkpoint_data=True,
    )

    if args.binary_state is not None:
        assert args.maxsoldepth is None
        sol_depth_or_bin_state = args.binary_state
        init_binary_state = True
    else:
        assert args.maxsoldepth is not None
        sol_depth_or_bin_state = args.maxsoldepth
        init_binary_state = False

    heuristics = []
    if not (args.heuristics is None or args.heuristics == "none"):
        raise NotImplementedError

    for env_seed, sol_depth_or_bin_state in product(
        args.envseed, sol_depth_or_bin_state
    ):
        if init_binary_state:
            # max. solution depth irrelevant
            env_kwargs["max_solution_depth"] = 1
        else:
            env_kwargs["max_solution_depth"] = sol_depth_or_bin_state
        env_kwargs["random_solution_depth"] = False

        env = load_env(
            env_name,
            reset_split,
            env_kwargs,
            time_limit=True,
            ext_timelimit_obs=config["ext_timelimit_obs"],
            prototype=False,
        )
        env.seed(env_seed)
        env.reset()

        if init_binary_state:
            assert all([s in ["0", "1"] for s in sol_depth_or_bin_state])
            binary_state = np.array([bool(int(s)) for s in sol_depth_or_bin_state])
            binary_state = binary_state.reshape(env.binary_symbolic_shape)
            print(f"Initializing binary state with {binary_state}")
            env.set_binary_symbolic_state(binary_state)

        if init_binary_state:
            init_id = f"bstate={sol_depth_or_bin_state}_"
        else:
            init_id = f"soldepth={sol_depth_or_bin_state}_"

        if heuristics:
            heuristics_str = "_".join(heuristics)
        else:
            heuristics_str = "none"

        filename_stem = (
            f"ckpt={checkpoint_step}_"
            f"seed={env_seed}_" + init_id + f"replan={args.replan}_"
            f"envnoise={args.envnoise}_heuristics={heuristics_str}"
        )

        info = {
            "ckpt_step": checkpoint_step,
            "replanning": args.replan,
            "env_seed": env_seed,
            "heuristics": heuristics,
            "init_binary_state": init_binary_state,
            "env_interactions_lpfm": checkpoint_data["env_interactions_lpfm"],
            "env_interactions_sac": checkpoint_data["env_interactions_sac"],
        }
        if init_binary_state:
            info["binary_state"] = sol_depth_or_bin_state
        else:
            info["max_solution_depth"] = sol_depth_or_bin_state

        info.update(noise_info)

        solve_seeded_board(
            hybrid_agent,
            results_dir,
            filename_stem,
            env,
            args.replan,
            args.render,
            heuristics,
            timeout_s=args.timeout_s,
            info=info,
        )


if __name__ == "__main__":
    main()
