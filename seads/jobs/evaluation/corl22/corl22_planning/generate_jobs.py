"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from seads import EXPERIMENT_DIR

this_dir = Path(__file__).parent.resolve()

base_dir = EXPERIMENT_DIR.joinpath("seads")

n_rollouts_per_depth = 20
seed_gen = np.random.RandomState(4343)


ENV = [
    "lightsout_cursor",
    "tileswap_cursor",
    "lightsout_reacher",
    "tileswap_reacher",
    "lightsout_jaco",
    "tileswap_jaco",
    "lightsout_jaco_stairs",
]
RUN_SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


MAX_SOL_DEPTH = range(1, 5 + 1)
ENV_SEED = [seed_gen.randint(0, int(1e8)) for _ in range(n_rollouts_per_depth)]
ENVNOISE = "none"
VARIANTS = [
    "default",
]
REPLAN = [True, False]


commands = []
for env, variant, run_seed, replan in tqdm(
    list(product(ENV, VARIANTS, RUN_SEED, REPLAN))
):

    run_prefix = "corl22"

    if "reacher" in env:
        CKPT_STEP = list(range(1000, 50_000 + 1, 1000))
    elif "jaco" in env:
        CKPT_STEP = list(range(1000, 50_000 + 1, 1000))
    else:
        CKPT_STEP = list(range(100, 10_000 + 1, 200))

    for ckpt in CKPT_STEP:

        run_id = f"{run_prefix}_{env}_{variant}_s{run_seed}"
        run_dir = base_dir.joinpath(run_id)

        if not run_dir.is_dir():
            print(f"Run dir not found: {run_dir}")

        ckpt_dir = run_dir.joinpath("checkpoints")

        if not ckpt_dir.joinpath(f"step_{ckpt}.pkl").exists():
            continue

        max_sol_depth_list = MAX_SOL_DEPTH

        # check if some eval rollouts already exist
        remaining_max_sol_depth = set()
        remaining_seeds = set()
        for max_sol_depth, env_seed in product(max_sol_depth_list, ENV_SEED):
            filename = (
                f"ckpt={ckpt}_"
                f"seed={env_seed}_"
                f"soldepth={max_sol_depth}_"
                f"replan={replan}_"
                f"envnoise={ENVNOISE}_heuristics=none.pkl"
            )
            path = run_dir.joinpath("eval_rollouts", filename)
            if not path.exists():
                # print(f"Evaluation {path} missing")
                remaining_max_sol_depth.add(max_sol_depth)
                remaining_seeds.add(env_seed)

        if len(remaining_max_sol_depth) == 0:
            continue

        command = (
            "python -m seads.solve_board "
            '--run_dir "{}" '
            "--ckpt {} "
            "--envseed {} "
            "--maxsoldepth {} "
            "--envnoise {}"
            "{} --timeout_s 60\n".format(
                run_dir.resolve(),
                ckpt,
                " ".join([str(s) for s in remaining_seeds]),
                " ".join([str(d) for d in remaining_max_sol_depth]),
                ENVNOISE,
                " --replan" if replan else "",
            )
        )
        commands.append(command)

# write commands to job file
with open(this_dir.joinpath("job_list.txt"), "w") as handle:
    handle.writelines(commands)
