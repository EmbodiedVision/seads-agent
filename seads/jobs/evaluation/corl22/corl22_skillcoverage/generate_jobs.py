"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

from itertools import product
from pathlib import Path

from seads import EXPERIMENT_DIR

this_dir = Path(__file__).parent.resolve()


ENV = [
    "lightsout_cursor",
    "tileswap_cursor",
    "lightsout_reacher",
    "tileswap_reacher",
    "lightsout_jaco",
    "tileswap_jaco",
    "lightsout_jaco_stairs",
]
VARIANTS = [
    "default",
]
RUN_SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

commands = []
for env, variant, run_seed in product(ENV, VARIANTS, RUN_SEED):
    if ("vic" in variant) and ("cursor" not in env):
        continue

    if env == "lightsout_jaco_stairs" and variant != "default":
        continue

    run_prefix = "corl22"
    run_id = f"{run_prefix}_{env}_{variant}_s{run_seed}"
    run_dir = EXPERIMENT_DIR.joinpath("seads", run_id)
    if not run_dir.is_dir():
        print(f"Run dir {run_dir} does not exist!")
        continue

    if "cursor" in env:
        CKPT_STEP = 100
    else:
        CKPT_STEP = 500

    command = (
        "python -m seads.evaluate_skill_coverage "
        f'--run_dir "{run_dir}" --ckpt_step {CKPT_STEP} \n'
    )
    commands.append(command)

# write commands to job file
with open(this_dir.joinpath("job_list.txt"), "w") as handle:
    handle.writelines(commands)
