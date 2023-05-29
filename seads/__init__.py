"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

from pathlib import Path

from seads.utils.patch_mode import patch_mode

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent.joinpath("experiments")

try:
    import seads_envs
except ImportError as e:
    raise ImportError(
        "The 'seads_envs' package could not be located. "
        "Please clone the repository at 'https://github.com/EmbodiedVision/seads-environments/' "
        "and see its README for installation instructions."
    )

patch_mode()
