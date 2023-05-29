"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np


def set_style(plt, fig_scale=1):
    small_size = 8 * fig_scale
    medium_size = 10 * fig_scale
    bigger_size = 10 * fig_scale

    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title

    tex_fonts = {
        "text.usetex": True,
        "font.family": "sans-serif",
        "figure.constrained_layout.use": True,
        "lines.markersize": 0.5 * np.sqrt(20),
        # "grid.linewidth": 0.5,
        "lines.linewidth": 0.5 * 1,
        "savefig.dpi": 300,
    }

    plt.rcParams.update(tex_fonts)
