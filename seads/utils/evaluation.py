"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import pandas as pd


def interpolate_env_steps(
    data_df, value_column_name, max_env_steps, seeds_to_retain=None, n_interp_points=100
):
    from scipy.interpolate import interp1d

    # Interpolate data given for disjoint sets of "env_steps" to a set of common
    # "interp_env_steps" with "interp_value_column_name".
    # The interpolation domain is decided by the runs which have the
    # highest initial seed / smallest terminal seed.
    # Optionally, only the 'seeds_to_retain' longest-running run_seeds may be retained.
    # data_df is assumed to have columns [run_seed env_steps <value_column>]
    # returns a dataframe with columns [run_seed interp_env_steps interp_<value_column>]

    train_steps_per_seed = data_df.groupby(["run_seed"])["env_steps"].max()
    seeds_avail = len(train_steps_per_seed)

    if seeds_to_retain is None:
        df_f = data_df
    elif isinstance(seeds_to_retain, str) and seeds_to_retain.startswith("farthest_"):
        n_to_ret = int(seeds_to_retain.split("_")[1])
        seeds_to_retain = list(train_steps_per_seed.nlargest(n_to_ret).index)
        df_f = data_df[data_df["run_seed"].isin(seeds_to_retain)]
    elif isinstance(seeds_to_retain, list):
        pass
        df_f = data_df[data_df["run_seed"].isin(seeds_to_retain)]
    else:
        raise ValueError

    initial_steps_per_seed = df_f.groupby(["run_seed"])["env_steps"].min()
    final_steps_per_seed = df_f.groupby(["run_seed"])["env_steps"].max()
    first_step_all = int(initial_steps_per_seed.max())
    last_step_all = int(final_steps_per_seed.min())
    if max_env_steps:
        last_step_all = min(max_env_steps, last_step_all)

    interp_x = np.linspace(first_step_all, last_step_all, n_interp_points)
    interp_y_array = []
    for seed in df_f["run_seed"].unique():
        df_f_s = df_f[df_f["run_seed"] == seed]
        df_f_s = df_f_s.sort_values("env_steps")
        np_array = df_f_s[["env_steps", value_column_name]].to_numpy()
        # np_array: col 0: env_steps, col 1: value_column_name
        interp_fcn = interp1d(
            np_array[:, 0],
            np_array[:, 1],
            kind="zero",
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_y = interp_fcn(interp_x)
        interp_y_array.append(interp_y)
    interp_y_array = np.stack(interp_y_array, axis=0)
    # interp_y_array: [seed x step_idx] -> interp_value

    df_x = pd.DataFrame(interp_x)
    df_x.columns = ["interp_env_steps"]
    df_x.index.names = ["step_idx"]
    df_y = pd.DataFrame(interp_y_array).unstack().reset_index()
    df_y.columns = ["step_idx", "run_seed", "interp_" + value_column_name]
    df_joined = df_y.join(df_x, on="step_idx")
    return df_joined, seeds_avail


def plot_interpolated_performance(
    ax,
    data_df,
    run_stem,
    value_column_name,
    max_env_steps=None,
    seeds_to_retain=None,
    label=None,
    plot_kwargs=None,
    plot_percent=False,
    legend_lists=None,
):
    df_f = data_df[data_df["run_stem"] == run_stem]
    interp_df, seeds_avail = interpolate_env_steps(
        df_f, value_column_name, max_env_steps, seeds_to_retain
    )

    n = "interp_" + value_column_name
    mean_ = np.array(
        interp_df.groupby("interp_env_steps")[n]
        .mean()
        .reset_index()
        .sort_values("interp_env_steps")
    )
    std_ = np.array(
        interp_df.groupby("interp_env_steps")[n]
        .std()
        .reset_index()
        .sort_values("interp_env_steps")
    )
    min_ = np.array(
        interp_df.groupby("interp_env_steps")[n]
        .min()
        .reset_index()
        .sort_values("interp_env_steps")
    )
    max_ = np.array(
        interp_df.groupby("interp_env_steps")[n]
        .max()
        .reset_index()
        .sort_values("interp_env_steps")
    )

    interp_x = mean_[:, 0]
    assert np.array_equal(interp_x, std_[:, 0])
    assert np.array_equal(interp_x, min_[:, 0])
    assert np.array_equal(interp_x, max_[:, 0])

    mean_ = mean_[:, 1]
    std_ = std_[:, 1]
    min_ = min_[:, 1]
    max_ = max_[:, 1]

    if plot_percent:
        mean_ = 100 * mean_
        std_ = 100 * std_
        max_ = 100 * max_
        min_ = 100 * min_

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    lp = ax.plot(interp_x, mean_, label=label, **plot_kwargs)
    f = ax.fill_between(interp_x, min_, max_, alpha=0.2, color=lp[0].get_color())

    if legend_lists is not None:
        legend_lists[0].append((lp[0], f))  # artists
        legend_lists[1].append(label)  # labels

    if max_env_steps:
        ax.set_xlim(0, 1.05 * max_env_steps)
    else:
        ax.set_xlim(0, 1.05 * interp_x[-1])

    print(run_stem, seeds_avail)

    return {
        "last_step": interp_x[-1],
        "last_mean": mean_[-1],
        "last_std": std_[-1],
        "last_min": min_[-1],
        "last_max": max_[-1],
        "seeds_avail": seeds_avail,
    }
