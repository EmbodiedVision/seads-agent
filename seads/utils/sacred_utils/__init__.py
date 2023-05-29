"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Union

import sacred

from seads.utils.sacred_utils.restartable_experiment import (
    InitializationFinishedInterrupt,
    RestartableExperiment,
    WaitingForRestartInterrupt,
)

__all__ = [
    "create_experiment",
    "WaitingForRestartInterrupt",
    "InitializationFinishedInterrupt",
    "load_json",
]


def create_experiment(
    experiment_name,
    run_base_dir,
    src_dir,
    mongodb_observer=False,
    zip_src_dir=True,
    restartable=False,
    add_src_to_sacred_recursively=False,
    add_caller_src_to_sacred=True,
) -> Union[sacred.Experiment, RestartableExperiment]:
    """
    Create a sacred experiment.
    The final run directory (run_directory) will be at
        * `run_base_dir`/<RUN_ID> for observed runs
        * `run_base_dir`/unobserved/<DATE_TIME> for unobserved runs.

    Parameters
    ----------
    experiment_name : str
        Experiment name
    run_base_dir : str or pathlib.Path
        Base directory for runs
    src_dir : str or pathlib.Path
        Source directory (for archiving)
    mongodb_observer: bool
        Register a MongoDB observer on the newly created experiment
    zip_src_dir : bool
        If True, `src_dir` is tar-gzed to <run_directory>/source.tgz.
    restartable : bool
        If True, creates a RestartableExperiment
    add_src_to_sacred_recursively: bool
        All .py files in `src_dir` will be added as source files recursively.
        This can be very slow for many source files.
    add_caller_src_to_sacred: bool
        Add source file of caller of this function to sacred

    Returns
    -------
    experiment : sacred.Experiment
        Sacred experiment
    """
    run_base_dir = Path(run_base_dir)
    src_dir = Path(src_dir)

    if restartable:
        experiment = RestartableExperiment(
            observer_base_directory=run_base_dir, name=experiment_name
        )
        if mongodb_observer:
            raise NotImplementedError(
                "A restartable experiment cannot be observed by a MongoDB observer"
            )
        # 'run_directory'/'parent_run_directory' property is provided by 'RestartableExperiment'
    else:
        experiment = sacred.Experiment(name=experiment_name)
        experiment.observers.append(sacred.observers.FileStorageObserver(run_base_dir))
        if mongodb_observer:
            raise NotImplementedError

        experiment._get_run_dir = _get_run_dir_fcn(experiment, run_base_dir)
        sacred.Experiment.run_directory = property(lambda self: self._get_run_dir())

    def pre_run_hook():
        _zip_source_dir(src_dir, experiment.run_directory)

    if zip_src_dir:
        experiment.pre_run_hook(pre_run_hook)

    if add_src_to_sacred_recursively:
        additional_files_list = []
        for filename in src_dir.glob("**/*"):
            if not filename.is_file():
                continue
            if filename.suffix not in [".py"]:
                continue
            experiment.add_source_file(str(filename))
            additional_files_list.append(filename)
        print(f"Added {len(additional_files_list)} additional source files")

    if add_caller_src_to_sacred:
        import inspect

        caller_filename = inspect.stack()[1].filename
        experiment.add_source_file(caller_filename)

    return experiment


def _get_run_dir_fcn(ex, run_base_dir):
    def _fcn(_run):
        if _run.unobserved:
            start_time_str = _run.start_time.strftime("%Y-%m-%dT%H-%M-%S-%fZ")
            run_dir = Path(run_base_dir, "unobserved", start_time_str)
            os.makedirs(run_dir, exist_ok=True)
            return run_dir
        else:
            # directory is created by sacred
            # noinspection PyProtectedMember
            return Path(run_base_dir, str(_run._id))

    return ex.capture(_fcn)


def _zip_source_dir(src_dir, run_dir):
    target_file = Path(run_dir).joinpath("source.tgz")
    if not target_file.exists():
        cmd = f'tar -c -z -f "{target_file}" -C "{str(src_dir.resolve())}" .'
        print(cmd)
        os.system(cmd)


def load_json(json_file, max_tries=10):
    """
    Load JSON file from run, e.g. config.json or run.json

    When attempting to load a run-related JSON file while the
    run is still running, IO errors can happen due to concurrent
    read/write operations. This causes a `JSONDecodeError`.
    This method tries to load the file multiple times until no
    decoding error occurs.

    Args:
        json_file: `str`
            Path to JSON file
        max_tries: `int`
            Maximum number of tries to load the JSON file, default: 10.

    Returns:
        data: `dict`
            JSON data
    """
    n_tries = 0
    while n_tries < max_tries:
        n_tries += 1
        try:
            with open(json_file, "r") as handle:
                data = json.load(handle)
            break
        except json.decoder.JSONDecodeError as e:
            if n_tries >= max_tries:
                warnings.warn(f"JSON file not readable after {n_tries} attempts")
                raise e
            else:
                warnings.warn(
                    f"JSON file not readable (attempt {n_tries}/{max_tries}), "
                    "try again in 0.5s..."
                )
                time.sleep(0.5)
    return data


def log_scalar(run, summary_writer, tag, value, step, print_=False):
    """
    Log scalar value to sacred run and summary writer.
    """
    run.log_scalar(tag, value, step)
    summary_writer.add_scalar(tag, value, step)
    if print_:
        print(step, tag, value)
