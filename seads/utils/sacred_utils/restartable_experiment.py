"""
This file implements restartable 'sacred' experiments

This source file contains modified or unmodified code excerpts from the directory
https://github.com/IDSIA/sacred,
commit `98a2d08dc8ac9b6330f81de555635b03a33749a1`, licensed under the MIT License.
A copy of the license is given in `SACRED_LICENSE` in this directory.
The above repository is abbreviated by `<SACRED>` in the following.
Code excerpts taken from this repository are marked case-by-case.
"""

import argparse
import functools
import inspect
import json
import os.path
import re
import sys
import time
from glob import glob
from pathlib import Path
from shutil import copyfile, move
from typing import Optional

from docopt import docopt
from sacred import Experiment, cli_option
from sacred.arg_parser import get_config_updates
from sacred.observers import FileStorageObserver
from sacred.run import Run
from sacred.utils import (
    SacredError,
    SacredInterrupt,
    ensure_wellformed_argv,
    format_sacred_error,
    print_filtered_stacktrace,
)


class WaitingForRestartInterrupt(SacredInterrupt):
    STATUS = "WAITING_FOR_RESTART"
    RETURN_CODE = 3


class InitializationFinishedInterrupt(SacredInterrupt):
    STATUS = "INITIALIZATION_FINISHED"
    RETURN_CODE = 0


class RestartableFileStorageObserver(FileStorageObserver):
    """
    Extends the :class:`FileStorageObserver` by capability to continue
    an existing run in WAITING_FOR_RESTART status. It does this
    by archiving the old run.json and config.json files.
    For this observer to work, the run id must be defined.

    See also :class:`FileStorageObserver`.
    """

    def __init__(self, *args, **kwargs):
        super(RestartableFileStorageObserver, self).__init__(*args, **kwargs)
        self.ready_to_start = False

    # code from <SACRED>/sacred/observers/file_storage.py, modified
    def _make_run_dir(self, _id):
        os.makedirs(self.basedir, exist_ok=True)
        self.dir = None
        assert _id is not None
        self.dir = os.path.join(self.basedir, str(_id))
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)

    def _archive_file(self, file_name):
        source_file = os.path.join(self.dir, file_name)
        for _id in range(1, 1000):
            target_file = os.path.join(self.dir, file_name + "." + str(_id))
            print(f"Probing file {target_file}")
            if not os.path.isfile(target_file):
                move(source_file, target_file)
                return
            time.sleep(0.1)
        raise RuntimeError(f"Cannot archive {file_name}")

    def _archive_previous_run(self):
        # keep cout.txt and metrics.json
        # archive run.json and config.json
        if self.ready_to_start:
            return
        run_file = os.path.join(self.dir, "run.json")
        if os.path.isfile(run_file):
            # there is previous data, check if we are waiting for restart
            # and archive old run.json / config.json
            with open(run_file, "r") as f:
                run_info = json.load(f)
            # if run_info["status"] == "RUNNING":
            #    raise RuntimeError("Cannot restart RUNNING experiment")
            self._archive_file("run.json")
            self._archive_file("config.json")
            self.ready_to_start = True
        else:
            # no previous data exists
            if any(
                os.path.isfile(os.path.join(self.dir, f))
                for f in ["run.json", "metrics.json", "cout.txt", "config.json"]
            ):
                raise RuntimeError()
            self.ready_to_start = True

    def queued_event(
        self, ex_info, command, host_info, queue_time, config, meta_info, _id
    ):
        self._make_run_dir(_id)
        self._archive_previous_run()
        return super().queued_event(
            ex_info, command, host_info, queue_time, config, meta_info, _id
        )

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        self._make_run_dir(_id)
        self._archive_previous_run()
        return super().started_event(
            ex_info, command, host_info, start_time, config, meta_info, _id
        )


class PrefixFileStorageObserver(FileStorageObserver):
    """
    Extends the :class:`FileStorageObserver` by an optional prefix
    for the run id. Thus, run ids are of the form "prefix".M,
    where M is an increasing number starting at 1.

    See also :class:`FileStorageObserver`.
    """

    def __init__(self, *, run_id_prefix=None, **kwargs):
        super(PrefixFileStorageObserver, self).__init__(**kwargs)
        self.run_id_prefix = run_id_prefix

    # code from <SACRED>/sacred/observers/file_storage.py, modified
    def _maximum_existing_run_id(self):
        if self.run_id_prefix is None:
            existing = [
                int(d)
                for d in os.listdir(self.basedir)
                if os.path.isdir(os.path.join(self.basedir, d)) and d.isdigit()
            ]
        else:
            prefix_dirs = [
                s
                for s in os.listdir(self.basedir)
                if os.path.isdir(os.path.join(self.basedir, s))
                and re.match(rf"{self.run_id_prefix}\.\d+", s)
            ]
            existing = [int(s.split(".")[1]) for s in prefix_dirs]
        if existing:
            return max(existing)
        else:
            return 0

    # code from <SACRED>/sacred/observers/file_storage.py, modified
    def _make_dir(self, _id):
        if self.run_id_prefix is not None:
            _id = str(self.run_id_prefix + "." + str(_id))
        new_dir = os.path.join(self.basedir, str(_id))
        os.mkdir(new_dir)
        self.dir = new_dir  # set only if mkdir is successful

    # code from <SACRED>/sacred/observers/file_storage.py, unmodified
    def _make_run_dir(self, _id):
        os.makedirs(self.basedir, exist_ok=True)
        self.dir = None
        if _id is None:
            fail_count = 0
            _id = self._maximum_existing_run_id() + 1
            while self.dir is None:
                try:
                    self._make_dir(_id)
                except FileExistsError:  # Catch race conditions
                    if fail_count < 1000:
                        fail_count += 1
                        _id += 1
                    else:  # expect that something else went wrong
                        raise
        else:
            self.dir = os.path.join(self.basedir, str(_id))
            os.mkdir(self.dir)


@cli_option("-z", "--force-id")
def force_id(args, run):
    run._id = args


class RestartableExperiment(Experiment):
    """
    The RestartableExperiment replaces the `main` and `automain`
    decorators of :class:`Experiment` by `on_init` and `on_continue`
    decorators. The function decorated by `on_init` will be called
    when the experiment is run for the first time. `on_continue`
    is called when an interrupted or failed experiment is continued.
    Both are captured functions.

    To initially start the experiment, the command line syntax
    follows the standard sacred syntax. A run id will be automatically assigned or
    can be set manually by --force-id=N.
    To re-start a failed or interrupted experiment, the command line syntax
    is::

        $ python ... restart_from=N [--comment=]

    or::

        $ python ... restart_base=N [--comment=]

    where N is the experiment to be continued.
    Restarted experiments can currently not be made unobserved.
    As run config, the config from the initial run will be inherited.

    When using restart_from, a restarted experiment will get a *new* run id,
    which is of the form N.M, where N is the run id of the experiment to continue, and M is
    an increasing number. Prior to the restart, tensorboard event files
    and 'metrics.json' will be copied to the run directory of the restarted experiment.
    Also, files in directories given by `artifact_directories` will be
    linked to the run directory of the restarted experiment.
    The 'run_directory' property will point to the new run directory ('N.M') while
    'parent_run_directory' will point to directory 'N'.

    When using restart_base, the run id will be preserved. The previous
    'run.json' and 'config.json' will be archived by appending
    an increasing integer to the filename. New metrics and couts will
    be appended to the respective files. Both properties
    'run_directory' and 'parent_run_directory' will point to the same
    directory as no "child" run directories are created.

    Currently, RestartableExperiment only supports a single
    :class:`FileStorageObserver`. Restarting is only possible
    if the run terminated with the :class:`WaitingForRestartInterrupt`
    or :class:`InitializationFinishedInterrupt`  interrupts.

    Example code ::

        experiment = RestartableExperiment(observer_base_directory="\\tmp")

        @experiment.on_init
        def experiment_init(batchsize):
            print("Initial run")
            raise WaitingForRestartInterrupt
            # or raise InitializationFinishedInterrupt

        @experiment.on_continue
        def experiment_continue(batchsize):
            print("Restarted run")
            if not finished:
                # raise this after, e.g., a fixed
                # number of iterations in your program
                raise WaitingForRestartInterrupt
            else:
                exit(0)

        if __name__ == "__main__":
            experiment.run_commandline()

    Example launch ::

        # Initial invocation
        python script.py with batchsize=32

        # Restart run 1
        python script.py restart_from=1
    """

    def __init__(
        self,
        observer_base_directory: str,
        enable_copy_event_files: bool = True,
        enable_copy_metrics: bool = True,
        artifact_directories: tuple = ("checkpoints",),
        enable_link_artifacts: bool = True,
        **kwargs,
    ):
        """
        Initialize `RestartableExperiment`

        Args:
            observer_base_directory: Base directory of `FileStorageObserver`
            enable_copy_event_files: Copy event files to restarted run
            enable_copy_metrics: Copy metrics.json to restarted run
            artifact_directories: Tuple of directores with artifacts
            enable_link_artifacts: Link files in artifact_directories to restarted run
            **kwargs: Parameters of `Experiment`
        """
        cli_options = [force_id]
        additional_cli_options = kwargs.get("additional_cli_options", None)
        if additional_cli_options:
            cli_options.extend(additional_cli_options)
        kwargs["additional_cli_options"] = cli_options
        host_info = []
        additional_host_info = kwargs.get("additional_host_info", None)
        if additional_host_info:
            host_info.extend(additional_host_info)
        kwargs["additional_host_info"] = host_info

        super(RestartableExperiment, self).__init__(**kwargs)
        self.observer_base_directory = observer_base_directory
        self.enable_copy_event_files = enable_copy_event_files
        self.enable_copy_metrics = enable_copy_metrics
        self.artifact_directories = artifact_directories
        self.enable_link_artifacts = enable_link_artifacts

        self._register_main = super().main
        self._on_init_fcn = None
        self._on_continue_fcn = None
        self._parent_run_id = None
        self._file_storage_observer = None
        self.observers = None

    @property
    def run_directory(self):
        if self.current_run.unobserved:
            path_exists = True
            increment = 0
            while path_exists:
                start_time_str = self.current_run.start_time.strftime(
                    "%Y-%m-%dT%H-%M-%S-%fZ"
                )
                run_dir = Path(
                    self._file_storage_observer.basedir,
                    "unobserved",
                    start_time_str + "_" + str(increment),
                )
                increment += 1
                path_exists = run_dir.exists()
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        else:
            return Path(self._file_storage_observer.dir)

    @property
    def parent_run_directory(self):
        return self._get_run_dir(self._parent_run_id)

    def on_init(self, function):
        """
        Decorates a function which is called when the experiment
        is freshly initialized
        """
        if self._on_init_fcn is not None:
            raise ValueError("Can only decorate a single function")
        self._on_init_fcn = function

    def on_continue(self, function):
        """
        Decorates a function which is called when the experiment
        is asked to continue.
        """
        if self._on_continue_fcn is not None:
            raise ValueError("Can only decorate a single function")
        self._on_continue_fcn = function

    def main(self, function):
        raise RuntimeError(
            "@main decoration not allowed. Use @on_init and @on_continue."
        )

    def automain(self, function):
        raise RuntimeError(
            "@automain decoration not allowed. "
            "Use @on_init and @on_continue and "
            "call ex.run_commandline() to start "
            "the experiment."
        )

    def _parse_args(self, argv):
        if len(argv) <= 1:
            return None
        restart_from_match = re.match(r"restart_from=(.+)", argv[1])
        restart_base_match = re.match(r"restart_base=(.+)", argv[1])
        if restart_from_match or restart_base_match:
            parser = argparse.ArgumentParser(description="Restart experiment")
            parser.add_argument("--comment", type=str, required=False)
            args, unknown_args = parser.parse_known_args(argv[2:])
            if len(unknown_args) > 0:
                raise ValueError("Invalid restart attempt.")
            if restart_from_match:
                args.restart_id = restart_from_match.group(1)
                args.restart_type = "restart_from"
            elif restart_base_match:
                args.restart_id = restart_base_match.group(1)
                args.restart_type = "restart_base"
            else:
                raise RuntimeError
            return args
        else:
            return None

    def _check_restart_preliminaries(self, parent_run_id):
        run_dir = os.path.join(self.observer_base_directory, parent_run_id)
        run_json_path = os.path.join(run_dir, "run.json")
        with open(run_json_path, "r") as f:
            run_info = json.load(f)
        if run_info["status"] == "FAILED":
            raise RuntimeError(
                "You tried to restart a run which has FAILED. \n If you have fixed the error's cause, you may either: \n"
                f"    (1) manually change the status in {run_json_path} from FAILED to WAITING_FOR_RESTART,\n"
                f" or (2) delete the run directory at {run_dir}, re-initialize, and re-start the run."
            )

    def _register_prefix_observer(self, parent_run_id):
        self._file_storage_observer = PrefixFileStorageObserver(
            basedir=self.observer_base_directory, run_id_prefix=parent_run_id
        )
        self.observers = tuple([self._file_storage_observer])

    def _register_restart_observer(self, parent_run_id):
        self._file_storage_observer = RestartableFileStorageObserver(
            basedir=self.observer_base_directory
        )
        self.observers = tuple([self._file_storage_observer])

    def _register_default_observer(self):
        self._file_storage_observer = FileStorageObserver(
            basedir=self.observer_base_directory
        )
        self.observers = tuple([self._file_storage_observer])

    def _get_run_dir(self, run_id):
        return Path(self.observer_base_directory).joinpath(str(run_id))

    def _copy_event_files(self, parent_run_id, child_run_id, _log):
        parent_dir = self._get_run_dir(parent_run_id)
        child_dir = self._get_run_dir(child_run_id)
        event_files = glob(str(parent_dir.joinpath("events.out.tfevents.*")))
        for source_filename in event_files:
            source_filename = Path(source_filename)
            dest_filename = child_dir.joinpath(source_filename.name)
            if source_filename.is_file():
                if dest_filename.is_file():
                    raise RuntimeError(f"Destination {dest_filename} already exists")
                _log.info(f"Copy event files {source_filename} to {dest_filename}")
                copyfile(source_filename, dest_filename)

    def _copy_metrics(self, parent_run_id, child_run_id, _log):
        parent_dir = self._get_run_dir(parent_run_id)
        child_dir = self._get_run_dir(child_run_id)
        source_filename = parent_dir.joinpath("metrics.json")
        dest_filename = child_dir.joinpath("metrics.json")
        if dest_filename.is_file():
            raise RuntimeError(f"Destination {dest_filename} already exists")
        copyfile(source_filename, dest_filename)
        _log.info(f"Copy metrics file from {source_filename} to {dest_filename}")

    def _link_artifacts(self, parent_run_id, child_run_id, _log):
        parent_dir = str(self._get_run_dir(parent_run_id))
        child_dir = str(self._get_run_dir(child_run_id))
        # create directory structure, but link files
        for (source_dir, _, files) in os.walk(parent_dir):
            # ignore root directory
            if source_dir == parent_dir:
                continue
            # strip off 'parent_dir/'
            relative_source_dir = source_dir[len(parent_dir) + len(os.sep) :]
            # check if first component of relative_source_dir is in self.artifact_directories
            if relative_source_dir.split(os.sep)[0] in self.artifact_directories:
                dest_dir = os.path.join(child_dir, relative_source_dir)
                if not os.path.isdir(dest_dir):
                    os.makedirs(dest_dir)
                if len(files) > 0:
                    _log.info(
                        f"Linking {len(files)} files from {source_dir} to {dest_dir}"
                    )
                for f in files:
                    source_file = os.path.join(source_dir, f)
                    dest_file = os.path.join(dest_dir, f)
                    rel_source_file = os.path.relpath(source_file, dest_dir)
                    os.symlink(rel_source_file, dest_file)

    def _parent_config_location(self, parent_run_id):
        config_location = os.path.join(
            self.observer_base_directory, parent_run_id, "config.json"
        )
        if not os.path.isfile(config_location):
            raise RuntimeError(f"Parent config does not exist at {config_location}")
        return config_location

    def _pre_continue_fcn(self, parent_id, _log, _run):
        child_run_id = _run._id
        assert child_run_id is not None
        if self.enable_copy_event_files:
            self._copy_event_files(parent_id, child_run_id, _log)
        if self.enable_copy_metrics:
            self._copy_metrics(parent_id, child_run_id, _log)
        if self.enable_link_artifacts:
            self._link_artifacts(parent_id, child_run_id, _log)

    def _pre_continue_decorator(self, func):
        signature = inspect.signature(func)
        params = [p for p in signature.parameters.values()]
        original_param_names = [p.name for p in params]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            parent_id = kwargs["parent_id"]
            _log = kwargs["_log"]
            _run = kwargs["_run"]
            self._pre_continue_fcn(parent_id, _log, _run)
            if "parent_id" not in original_param_names:
                kwargs.pop("parent_id")
            if "_log" not in original_param_names:
                kwargs.pop("_log")
            if "_run" not in original_param_names:
                kwargs.pop("_run")
            return func(*args, **kwargs)

        # add 'parent_id' and '_run' to kwarg-only parameters of 'wrapper'
        if "parent_id" not in original_param_names:
            params.append(
                inspect.Parameter("parent_id", kind=inspect.Parameter.KEYWORD_ONLY)
            )
        if "_log" not in original_param_names:
            params.append(
                inspect.Parameter("_log", kind=inspect.Parameter.KEYWORD_ONLY)
            )
        if "_run" not in original_param_names:
            params.append(
                inspect.Parameter("_run", kind=inspect.Parameter.KEYWORD_ONLY)
            )
        new_signature = signature.replace(parameters=params)
        # inspect.signature will look for __signature__ first
        wrapper.__signature__ = new_signature
        return wrapper

    def _build_restart_from_args(self, args, parent_run_id):
        config_location = self._parent_config_location(parent_run_id)
        # sacred's run_commandline expects the program name in argv
        restart_args = ["dummy"]
        if args.comment:
            restart_args += '--comment="' + args.comment + '"'
        restart_args.extend(["with", config_location, f"parent_id={parent_run_id}"])
        return restart_args

    def _build_restart_base_args(self, args, parent_run_id):
        config_location = self._parent_config_location(parent_run_id)
        # sacred's run_commandline expects the program name in argv
        restart_args = ["dummy", f"--force-id={parent_run_id}"]
        if args.comment:
            restart_args += '--comment="' + args.comment + '"'
        restart_args.extend(["with", config_location])
        return restart_args

    # code from <SACRED>/sacred/experiment.py, modified
    def _run_commandline_base(self, argv=None) -> Optional[Run]:
        """
        Run the command-line interface of this experiment.

        If ``argv`` is omitted it defaults to ``sys.argv``.

        Parameters
        ----------
        argv
            Command-line as string or list of strings like ``sys.argv``.

        Returns
        -------
        The Run object corresponding to the finished run.

        """
        argv = ensure_wellformed_argv(argv)
        short_usage, usage, internal_usage = self.get_usage()
        args = docopt(internal_usage, [str(a) for a in argv[1:]], help=False)

        cmd_name = args.get("COMMAND") or self.default_command
        config_updates, named_configs = get_config_updates(args["UPDATE"])

        err = self._check_command(cmd_name)
        if not args["help"] and err:
            print(short_usage)
            print(err)
            sys.exit(1)

        if self._handle_help(args, usage):
            sys.exit()

        try:
            return self.run(
                cmd_name,
                config_updates,
                named_configs,
                info={},
                meta_info={},
                options=args,
            )
        except Exception as e:
            if self.current_run:
                debug = self.current_run.debug
            else:
                # The usual command line options are applied after the run
                # object is built completely. Some exceptions (e.g.
                # ConfigAddedError) are raised before this. In these cases,
                # the debug flag must be checked manually.
                debug = args.get("--debug", False)

            if debug:
                # Debug: Don't change behavior, just re-raise exception
                raise
            elif self.current_run and self.current_run.pdb:
                # Print exception and attach pdb debugger
                import pdb
                import traceback

                traceback.print_exception(*sys.exc_info())
                pdb.post_mortem()
            else:
                # Handle pretty printing of exceptions. This includes
                # filtering the stacktrace and printing the usage, as
                # specified by the exceptions attributes
                if isinstance(e, SacredError):
                    print(format_sacred_error(e, short_usage), file=sys.stderr)
                else:
                    print_filtered_stacktrace()
                # The two lines below are the only line changed compared to the original code
                return_code = 1 if not hasattr(e, "RETURN_CODE") else e.RETURN_CODE
                sys.exit(return_code)
                # was: sys.exit(1)

    def run_commandline(self, argv=None):
        if argv is None:
            argv = sys.argv
        args = self._parse_args(argv)
        if args is not None and args.restart_type == "restart_from":
            parent_run_id = args.restart_id
            self._parent_run_id = parent_run_id
            self._check_restart_preliminaries(parent_run_id)
            experiment_argv = self._build_restart_from_args(args, parent_run_id)
            self._register_prefix_observer(parent_run_id)
            super().main(self._pre_continue_decorator(self._on_continue_fcn))
        elif args is not None and args.restart_type == "restart_base":
            parent_run_id = args.restart_id
            self._parent_run_id = parent_run_id
            self._check_restart_preliminaries(parent_run_id)
            experiment_argv = self._build_restart_base_args(args, parent_run_id)
            self._register_restart_observer(parent_run_id)
            super().main(self._on_continue_fcn)
        else:
            # launch on_init_fcn as experiment main
            experiment_argv = argv
            self._register_default_observer()
            super().main(self._on_init_fcn)

        self._run_commandline_base(experiment_argv)
