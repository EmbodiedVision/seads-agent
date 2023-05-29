"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pickle

import torch


class ArgumentDict:
    def __init__(self, args):
        self.args = args

    def __getattr__(self, item):
        if item in self.args:
            return self.args[item]
        raise AttributeError

    def __getitem__(self, item):
        return self.args[item]


# We patch pickle.Unpickler to be able to load old checkpoints with different module naming
class RenamingUnpickler(pickle.Unpickler):
    IS_RENAMING_UNPICKLER = True

    def find_class(self, module, name):
        new_module, new_name = module, name
        if module == "physics_planning.data.common" and name == "Episode":
            new_module = "seads.common"
        elif module.startswith("physics_planning.envs"):
            new_module = "seads." + ".".join(module.split(".")[1:])
        elif module.startswith("seads.envs"):
            new_module = "seads_envs." + ".".join(module.split(".")[2:])
        return super().find_class(new_module, new_name)


def torch_load(*args, **kwargs):
    if not hasattr(pickle.Unpickler, "IS_RENAMING_UNPICKLER"):
        pickle.Unpickler = RenamingUnpickler
    return torch.load(*args, **kwargs)


def pickle_load(file, **kwargs):
    if not hasattr(pickle.Unpickler, "IS_RENAMING_UNPICKLER"):
        pickle.Unpickler = RenamingUnpickler
    return pickle.Unpickler(file, **kwargs).load()
