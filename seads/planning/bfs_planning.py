"""
Copyright 2022 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.md file in the root directory of this source tree.
"""

import sys

import numpy as np
import torch


def pack_state(state):
    assert state.dtype == bool
    state_flat = state.reshape(-1)
    state_packed = np.packbits(state_flat)
    return state_packed


def unpack_state_batch(state_batch_packed, state_shape):
    assert state_batch_packed.dtype == np.uint8
    assert state_batch_packed.ndim == 2
    state_unpacked = np.unpackbits(state_batch_packed, axis=1)
    numel = np.prod(state_shape)
    state_unpacked = state_unpacked[:, :numel].reshape(
        state_batch_packed.shape[0], *state_shape
    )
    return state_unpacked.astype(bool)


def states_to_fingerprints(states):
    """
    Turn a batch of symbolic states into a batch of fingerprints

    Parameters
    -------------
    states: np.ndarray, shape [B x <N>]
        The states to encode

    Returns
    -------------
    fingerprints: list[str]
        The fingerprints
    """
    assert states.dtype == bool
    states_flat = states.reshape(states.shape[0], -1)
    state_packed = np.packbits(states_flat, axis=1)
    fingerprints = []
    for row in state_packed:
        fingerprints.append(row.tobytes().hex())
    return fingerprints


def trace_back_actions(explored, target_fingerprint):
    # Traverse through 'explored' until root node is reached
    actions = []
    prev_fingerprint = target_fingerprint
    while prev_fingerprint is not None:
        action, prev_fingerprint = explored[prev_fingerprint]
        actions.append(action)
    return list(reversed(actions[:-1]))


class PlanningFailedError(Exception):
    pass


class FringeSizeExceededError(Exception):
    pass


def bfs_search(
    initial_state,
    target_state,
    forward_model,
    n_steps,
    batch_size,
    device="cuda",
    fringe_size_limit=None,
    heuristics=None,
):
    """
    Given a lights out board, "initial_board", apply BFS to find the shortest
    sequence of actions to turn off all lights.

    Parameters
    -------------
    initial_state: torch.BoolTensor or np.ndarray, shape [<N>]
        The initial symbolic (binary) state
    target_state: torch.BoolTensor or np.ndarray, shape [<N>]
        The target symbolic (binary) state
    forward_model: `torch.nn.Module`
        The forward model to advance the states
    n_steps: int
        How deep the BFS should go before terminating
    batch_size: int
        The number of boards that are fed into the NLM model at once
    device: str
        The device the NLM model should be evaluated on
    fringe_size_limit: int
        Maximal fringe size, in bytes
    heuristics: list
        List of heuristics to apply. Currently, no heuristic is implemented.
    """
    if isinstance(initial_state, np.ndarray):
        assert initial_state.dtype == bool
    elif isinstance(initial_state, torch.Tensor):
        assert initial_state.dtype == torch.bool
        initial_state = initial_state.cpu().numpy()
    else:
        raise ValueError

    if isinstance(target_state, np.ndarray):
        assert target_state.dtype == bool
    elif isinstance(target_state, torch.Tensor):
        assert target_state.dtype == torch.bool
        target_state = target_state.cpu().numpy()
    else:
        raise ValueError

    if heuristics is None:
        heuristics = []

    if len(heuristics) > 0:
        raise NotImplementedError

    forward_model = forward_model.to(device)
    n_actions = forward_model.num_skills

    initial_fingerprint = states_to_fingerprints(
        np.array(
            [
                initial_state,
            ]
        )
    )[0]
    target_fingerprint = states_to_fingerprints(
        np.array(
            [
                target_state,
            ]
        )
    )[0]

    explored = {initial_fingerprint: (None, None)}
    state_shape = initial_state.shape
    initial_state_packed = pack_state(initial_state)
    fringe = [torch.from_numpy(initial_state_packed)]
    action_single = torch.arange(n_actions, dtype=torch.int64)

    for step in range(n_steps):
        if len(fringe) == 0:
            raise PlanningFailedError

        new_fringe = []
        new_fringe_size = 0

        actions = action_single.repeat(len(fringe))

        for batch_states_packed, batch_actions in zip(
            torch.split(
                torch.repeat_interleave(torch.stack(fringe, dim=0), n_actions, dim=0),
                batch_size,
            ),
            torch.split(actions, batch_size),
        ):

            batch_states = torch.from_numpy(
                unpack_state_batch(batch_states_packed.cpu().numpy(), state_shape)
            )
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            with torch.no_grad():
                next_states = forward_model.predict(batch_states, batch_actions).mode
                assert next_states.dtype == torch.bool

            next_states = next_states.cpu()
            next_fingerprints = states_to_fingerprints(next_states.cpu().numpy())
            origin_fingerprints = states_to_fingerprints(batch_states.cpu().numpy())

            if target_fingerprint in next_fingerprints:
                idx = next_fingerprints.index(target_fingerprint)
                explored[target_fingerprint] = (
                    int(batch_actions[idx]),
                    origin_fingerprints[idx],
                )
                return trace_back_actions(explored, target_fingerprint)

            for origin_fingerprint, next_state, action, next_fingerprint in zip(
                origin_fingerprints, next_states, batch_actions, next_fingerprints
            ):
                if next_fingerprint not in explored:
                    explored[next_fingerprint] = (
                        int(action),
                        origin_fingerprint,
                    )
                    next_state_packed = torch.from_numpy(pack_state(next_state.numpy()))
                    new_fringe.append(next_state_packed)
                    new_fringe_size += sys.getsizeof(next_state_packed)
                    if fringe_size_limit and new_fringe_size > fringe_size_limit:
                        raise FringeSizeExceededError

        print(f"Step {step}, fringe size {new_fringe_size}")

        fringe = new_fringe

    raise PlanningFailedError
