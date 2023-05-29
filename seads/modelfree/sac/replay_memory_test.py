from tempfile import NamedTemporaryFile
from unittest import TestCase

import numpy as np

from .replay_memory import ReplayMemorySerializable


class TestReplayMemorySerializable(TestCase):
    # Code author: Jan Achterhold

    def test_push(self):
        state_dim = 32
        action_dim = 4
        capacity = 64
        seed = 1
        replay_memory = ReplayMemorySerializable(state_dim, action_dim, capacity, seed)
        rng = np.random.RandomState(42)
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for _ in range(2 * capacity):
            state = rng.rand(state_dim)
            next_state = rng.rand(state_dim)
            action = rng.rand(action_dim)
            reward = rng.rand()
            done = bool(rng.randint(2))
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            replay_memory.push(state, action, reward, next_state, done)

        fields = ["state", "action", "reward", "next_state", "done"]
        # buffer should retain last "capacity" elements
        for f in fields:
            np.testing.assert_equal(
                np.stack(locals()[f"{f}s"][-capacity:]), replay_memory.buffer[f]
            )

        # size should not exceed capacity
        self.assertTrue(replay_memory.size == capacity)

        # save to temporary file
        with NamedTemporaryFile() as tmp_file:
            replay_memory.save_to_file(tmp_file.name + ".npz")

            # load from file
            replay_memory_loaded = ReplayMemorySerializable(
                state_dim, action_dim, capacity, seed
            )
            replay_memory_loaded.load_from_file(tmp_file.name + ".npz")

            assert replay_memory_loaded.size == capacity
            assert replay_memory_loaded.position == replay_memory.position

            for f in fields:
                np.testing.assert_equal(
                    np.stack(locals()[f"{f}s"][-capacity:]),
                    replay_memory_loaded.buffer[f],
                )
