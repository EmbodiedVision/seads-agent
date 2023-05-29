import random

import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ReplayMemorySerializable:
    # Code author: Jan Achterhold

    def __init__(self, state_dim, action_dim, capacity, seed):
        self.buffer = {
            "state": np.empty((capacity, state_dim)),
            "action": np.empty((capacity, action_dim)),
            "reward": np.empty((capacity,)),
            "next_state": np.empty((capacity, state_dim)),
            "done": np.empty((capacity,), dtype=bool),
        }
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer["state"][self.position] = state
        self.buffer["action"][self.position] = action
        self.buffer["reward"][self.position] = reward
        self.buffer["next_state"][self.position] = next_state
        self.buffer["done"][self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch_indices = self.rng.randint(0, self.size, batch_size)
        state = np.stack([self.buffer["state"][idx] for idx in batch_indices])
        action = np.stack([self.buffer["action"][idx] for idx in batch_indices])
        reward = np.stack([self.buffer["reward"][idx] for idx in batch_indices])
        next_state = np.stack([self.buffer["next_state"][idx] for idx in batch_indices])
        done = np.stack([self.buffer["done"][idx] for idx in batch_indices])
        return state, action, reward, next_state, done

    def __len__(self):
        return self.size

    def load_from_file(self, filename):
        load_dict = np.load(filename, allow_pickle=True)
        self.load_from_dict(load_dict["arr_0"].item()["buf"])

    def load_from_dict(self, load_dict):
        if not load_dict["capacity"] == self.capacity:
            raise ValueError("Mismatching data file")
        self.buffer = {
            "state": load_dict["state"],
            "action": load_dict["action"],
            "reward": load_dict["reward"],
            "next_state": load_dict["next_state"],
            "done": load_dict["done"],
        }
        self.position = load_dict["position"]
        self.size = load_dict["size"]
        self.rng = load_dict["rng"]

    def save_to_dict(self):
        save_dict = dict(
            **self.buffer,
            capacity=self.capacity,
            position=self.position,
            size=self.size,
            rng=self.rng,
        )
        return save_dict

    def save_to_file(self, filename):
        save_dict = self.save_to_dict()
        np.savez(filename, {"buf": save_dict})
