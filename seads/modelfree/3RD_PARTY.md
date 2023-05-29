The code in the `sac` directory is a modified version of

https://github.com/pranz24/pytorch-soft-actor-critic

at commit `1bd1158116edb02b09a8f8f361640eaeee84447e`, licensed under the MIT license, given in [sac/LICENSE](sac/LICENSE).

We have made the following modifications:

* `main.py`, `sac.py`: Changed the meaning of 'mask' ('not done') to 'done' for consistency
* `model.py`: Introduced `build_net` to allow arbitrary number of hidden layers in actor/critic models
* `model.py`: Implemented a `RelaxedBernoulliPolicy` for binary action spaces
* `replay_memory.py`, `replay_memory_test.py`: Implemented a `ReplayMemorySerializable` (and test) for storing the replay memory
* `sac.py`: Implemented `state_dict`, `load_state_dict` to (de)serialize the SAC agent

In addition, the code has been reformatted using the black formatter with version 20.8b1.
