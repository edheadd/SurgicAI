import sys
import time
import os
from RL.Domain_randomization.Domain_callback import DomainRandomizationCallback

class DummyEnv:
    def __init__(self):
        self.unwrapped = self
    def reset(self):
        print("DummyEnv: reset called")

# Dummy Randomization message for testing if ROS is not available


if __name__ == "__main__":
    print("Testing DomainRandomizationCallback...")
    env = DummyEnv()
    callback = DomainRandomizationCallback(env, randomization_args="0,0,0,1,1", seed=123)
    print("Calling randomize() directly...")
    while True:
        callback.randomize()

    print("Test complete.")
