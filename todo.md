Changes since 3/2025
- Added ros-based domain randomization plugin to change visual and physical properties of the AMBF simulator environments. Parameters can be adjusted in the plugin.
- Updated IL module for domain randomized recording and training.
- Updated and refactored RL module to be ROS2 compatible, be independent of Surgical Robotics Challenge, provide an interface for the model more similar to the robot, and be better future-compatible.

Todo
- Refactor IL, currently not very clean. (small step)
- Add vision-only IL data collection and training. (big step)