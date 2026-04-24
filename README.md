# SurgicAI: A Fine-grained Platform for Data Collection and Benchmarking in Surgical Policy Learning

## Prerequistes
This section introduces the necessary configuration you need.
### System Requirements
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange?style=flat-square&logo=ubuntu) ![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0.0-blue?style=flat-square&logo=github) ![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.7.1-green?style=flat-square&logo=python) ![ROS (any version)](https://img.shields.io/badge/ROS-Noetic-blue?style=flat-square&logo=ros) ![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python) ![Torch](https://img.shields.io/badge/Torch-2.10.0-red?style=flat-square&logo=pytorch) ![ambf](https://img.shields.io/badge/ambf-3.0-yellow?style=flat-square&logo=github) ![d3rlpy](https://img.shields.io/badge/d3rlpy-2.8.1-purple?style=flat-square&logo=python)

### Installation
* Install the [Surgical Robotics Challenge environment](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) (SRC), [Asynchronous Multi-Body Framework](https://github.com/WPI-AIM/ambf/wiki/Installing-AMBF) (AMBF), and ROS, as per the instructions in the AMBF link. SRC provides the simulated suturing phantom and da Vinci Surgical System. AMBF provides the fine-grained simulation platform and ROS bridges SurgicAI with AMBF.
```
git clone https://github.com/surgical-robotics-ai/surgical_robotics_challenge
```
* Install Gymnasium: [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) is a branch and updated version of OpenAI Gym. It provides standard API for the communication between the simulated environment and learning algorithms.
```
pip install gymnasium
```

* Configure the [Pytorch](https://pytorch.org/) and [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (if equipped with NVIDIA card) based on your hardware.

* Install [Stable Baseline3](https://github.com/DLR-RM/stable-baselines3) (SB3) and [d3rlpy](https://github.com/takuseno/d3rlpy): SB3 and d3rlpy are open-sourced Python libraries providing implementations of state-of-the-art RL algorithms. In this project, they are used to interaction with Gymnasium environment and offering interface for training, evaluating, and testing RL models.
```
pip install stable-baselines3[extra] d3rlpy
```

## Supported Tasks
SurgicAI supports five fine-grained surgical tasks for benchmarking policy learning. More tasks can be easily defined and added to the pipeline.

*Approach: Moving the tool to the proximity of the target.

*Place: Precisely positioning the needle at the entry point.

*Insert: Driving the needle through the simulated tissue.

*Handoff: Transferring the needle from one surgical tool to another.

*Pullout: Extracting the needle from the tissue exit point.

## Reinforcement Learning (RL) Training
This section introduce the basic procedure for model training with defined Gymnasium environment.

### Run the SRC Environment
Make sure ROS and SRC is running before moving forward to the following steps. You can simply run the following command or refer to this [link](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) for details.

```
cd ~/surgical_robotics_challenge
./run_env_3D_MED_COMPLEX_LND_420006.sh
```

### Low-level Training
In order to train a model specifically for a low level policy, you can directly run with the command below from the base SurgicAI directory:
```python
python3 RL/RL_training_online.py \
    --algorithm "$algorithm" \
    --task_name "$task" \
    --reward_type "$REWARD_TYPE" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --save_freq "$SAVE_FREQ" \
    --seed "$SEED" \
    --trans_error "$TRANS_ERROR" \
--angle_error "$ANGLE_ERROR"
``` 

More common arguments are described in RL_training_online.py and cli_args.py, including IL data sources,logging destinations, GUI enable/disable, and more.

### Model Evaluation
The command evaluates the success rate, trajectory length, and time steps across five policies with different random seeds.
```python
python3 RL/Model_evaluation.py \
    --algorithm "$algorithm" \
    --task_name "$task" \
    --reward_type "$REWARD_TYPE" \
    --trans_error "$TRANS_ERROR" \
    --angle_error "$ANGLE_ERROR" \
    --eval_seed "$EVAL_SEED"
```

## Image-based Imitation Learning (Image-IL)

This module provides the pipeline for training and evaluating imitation learning policies using visual observations (camera feeds).

### Dependency: R3M
For visual feature extraction, we utilize **[R3M (Reusable Representations for Robotic Manipulation)](https://github.com/facebookresearch/r3m)**. R3M is a robust pre-trained visual encoder that enables the agent to learn effective policies from camera frames by leveraging representations learned from large-scale human video datasets.

To install R3M, navigate to the submodule directory and install it in editable mode:
```bash
cd Image_IL/r3m
pip install -e .
```

### Training
Use the training scripts to map camera observations to robot actions. You can configure the task, camera view, and randomization parameters via command-line arguments:
```python
python3 Image_IL/<your_algo_name>_train.py \
    --task_name "$TASK_NAME" \
    --view_name "$VIEW_NAME" \
    --trans_error "$TRANS_ERROR" \
    --angle_error "$ANGLE_ERROR"
```

### Evaluation
To evaluate the success rate and trajectory efficiency of a trained Image-IL policy, run:
```
python3 Image_IL/Task_evaluation_<your_algo_name>.py \
    --task_name "$TASK_NAME" \
    --view_name "$VIEW_NAME" \
    --trans_error "$TRANS_ERROR" \
    --angle_error "$ANGLE_ERROR"
```

## Environment variables (recommended)
Many scripts default to repo-relative paths, but you can override locations via:

- `SURGICAI_ROOT`: repo root (only needed if running scripts from outside the repo)
- `SURGICAI_DATA_DIR`: base directory for RL experiment outputs (checkpoints, final models)
- `SURGICAI_IL_DATA_DIR`: base directory for Image-IL training datasets
- `SURGICAI_IL_OUT_DIR`: base directory for Image-IL outputs (models/results)
- `SURGICAI_POLICY_DIR`: directory containing pretrained low-level `.zip` policies (used by `RL/Low_env_init.py`)
- `SURGICAI_ENV_INFO_DIR`: directory containing env info txt files (defaults to `RL/Env_info/`)
- `SURGICAI_APPROACH_VISDR_DIR`: directory for vis-DR image/transition recording outputs (defaults under `SURGICAI_DATA_DIR`)

## High-level Training
See in [High_level_HLP.ipynb](./RL/High_level_HLP.ipynb) for more details.

The following video demonstrates the complete suturing procedure by our training policy.

[demo](https://github.com/surgical-robotics-ai/SurgicAI/assets/147576462/1927a1cf-096f-444d-a878-6c0f96b152d4)

Here's some progress demonstrating our pipeline's transition to the latest SRC, focusing on the low-level task: 'Place'.

[New_SRC_demo](https://github.com/user-attachments/assets/faf0d821-2b6c-4524-be26-565dc2f4a600)

If you find our work userful, please cite it as:
```bibtex
@misc{wu2024surgicaifinegrainedplatformdata,
      title={SurgicAI: A Fine-grained Platform for Data Collection and Benchmarking in Surgical Policy Learning}, 
      author={Jin Wu and Haoying Zhou and Peter Kazanzides and Adnan Munawar and Anqi Liu},
      year={2024},
      eprint={2406.13865},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2406.13865}, 
}


