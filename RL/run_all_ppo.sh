#!/bin/bash

TIMESTEPS=150000
RANDON="--randomization_params \"1,1,1,1,1\""

run_exp() {
  local ALGO=$1
  local TASK=$2
  local REWARD=$3
  local SEED=$4
  local RAND=$5
  local TRANS_ERROR=$6
  local ANGLE_ERROR=$7

  if [ "$RAND" == "on" ]; then
    CMD="python3 RL_training_online.py \
      --algorithm \"$ALGO\" \
      --task_name \"$TASK\" \
      --reward_type \"$REWARD\" \
      --total_timesteps \"$TIMESTEPS\" \
      --trans_error \"$TRANS_ERROR\" \
      --angle_error \"$ANGLE_ERROR\" \
      --seed $SEED \
      $RANDON"
  else
    CMD="python3 RL_training_online.py \
      --algorithm \"$ALGO\" \
      --task_name \"$TASK\" \
      --reward_type \"$REWARD\" \
      --total_timesteps \"$TIMESTEPS\" \
      --trans_error \"$TRANS_ERROR\" \
      --angle_error \"$ANGLE_ERROR\" \
      --seed $SEED"
  fi

  echo ">>> Running: $CMD"
  eval $CMD
}

# ==============================================
# PPO Experiments — All Tasks / Rewards / Seeds
# ==============================================

TASKS=(
  "Approach 1 10"
  "Place 5 10"
  "Insert 5 10"
  "Pullout 2 15"
  "Regrasp 5 20"
)

REWARDS=("dense" "sparse")
RANDS=("on" "off")
SEEDS=(1 10 100 1000 10000)

for TASK_INFO in "${TASKS[@]}"; do
  read TASK TRANS_ERROR ANGLE_ERROR <<< "$TASK_INFO"
  for REWARD in "${REWARDS[@]}"; do
    for RAND in "${RANDS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        run_exp "PPO" "$TASK" "$REWARD" "$SEED" "$RAND" "$TRANS_ERROR" "$ANGLE_ERROR"
      done
    done
  done
done
