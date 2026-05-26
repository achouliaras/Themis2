#!/bin/bash
set -e

env="BlockedUnlockPickup" # BlockedUnlockPickup
int_rew_source="AEGIS_B_group"
seed=0
group_name="XAI_EVAL"
exp_group_name="agroup"

# pretraining
pretrain_percentage=0.46
pretraining_num_processes=8

# reward learning
start_iter=0
curr_iter=0
total_iterations=1
reward_learning_frequency=-1 # (-1 for human feedback)

# Trajectory generation
traj_overwrite=True
episode_num=10 # Different Episode videos to generate
tries_per_episode=3 # How many times the agent tries to solve each episode

lock_env_run_id=False # Whether to lock the env seed across runs (to generate identical env setups across different agent seeds)
chunk_size=32 # Video generation batch size
fps=1 # Frames per second for generated videos
use_xai_videos=True
xai_method="integrated_gradients" # "saliency", "grad_cam", "integrated_gradients", "input_x_gradient", "guided_backprop", "deconvolution" "gradient_shap" "deep_lift" "deeplift_shap" 

# Human Labelling
sampling_strategy="custom" # "Uniform", "SwissInfoGain", "trueskill"
pair_num=8 # Pairs to generate (relevant for uniform sampling)
notifications=True # Wether to send email notifications to annotators about new iterations and rounds (only relevant for human feedback setting)
edit_videos_num_processes=8

# training
total_steps=1_000_000
int_rew_momentum=0.9
rnd_err_norm=1
int_rew_coef=1e-2

# # PRETRAINING PHASE
# PYTHONPATH=./ python3 src/train.py \
#           --run_id=$seed \
#           --num_processes=$pretraining_num_processes \
#           --total_steps=$total_steps \
#           --pretrain_percentage=$pretrain_percentage \
#           --int_rew_source=$int_rew_source \
#           --env_source=minigrid \
#           --game_name=$env \
#           --features_dim=64 \
#           --model_features_dim=64 \
#           --latents_dim=128 \
#           --model_latents_dim=128 \
#           --int_rew_coef=$int_rew_coef \
#           --int_rew_momentum=$int_rew_momentum \
#           --rnd_err_norm=$rnd_err_norm \
#           --reward_learning_frequency=$reward_learning_frequency \
#           --episode_num=$episode_num \

# train_for=$total_steps * (1 - $pretrain_percentage) // $total_iterations
# train_for=$(echo "(${total_steps//_/} * (1.0 - $pretrain_percentage) / ${total_iterations//_/}) / 1" | bc)

# PreTrain/Train model
# seed 0,1,4,5,7,9: 6,4
# seed 2,3,6,8    : 8,2
# Sum             : 68,32
# 48 rounds with 4 errors generate 2126 pairs 1 2 3 4 5 6 7 8 9

for seed in 0 1 2 3 4 5 6 7 8 9; do
    PYTHONPATH=./ python3 src/generate_trajectories.py \
              --group_name=$group_name \
              --run_id=$seed \
              --num_processes=$pretraining_num_processes \
              --total_steps=$total_steps \
              --int_rew_source=$int_rew_source \
              --env_source=minigrid \
              --game_name=$env \
              --features_dim=64 \
              --model_features_dim=64 \
              --latents_dim=128 \
              --model_latents_dim=128 \
              --int_rew_coef=$int_rew_coef \
              --int_rew_momentum=$int_rew_momentum \
              --rnd_err_norm=$rnd_err_norm \
              --reward_learning_frequency=$reward_learning_frequency \
              --episode_num=$episode_num \
              --tries_per_episode=$tries_per_episode \
              --lock_env_run_id=$lock_env_run_id \
              --chunk_size=$chunk_size \
              --fps=$fps \
              --gen_xai_videos=$use_xai_videos \
              --xai_method=$xai_method \
              --traj_overwrite=$traj_overwrite \
              --curr_iter=$curr_iter

# for ((curr_iter=$start_iter; curr_iter<total_iterations; curr_iter++)); do
#     # echo "Starting iteration $curr_iter, generating $episode_num trajectories, training for $train_for steps"

#     PYTHONPATH=./ python3 src/video_pipeline.py \
#               --run_id=$seed \
#               --env_source=minigrid \
#               --game_name=$env \
#               --exp_group_name=$exp_group_name \
#               --notifications=$notifications \
#               --pair_num=$pair_num \
#               --int_rew_source=$int_rew_source \
#               --sampling_strategy=$sampling_strategy \
#               --video_processing_mode="SideBySide" \
#               --num_processes=$edit_videos_num_processes \
#               --add_xai_videos=$use_xai_videos \
#               --curr_iter=$curr_iter

    # # Training with human feedback for one iteration
    # PYTHONPATH=./ python3 src/train.py \
    #           --run_id=$seed \
    #           --num_processes=$pretraining_num_processes \
    #           --total_steps=$total_steps \
    #           --pretrain_percentage=$pretrain_percentage \
    #           --int_rew_source=$int_rew_source \
    #           --env_source=minigrid \
    #           --game_name=$env \
    #           --features_dim=64 \
    #           --model_features_dim=64 \
    #           --latents_dim=128 \
    #           --model_latents_dim=128 \
    #           --int_rew_coef=$int_rew_coef \
    #           --int_rew_momentum=$int_rew_momentum \
    #           --rnd_err_norm=$rnd_err_norm \
    #           --reward_learning_frequency=$reward_learning_frequency \
    #           --curr_iter=$curr_iter \
    #           --train_for=$train_for \
    #           --episode_num=$episode_num
done