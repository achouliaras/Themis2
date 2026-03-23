#!/bin/bash
set -e

env="BlockedUnlockPickup" # BlockedUnlockPickup
int_rew_source="AEGIS"
seed=0
exp_group_name="cgroup"

# pretraining
pretrain_percentage=0.5
pretraining_num_processes=8

# reward learning
total_iterations=1
reward_learning_frequency=-1 # (-1 for human feedback)

# Trajectory generation
episode_num=16 # Episode videos to generate per iteration (100 for human feedback, 5 for testing)
chunk_size=32 # Video generation batch size
fps=2 # Frames per second for generated videos (Without xai videos 5 is ok for minigrid, with xai videos 2 is better for easier understanding)
use_xai_videos=True
xai_method="integrated_gradients" # "saliency", "integrated_gradients", "input_x_gradient", "guided_backprop", "deconvolution" "gradient_shap" "deep_lift" "deeplift_shap" "grad_cam"

# Human Labelling
sampling_strategy="trueskill" # "Uniform", "SwissInfoGain", "trueskill"
pair_num=8 # Pairs to generate (relevant for uniform sampling)
notifications=True # Wether to send email notifications to annotators about new iterations and rounds (only relevant for human feedback setting)
edit_videos_num_processes=8

# training
total_steps=1_000_000
int_rew_momentum=0.9
rnd_err_norm=1
int_rew_coef=1e-2

# # PRETRAINING PHASE
PYTHONPATH=./ python3 src/train.py \
          --run_id=$seed \
          --num_processes=$pretraining_num_processes \
          --total_steps=$total_steps \
          --pretrain_percentage=$pretrain_percentage \
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

# train_for=$total_steps * (1 - $pretrain_percentage) // $total_iterations
train_for=$(echo "(${total_steps//_/} * (1.0 - $pretrain_percentage) / ${total_iterations//_/}) / 1" | bc)

for ((curr_iter=0; curr_iter<total_iterations; curr_iter++)); do
    echo "Starting iteration $curr_iter, generating $episode_num trajectories, training for $train_for steps"

    PYTHONPATH=./ python3 src/generate_trajectories.py \
              --run_id=$seed \
              --num_processes=1 \
              --int_rew_source=$int_rew_source \
              --env_source=minigrid \
              --game_name=$env \
              --features_dim=64 \
              --model_features_dim=64 \
              --latents_dim=128 \
              --model_latents_dim=128 \
              --reward_learning_frequency=$reward_learning_frequency \
              --episode_num=$episode_num \
              --chunk_size=$chunk_size \
              --fps=$fps \
              --gen_xai_videos=$use_xai_videos \
              --xai_method=$xai_method \
              --traj_overwrite=True \

    # PYTHONPATH=./ python3 src/video_pipeline.py \
    #           --run_id=$seed \
    #           --env_source=minigrid \
    #           --game_name=$env \
    #           --exp_group_name=$exp_group_name \
    #           --notifications=$notifications \
    #           --pair_num=$pair_num \
    #           --int_rew_source=$int_rew_source \
    #           --sampling_strategy=$sampling_strategy \
    #           --video_processing_mode="SideBySide" \
    #           --num_processes=$edit_videos_num_processes \
    #           --add_xai_videos=$use_xai_videos \
    #           --traj_overwrite=True \
    #           --curr_iter=$curr_iter \

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
    #           --episode_num=$episode_num \
done



