env="DoorKey-8x8" # BlockedUnlockPickup
int_rew_source="AEGIS"
seed=0
num_processes=1
# pretraining
pretrain_percentage=0.2
# reward learning
reward_learning_frequency=-1 # (-1 for human feedback)
episode_num=64
chunk_size=64
fps=5
# training
total_steps=500_000
int_rew_momentum=0.9
rnd_err_norm=1
int_rew_coef=1e-2

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
          --traj_overwrite=True \


# PYTHONPATH=./ python3 src/train.py \
#           --run_id=$seed \
#           --num_processes=$num_processes \
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