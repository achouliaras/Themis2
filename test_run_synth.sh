env="DoorKey-8x8" # BlockedUnlockPickup
int_rew_source="AEGIS"
seed=0
# pretraining
pretrain_percentage=0.2
pretraining_num_processes=8
# reward learning
reward_learning_frequency=10000
traj_gen_num_processes=1
edit_videos_num_processes=8
episode_num=6
pair_num=8
exp_group_name="cgroup"
chunk_size=64
fps=5
# training
total_steps=500_000
int_rew_momentum=0.9
rnd_err_norm=1
int_rew_coef=1e-2

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




