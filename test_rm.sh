env="BlockedUnlockPickup"
int_rew_source="AEGIS"
seed=0
exp_group_name="GGgroup"

# pretraining
pretrain_percentage=0.5
pretraining_num_processes=8
total_steps=1_000_000
int_rew_momentum=0.9
rnd_err_norm=1
int_rew_coef=1e-2

# reward learning
start_iter=0
total_iterations=1
reward_learning_frequency=-1 # (-1 for human feedback)
reward_learning_rate=3e-2

# Alignment training
rm_rew_coef=5e-3
merge_rm_true_rew=True
int_rew_decay=True

# Trajectory generation
episode_num=5 # Episode videos to generate per iteration (100 for human feedback, 5 for testing)

# # PRETRAINING/TRAINING PHASE
train_for=$(echo "(${total_steps//_/} * (1.0 - $pretrain_percentage) / ${total_iterations//_/}) / 1" | bc)

for ((curr_iter=$start_iter; curr_iter<total_iterations; curr_iter++)); do
    echo "Starting iteration $curr_iter, generating $episode_num trajectories, training for $train_for steps"

    # PYTHONPATH=./ python3 src/train_rm.py \
    #         --run_id=$seed \
    #         --num_processes=$pretraining_num_processes \
    #         --total_steps=$total_steps \
    #         --pretrain_percentage=$pretrain_percentage \
    #         --int_rew_source=$int_rew_source \
    #         --env_source=minigrid \
    #         --game_name=$env \
    #         --exp_group_name=$exp_group_name \
    #         --features_dim=64 \
    #         --model_features_dim=64 \
    #         --latents_dim=128 \
    #         --model_latents_dim=128 \
    #         --int_rew_coef=$int_rew_coef \
    #         --rm_rew_coef=$rm_rew_coef \
    #         --int_rew_momentum=$int_rew_momentum \
    #         --rnd_err_norm=$rnd_err_norm \
    #         --reward_learning_frequency=$reward_learning_frequency \
    #         --reward_learning_rate=$reward_learning_rate \
    #         --curr_iter=$curr_iter \
    #         --train_for=$train_for \
    #         --episode_num=$episode_num \
    
    PYTHONPATH=./ python3 src/train.py \
            --run_id=$seed \
            --num_processes=$pretraining_num_processes \
            --total_steps=$total_steps \
            --pretrain_percentage=$pretrain_percentage \
            --int_rew_source=$int_rew_source \
            --env_source=minigrid \
            --game_name=$env \
            --exp_group_name=$exp_group_name \
            --features_dim=64 \
            --model_features_dim=64 \
            --latents_dim=128 \
            --model_latents_dim=128 \
            --int_rew_coef=$int_rew_coef \
            --rm_rew_coef=$rm_rew_coef \
            --int_rew_momentum=$int_rew_momentum \
            --rnd_err_norm=$rnd_err_norm \
            --reward_learning_frequency=$reward_learning_frequency \
            --merge_rm_true_rew=$merge_rm_true_rew \
            --int_rew_decay=$int_rew_decay \
            --curr_iter=$curr_iter \
            --train_for=$train_for \
            --episode_num=$episode_num
done