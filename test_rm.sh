env="BlockedUnlockPickup"
int_rew_source="AEGIS__pilot"
group_name="RM_5e-3_decay"
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
curr_iter=0
total_iterations=1
train_for=$(echo "(${total_steps//_/} * (1.0 - $pretrain_percentage) / ${total_iterations//_/}) / 1" | bc)

reward_learning_frequency=-1 # (-1 for human feedback)
reward_learning_rate=3e-2

# Alignment training
rm_rew_coef=5e-3
merge_rm_true_rew=True
int_rew_decay=True

for seed in 0 1 2 3 4 5 6 7 8 9; do
    PYTHONPATH=./ python3 src/train_rm.py \
                --group_name=$group_name \
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
                --reward_learning_rate=$reward_learning_rate \
                --curr_iter=$curr_iter \
                --train_for=$train_for
done

# for seed in 0 1 2 3 4 5 6 7 8 9; do
#     PYTHONPATH=./ python3 src/train.py \
#             --group_name=$group_name \
#             --run_id=$seed \
#             --num_processes=$pretraining_num_processes \
#             --total_steps=$total_steps \
#             --pretrain_percentage=$pretrain_percentage \
#             --int_rew_source=$int_rew_source \
#             --env_source=minigrid \
#             --game_name=$env \
#             --exp_group_name=$exp_group_name \
#             --features_dim=64 \
#             --model_features_dim=64 \
#             --latents_dim=128 \
#             --model_latents_dim=128 \
#             --int_rew_coef=$int_rew_coef \
#             --rm_rew_coef=$rm_rew_coef \
#             --int_rew_momentum=$int_rew_momentum \
#             --rnd_err_norm=$rnd_err_norm \
#             --reward_learning_frequency=$reward_learning_frequency \
#             --merge_rm_true_rew=$merge_rm_true_rew \
#             --int_rew_decay=$int_rew_decay \
#             --curr_iter=$curr_iter \
#             --train_for=$train_for
# done