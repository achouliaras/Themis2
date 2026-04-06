import click
import warnings

# Suppress Pydantic v2 field attribute warnings from dependencies
warnings.filterwarnings("ignore", message=".*attribute with value.*was provided to the.*Field.*function.*")

from concurrent.futures import ThreadPoolExecutor
import os, re, time, glob
import imageio
import numpy as np
import pandas as pd
import torch as th
import gymnasium as gym

from src.env.minigrid_envs import *
from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.utils.configs import TrainingConfig
from src.utils.enum_types import EnvSrc, XplainMethod
from src.utils.xai_utils import ValueNetworkWrapper, fetch_captum_explainer, generate_attribution_map
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import obs_as_tensor

def get_trajectory_ids(path, run_id):
    # Matches 'traj', then your run_id, then captures 3 digits, then '.mp4'
    pattern = re.compile(rf"^traj{run_id:02}(\d{{3}})\.mp4$")
    
    # 1. Extract all matching IDs into a set
    existing = {int(pattern.match(f).group(1)) for f in os.listdir(path) if pattern.match(f)}
    
    # 2. Identify missing IDs and the next available ID
    max_id = max(existing) if existing else -1
    missing = {i for i in range(max_id) if i not in existing}
    next_id = max_id + 1
    
    return existing, missing, next_id

def generate_trajectories(config):
    th.autograd.set_detect_anomaly(False)
    th.set_default_dtype(th.float32)
    th.backends.cudnn.benchmark = False

    if config.gen_xai_videos:
        traj_xai_videos_path = os.path.join(config.log_dir, "traj_xai_videos")
        os.makedirs(traj_xai_videos_path, exist_ok=True)
    traj_videos_path = os.path.join(config.log_dir, "traj_videos")
    os.makedirs(traj_videos_path, exist_ok=True)
    traj_data_path = os.path.join(config.log_dir, "traj_data")
    os.makedirs(traj_data_path, exist_ok=True)
    
    existing_video_ids = set()
    missing_video_ids = set()
    displacement = 0

    if config.traj_overwrite:
        # Remove existing trajectory videos and data if overwrite is enabled
        if config.gen_xai_videos:
             for filename in os.listdir(traj_xai_videos_path):
                file_path = os.path.join(traj_xai_videos_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        for filename in os.listdir(traj_videos_path):
            file_path = os.path.join(traj_videos_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        for filename in os.listdir(traj_data_path):
            file_path = os.path.join(traj_data_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    existing_video_ids, missing_video_ids, displacement = get_trajectory_ids(traj_videos_path, config.run_id)

    # Determine which trajectory video IDs need to be generated based on existing files and the overwrite setting
    gen_video_ids = missing_video_ids.union(set(range(displacement, displacement + config.episode_num)))
    gen_video_ids = sorted(gen_video_ids)[:config.episode_num] # Ensure we only generate the required number of episodes
    print(f"==Existing_video_ids: {existing_video_ids}\n==Displacement: {displacement}\n==Missing_video_ids: {missing_video_ids}\n==Gen_video_ids: {gen_video_ids}")
    
    callbacks = config.get_callbacks()
    optimizer_class, optimizer_kwargs = config.get_optimizer()
    activation_fn, cnn_activation_fn, reward_activation_fn = config.get_activation_fn()

    config.cast_enum_values()

    policy_features_extractor_class, \
        features_extractor_common_kwargs, \
        model_cnn_features_extractor_class, \
        model_features_extractor_common_kwargs = \
        config.get_cnn_kwargs(cnn_activation_fn)
    
    policy_kwargs = dict(
        run_id=config.run_id,
        n_envs=config.num_processes,
        activation_fn=activation_fn,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        features_extractor_class=policy_features_extractor_class,
        features_extractor_kwargs=features_extractor_common_kwargs,
        model_cnn_features_extractor_class=model_cnn_features_extractor_class,
        model_cnn_features_extractor_kwargs=model_features_extractor_common_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=config.max_grad_norm,
        model_features_dim=config.model_features_dim,
        latents_dim=config.latents_dim,
        model_latents_dim=config.model_latents_dim,
        policy_mlp_norm=config.policy_mlp_norm,
        model_mlp_norm=config.model_mlp_norm,
        model_cnn_norm=config.model_cnn_norm,
        model_mlp_layers=config.model_mlp_layers,
        use_status_predictor=config.use_status_predictor,
        gru_layers=config.gru_layers,
        policy_mlp_layers=config.policy_mlp_layers,
        policy_gru_norm=config.policy_gru_norm,
        use_model_rnn=config.use_model_rnn,
        model_gru_norm=config.model_gru_norm,
        total_timesteps=config.total_steps,
        n_steps=config.n_steps,
        int_rew_source=config.int_rew_source,
        icm_forward_loss_coef=config.icm_forward_loss_coef,
        ngu_knn_k=config.ngu_knn_k,
        ngu_dst_momentum=config.ngu_dst_momentum,
        ngu_use_rnd=config.ngu_use_rnd,
        rnd_err_norm=config.rnd_err_norm,
        rnd_err_momentum=config.rnd_err_momentum,
        rnd_use_policy_emb=config.rnd_use_policy_emb,
        dsc_obs_queue_len=config.dsc_obs_queue_len,
        log_dsc_verbose=config.log_dsc_verbose,
        aegis_nov_exp_mem_capacity = config.aegis_nov_exp_mem_capacity,
        aegis_knn_k=config.aegis_knn_k,
        aegis_dst_momentum=config.aegis_knn_k,
    )

    if config.gen_xai_videos:
        xai_kwargs = {}

        
    start_time = time.perf_counter()
    chunks = [gen_video_ids[i:i + config.chunk_size] for i in range(0, len(gen_video_ids), config.chunk_size)]
    for chunk in chunks:
        current_batch_size = len(chunk)
        wrapper_class = config.get_wrapper_class()
        env = config.get_env(wrapper_class, num_processes=len(chunk), seed=config.run_id*12345 + chunk[0] if config.fixed_seed >= 0 else None)

        trainer_kwargs = dict(
            policy=PPOModel,
            env=env,
            seed=config.run_id,
            run_id=config.run_id,
            can_see_walls=config.can_see_walls,
            image_noise_scale=config.image_noise_scale,
            total_timesteps=config.total_steps,
            n_steps=config.n_steps,
            n_epochs=config.n_epochs,
            model_n_epochs=config.model_n_epochs,
            learning_rate=config.learning_rate,
            model_learning_rate=config.model_learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            batch_size=config.batch_size,
            pg_coef=config.pg_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            ext_rew_coef=config.ext_rew_coef,
            ext_rew_pretrain_coef=config.ext_rew_pretrain_coef,
            int_rew_source=config.int_rew_source,
            int_rew_coef=config.int_rew_coef,
            int_rew_norm=config.int_rew_norm,
            int_rew_momentum=config.int_rew_momentum,
            int_rew_eps=config.int_rew_eps,
            int_rew_clip=config.int_rew_clip,
            adv_momentum=config.adv_momentum,
            adv_norm=config.adv_norm,
            adv_eps=config.adv_eps,
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range_vf,
            policy_kwargs=policy_kwargs,
            env_source=config.env_source,
            env_render=config.env_render,
            fixed_seed=config.fixed_seed,
            use_wandb=config.use_wandb,
            local_logger=config.local_logger,
            enable_plotting=config.enable_plotting,
            plot_interval=config.plot_interval,
            plot_colormap=config.plot_colormap,
            log_explored_states=config.log_explored_states,
            verbose=0,
        )
        
        if config.reward_learning_frequency >= config.total_steps or config.reward_learning_frequency == -1 and config.curr_iter > 0:
            # It's Human Teacher setting and there is at least one pretrained model.
            # Load latest trained model chechpoint instead of pretrained model, to resume interrupted training.
            search_pattern = os.path.join(config.log_dir, 'train_model_*')
            matching_files = glob.glob(search_pattern)
            if not matching_files:
                raise ValueError(f"Training model checkpoint not found for resuming training at iteration {config.curr_iter}. Expected at least one file matching: {search_pattern}")
            latest_model_path = max(matching_files, key=os.path.getmtime)
        else:
            search_pattern = os.path.join(config.log_dir, 'pretrain_model_*')
            matching_files = glob.glob(search_pattern)
            latest_model_path = max(matching_files, key=os.path.getmtime)

        model = PPOTrainer.load(load_path=latest_model_path, **trainer_kwargs)
        model.policy.eval() # Set the policy to evaluation mode for trajectory generation

        ptr_str = f"Generating trajectories {chunk[0]}-{chunk[-1]}/{config.episode_num}: "

        writers = []
        xai_writers = []
        for vid in chunk:
            writers.append(imageio.get_writer(
                    f"{config.log_dir}/traj_videos/traj{config.run_id:02}{vid:03}.mp4",
                    fps=config.fps,
                    codec="libx264",
                    quality=8
                )
            )
            if config.gen_xai_videos:
                xai_writers.append(imageio.get_writer(
                    f"{config.log_dir}/traj_xai_videos/traj{config.run_id:02}{vid:03}.mp4",
                    fps=config.fps,
                    codec="libx264",
                    quality=8
                )
            )
        
        last_obs_list = [[] for _ in range(current_batch_size)]
        next_obs_list = [[] for _ in range(current_batch_size)]
        last_policy_mems_list = [[] for _ in range(current_batch_size)]
        action_list = [[] for _ in range(current_batch_size)]
        reward_list = [[] for _ in range(current_batch_size)]
        episode_starts_list = [[] for _ in range(current_batch_size)]
        done_list = [[] for _ in range(current_batch_size)]

        last_obs = env.reset()
    
        def float_zeros(tensor_shape):
            return th.zeros(tensor_shape, device=model.device, dtype=th.float32)
        
        last_policy_mems = float_zeros([current_batch_size, model.policy.gru_layers, model.policy.dim_policy_features])
        dones = np.zeros(current_batch_size, dtype=bool)
        dones_target = np.zeros(current_batch_size, dtype=bool) # To track which environments have reached done=True at least once
        if config.gen_xai_videos:
            wrapped_model = ValueNetworkWrapper(model, config.xai_network)
            xai_method = XplainMethod.get_enum_xplain_method(config.xai_method)
            xplainer = fetch_captum_explainer(xai_method, wrapped_model, model, kwargs=xai_kwargs) # Initialize the Captum explainer with the custom wrapper

        for step in range(config.video_length):
            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, model.device)
                action, value, log_prob, policy_mem = model.policy.forward(obs_tensor, last_policy_mems) 
                action = action.cpu().numpy()

            clipped_action = action
            # Clip the action to avoid out of bound error
            if isinstance(model.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, model.action_space.low, model.action_space.high)

            next_obs, rewards, dones, infos = env.step(clipped_action)
            frames = env.get_images()
            for i, writer in enumerate(writers):
                if dones_target[i]: # Stop recording if that specific env finished
                    continue
                writer.append_data(frames[i])
            if config.gen_xai_videos:
                attribution_maps = generate_attribution_map(last_obs,
                                                            last_policy_mems, 
                                                            action,
                                                            env, 
                                                            model.device,
                                                            xplainer,
                                                            frames, 
                                                            xai_method) # Generate attribution map for the current step
                for i, xai_writer in enumerate(xai_writers):
                    if dones_target[i]: # Stop recording if that specific env finished
                        continue
                    xai_writer.append_data(attribution_maps[i])
            
            for i in range(current_batch_size):
                if dones_target[i]: # Skip logging for this environment if it has already reached done=True at least once
                    continue
                last_obs_list[i].append(last_obs[i])
                next_obs_list[i].append(next_obs[i])
                last_policy_mems_list[i].append(last_policy_mems[i].clone().cpu().numpy())
                action_list[i].append(action[i])
                reward_list[i].append(rewards[i]) # Use the individual reward for each environment
                episode_starts_list[i].append(step == 0)
                done_list[i].append(dones[i])
            last_obs = next_obs
            last_policy_mems = policy_mem
            if dones.any():
                dones_target = np.logical_or(dones_target, dones) # Update the target tracking which envs have reached done=True
                if dones_target.all(): # If all environments have reached done=True at least once, we can stop the trajectory generation early
                    break

        for i in range(current_batch_size):
            if not dones_target[i]: # Episode did not finish
                done_list[i][-1] = True # Mark the last step as done to indicate episode termination in the saved data
                
        env.close()
        writer.close()
        if config.gen_xai_videos:
            xai_writer.close()

        for i in range(current_batch_size):
            data = {
                "last_obs": np.array(last_obs_list[i], dtype=np.uint8),
                "next_obs": np.array(next_obs_list[i], dtype=np.uint8),
                "last_policy_mems": np.array(last_policy_mems_list[i], dtype=np.float32),
                "actions": np.array(action_list[i]),
                "rewards": np.array(reward_list[i], dtype=np.float32), # Downcast to save space
                "episode_starts": np.array(episode_starts_list[i], dtype=bool),
                "dones": np.array(done_list[i], dtype=bool)
            }
            video_id = chunk[i]
            np.savez_compressed(f"{config.log_dir}/traj_data/traj{config.run_id:02}{video_id:03}.npz", **data)
        print(f"{ptr_str} Trajectories {chunk[0]}-{chunk[-1]} completed. Lengths: {[len(reward_list[i]) for i in range(current_batch_size)]}.")

    end_time = time.perf_counter()
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration:.2f} seconds")
    print(f"Average time per video: {total_duration / len(gen_video_ids):.2f} seconds")

@click.command()
# Experiment params
@click.option('--run_id', default=0, type=int, help='Index (and seed) of the current run')
@click.option('--group_name', type=str, help='Group name (wandb option), leave blank if not logging with wandb')
@click.option('--log_dir', default='./logs', type=str, help='Directory for saving training logs')
@click.option('--total_steps', default=int(1e6), type=int, help='Total number of frames to run for training')
# Agent params
@click.option('--features_dim', default=64, type=int, help='Number of neurons of a learned embedding (PPO)')
@click.option('--model_features_dim', default=128, type=int,
              help='Number of neurons of a learned embedding (dynamics model)')
@click.option('--learning_rate', default=3e-4, type=float, help='Learning rate of PPO')
@click.option('--model_learning_rate', default=3e-4, type=float, help='Learning rate of the dynamics model')
@click.option('--num_processes', default=8, type=int, help='Number of training processes (workers)')
@click.option('--batch_size', default=512, type=int, help='Batch size')
@click.option('--n_steps', default=512, type=int, help='Number of steps to run for each process per PPO update during training')
# Env params
@click.option('--env_source', default='minigrid', type=str, help='minigrid or procgen')
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, ninja, jumper')
@click.option('--project_name', required=False, type=str, help='Where to store training logs (wandb option)')
@click.option('--map_size', default=5, type=int, help='Size of the minigrid room')
@click.option('--can_see_walls', default=1, type=int, help='Whether walls are visible to the agent')
@click.option('--fully_obs', default=0, type=int, help='Whether the agent can receive full observations')
@click.option('--image_noise_scale', default=0.0, type=float, help='Standard deviation of the Gaussian noise')
@click.option('--procgen_mode', default='hard', type=str, help='Mode of ProcGen games (easy or hard)')
@click.option('--procgen_num_threads', default=4, type=int, help='Number of parallel ProcGen threads')
@click.option('--log_explored_states', default=0, type=int, help='Whether to log the number of explored states')
@click.option('--fixed_seed', default=-1, type=int, help='Whether to use a fixed env seed (MiniGrid)')
# Algo params
@click.option('--n_epochs', default=4, type=int, help='Number of epochs to train policy and value nets')
@click.option('--model_n_epochs', default=4, type=int, help='Number of epochs to train common_models')
@click.option('--gamma', default=0.99, type=float, help='Discount factor')
@click.option('--gae_lambda', default=0.95, type=float, help='GAE lambda')
@click.option('--pg_coef', default=1.0, type=float, help='Coefficient of policy gradients')
@click.option('--vf_coef', default=0.5, type=float, help='Coefficient of value function loss')
@click.option('--ent_coef', default=0.01, type=float, help='Coefficient of policy entropy')
@click.option('--max_grad_norm', default=0.5, type=float, help='Maximum norm of gradient')
@click.option('--clip_range', default=0.2, type=float, help='PPO clip range of the policy network')
@click.option('--clip_range_vf', default=-1, type=float,
              help='PPO clip range of the value function (-1: disabled, >0: enabled)')
@click.option('--adv_norm', default=2, type=int,
              help='Normalized advantages by: [0] No normalization [1] Standardization per mini-batch [2] Standardization per rollout buffer [3] Standardization w.o. subtracting the mean per rollout buffer')
@click.option('--adv_eps', default=1e-5, type=float, help='Epsilon for advantage normalization')
@click.option('--adv_momentum', default=0.9, type=float, help='EMA smoothing factor for advantage normalization')
# Reward Model params
@click.option('--reward_learning_frequency', default=int(0), type=int, help='Frequency of Reward Model updates per agent updates (0: no updates, -1: only once at the beginning, >=1: every X steps)')
@click.option('--episode_num', default=64, type=int, help='Number of episodes to be generated for Reward Model training')
@click.option('--preference_buffer_capacity', default=int(1e4), type=int, help='Number of episodes that can be stored in the preference buffer')
@click.option('--sampling_strategy', default='Uniform', type=str, help='Sampling strategy for generating preference pairs: [Uniform|SwissInfoGain]')
@click.option('--pair_num', default=128, type=int, help='Number of preference pairs to be generated for Reward Model training (used for relevant strategies)')
@click.option('--curr_iter', default=0, type=int, help='Current iteration of reward model training (used for sampling strategy state management)')
@click.option('--reward_epochs', default=100, type=int, help='Number of epochs to train Reward Model')
@click.option('--reward_batch_size', default=32, type=int, help='Batch size for Reward Model training (Preferred, cause of variable-length sequences)')
@click.option('--reward_learning_rate', default=3e-2, type=float, help='Learning rate of Reward Model')
@click.option('--reward_ensemble_size', default=3, type=int, help='Number of models in the Reward Model ensemble')
@click.option('--reward_activation_fn', default='relu', type=str, help='Activation function for Reward Model')
# Intrinsic Reward params
@click.option('--ext_rew_coef', default=1.0, type=float, help='Coefficient of extrinsic rewards')
@click.option('--ext_rew_pretrain_coef', default=0.0, type=float, help='Coefficient of extrinsic rewards during pretraining')
@click.option('--int_rew_coef', default=1e-2, type=float, help='Coefficient of intrinsic rewards (IRs)')
@click.option('--int_rew_source', default='NoModel', type=str,
              help='Source of IRs: [NoModel|AEGIS|DEIR|ICM|RND|NGU|NovelD|PlainDiscriminator|PlainInverse|PlainForward]')
@click.option('--int_rew_norm', default=1, type=int,
              help='Normalized IRs by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--int_rew_momentum', default=0.9, type=float,
              help='EMA smoothing factor for IR normalization (-1: total average)')
@click.option('--int_rew_eps', default=1e-5, type=float, help='Epsilon for IR normalization')
@click.option('--int_rew_clip', default=-1, type=float, help='Clip IRs into [-X, X] when X>0')
@click.option('--aegis_nov_exp_mem_capacity', default=10000, type=int, help='Novel experience memory capacity (AEGIS)')
@click.option('--aegis_knn_k', default=5, type=int, help='Search for K nearest neighbors (AEGIS)')
@click.option('--aegis_dst_momentum', default=0.997, type=float, help='EMA smoothing factor for averaging embedding distances (AEGIS)')
@click.option('--dsc_obs_queue_len', default=100000, type=int, help='Maximum length of observation queue (DEIR)')
@click.option('--icm_forward_loss_coef', default=0.2, type=float, help='Coefficient of forward model losses (ICM)')
@click.option('--ngu_knn_k', default=10, type=int, help='Search for K nearest neighbors (NGU)')
@click.option('--ngu_use_rnd', default=1, type=int, help='Whether to enable lifelong IRs generated by RND (NGU)')
@click.option('--ngu_dst_momentum', default=0.997, type=float, help='EMA smoothing factor for averaging embedding distances (NGU)')
@click.option('--rnd_use_policy_emb', default=1, type=int, help='Whether to use the embeddings learned by policy/value nets as inputs (RND)')
@click.option('--rnd_err_norm', default=1, type=int, help='Normalized RND errors by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--rnd_err_momentum', default=-1, type=float, help='EMA smoothing factor for RND error normalization (-1: total average)')
# Network params
@click.option('--use_model_rnn', default=1, type=int, help='Whether to enable RNNs for the dynamics model')
@click.option('--latents_dim', default=256, type=int, help='Dimensions of latent features in policy/value nets\' MLPs')
@click.option('--model_latents_dim', default=256, type=int, help='Dimensions of latent features in the dynamics model\'s MLP')
@click.option('--policy_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--policy_mlp_layers', default=1, type=int, help='Number of latent layers used in the policy\'s MLP')
@click.option('--policy_cnn_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' CNN')
@click.option('--policy_mlp_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' MLP')
@click.option('--policy_gru_norm', default='NoNorm', type=str, help='Normalization type for policy/value nets\' GRU')
@click.option('--model_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--model_mlp_layers', default=1, type=int, help='Number of latent layers used in the model\'s MLP')
@click.option('--model_cnn_norm', default='BatchNorm', type=str, help='Normalization type for the dynamics model\'s CNN')
@click.option('--model_mlp_norm', default='BatchNorm', type=str, help='Normalization type for the dynamics model\'s MLP')
@click.option('--model_gru_norm', default='NoNorm', type=str, help='Normalization type for the dynamics model\'s GRU')
@click.option('--activation_fn', default='relu', type=str, help='Activation function for non-CNN layers')
@click.option('--cnn_activation_fn', default='relu', type=str, help='Activation function for CNN layers')
@click.option('--gru_layers', default=1, type=int, help='Number of GRU layers in both the policy and the model')
# Optimizer params
@click.option('--optimizer', default='adam', type=str, help='Optimizer, adam or rmsprop')
@click.option('--optim_eps', default=1e-5, type=float, help='Epsilon for optimizers')
@click.option('--adam_beta1', default=0.9, type=float, help='Adam optimizer option')
@click.option('--adam_beta2', default=0.999, type=float, help='Adam optimizer option')
@click.option('--rmsprop_alpha', default=0.99, type=float, help='RMSProp optimizer option')
@click.option('--rmsprop_momentum', default=0.0, type=float, help='RMSProp optimizer option')
# Logging & Video Generation options
@click.option('--write_local_logs', default=0, type=int, help='Whether to output training logs locally')
@click.option('--enable_plotting', default=0, type=int, help='Whether to generate plots for analysis')
@click.option('--plot_interval', default=10, type=int, help='Interval of generating plots (iterations)')
@click.option('--plot_colormap', default='Blues', type=str, help='Colormap of plots to generate')
@click.option('--chunk_size', default=32, type=int, help='Chunk size for trajectory generation (number of videos generated at the same time')
@click.option('--fps', default=5, type=int, help='FPS of generated trajectory videos (default is for MiniGrid)')
@click.option('--video_length', default=100, type=int, help='Max length of the video (frames)')
@click.option('--gen_xai_videos', default=False, type=bool, help='Whether to generate XAI videos saliency maps of policy predictions')
@click.option('--xai_method', default='saliency', type=str, help='Method for generating XAI videos saliency maps of policy predictions')
@click.option('--xai_network', default='value', type=str, help='Network for generating XAI videos saliency maps of policy predictions: [value|policy]')
@click.option('--traj_overwrite', default=True, type=bool, help='Whether the generated trajectories should overwrite existing ones in the log directory (if 0, trajectories will be saved with an incremental index)')
@click.option('--record_video', default=0, type=int, help='Whether the environment should be wrapped in a video recorder (don\'t use for human feedback setting)')
@click.option('--log_dsc_verbose', default=0, type=int, help='Whether to record the discriminator loss for each action')
@click.option('--env_render', default=0, type=int, help='Whether to render games in human mode')
@click.option('--use_status_predictor', default=0, type=int, help='Whether to train status predictors for analysis (MiniGrid only)')

def main(run_id, group_name, log_dir, total_steps, features_dim, model_features_dim, learning_rate, model_learning_rate, num_processes, batch_size, n_steps, env_source, game_name, project_name, map_size, 
         can_see_walls, fully_obs, image_noise_scale, procgen_mode, procgen_num_threads, log_explored_states, fixed_seed, 
         n_epochs, model_n_epochs, gamma, gae_lambda, pg_coef, vf_coef, ent_coef, max_grad_norm, clip_range, clip_range_vf, 
         adv_norm, adv_eps, adv_momentum, reward_learning_frequency, episode_num, preference_buffer_capacity, sampling_strategy, pair_num, curr_iter,
         reward_epochs, reward_batch_size, reward_learning_rate, reward_ensemble_size, reward_activation_fn, ext_rew_coef, ext_rew_pretrain_coef, int_rew_coef,
         int_rew_source, int_rew_norm, int_rew_momentum, int_rew_eps, int_rew_clip, aegis_nov_exp_mem_capacity, aegis_knn_k, aegis_dst_momentum, dsc_obs_queue_len, log_dsc_verbose, 
         icm_forward_loss_coef, ngu_knn_k, ngu_dst_momentum, ngu_use_rnd, rnd_err_norm, rnd_err_momentum, use_model_rnn, rnd_use_policy_emb,
         latents_dim, model_latents_dim, policy_cnn_type, policy_mlp_layers, policy_cnn_norm, policy_mlp_norm, policy_gru_norm, model_cnn_type, 
         model_mlp_layers, model_cnn_norm, model_mlp_norm, model_gru_norm, activation_fn, cnn_activation_fn, gru_layers, optimizer, 
         optim_eps, adam_beta1, adam_beta2, rmsprop_alpha, rmsprop_momentum, write_local_logs, enable_plotting, plot_interval, plot_colormap, chunk_size, fps, video_length, gen_xai_videos, xai_method, xai_network,
         traj_overwrite, record_video, env_render, use_status_predictor):
    
    set_random_seed(run_id, using_cuda=True)
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)

    config.init_env_name(game_name, project_name)
    config.init_meta_info()
    config.init_logger()
    config.init_values()

    generate_trajectories(config)

    config.close()

if __name__ == '__main__':
    main()