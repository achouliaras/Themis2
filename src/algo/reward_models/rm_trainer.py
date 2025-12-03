import gymnasium as gym
import numpy as np
import time
import torch as th
from torch import nn
import torch.nn.functional as F
import wandb
import warnings
from minigrid.core.world_object import Key, Door, Goal
from src.algo.buffers.preference_buffer import PreferenceBuffer
from src.algo.reward_models.mlp_ensemble import MLPEnsemble
from src.utils.loggers import StatisticsLogger, LocalLogger
from src.utils.common_func import set_random_seed
from src.utils.enum_types import ModelType, EnvSrc

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback

from typing import Any, Dict, Optional, Tuple, Type, Union

class RewardModelTrainer(BaseAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[MLPEnsemble]],
        env: Union[GymEnv, str],
        run_id: int,
        reward_epochs: int,
        reward_batch_size: int,
        reward_learning_rate: Union[float, Schedule],
        preference_buffer_capacity: int,
        rl_policy: ActorCriticPolicy,
        int_rew_source: ModelType,
        int_rew_coef: float,
        int_rew_norm : int,
        int_rew_momentum: Optional[float],
        int_rew_eps : float,
        int_rew_clip : float,
        image_noise_scale : float,
        can_see_walls : int,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        enable_plotting: int = 0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        env_source: Optional[EnvSrc] = None,
        env_render: Optional[int] = None,
        fixed_seed: Optional[int] = None,
        log_explored_states: Optional[int] = None,
        local_logger: Optional[LocalLogger] = None,
        use_wandb: bool = False,
        sampling_strategy: str = "Uniform",
    ):
        super(RewardModelTrainer, self).__init__(
            policy=policy,
            env=env,
            learning_rate=reward_learning_rate,
            policy_kwargs=None,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
        )
        self.run_id = run_id
        self.reward_epochs = reward_epochs
        self.reward_batch_size = reward_batch_size
        self.preference_buffer_capacity = preference_buffer_capacity
        self.rl_policy = rl_policy
        self.int_rew_source = int_rew_source
        self.int_rew_coef = int_rew_coef
        self.int_rew_norm = int_rew_norm
        self.int_rew_eps = int_rew_eps
        self.int_rew_clip = int_rew_clip
        self.image_noise_scale = image_noise_scale
        self.can_see_walls = can_see_walls
        self.int_rew_momentum = int_rew_momentum
        self.env_source = env_source
        self.env_render = env_render
        self.fixed_seed = fixed_seed
        self.log_explored_states = log_explored_states
        self.local_logger = local_logger
        self.use_wandb = use_wandb
        self.enable_plotting = enable_plotting
        self.policy_kwargs = policy_kwargs if policy_kwargs is not None else {}
        self.sampling_strategy = sampling_strategy
        
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            reward_batch_size > 1
        ), "`reward_batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
        assert self.preference_buffer_capacity > 0, "`preference_buffer_capacity` must be greater than 0"
        
        if _init_setup_model:
            self._setup_model()

    @property
    def max_episode_length(self) -> int:
        """
        Return the maximum episode length from the base environment,
        """
        try:
            # For vectorized environments, get the attribute from the first environment
            max_steps = self.env.get_attr('max_steps')[0]
            return max_steps
        except (AttributeError, IndexError):
            pass
        
        raise AttributeError("Could not determine max episode length from environment.")

    def _setup_model(self) -> None:
        set_random_seed(self.seed)
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        if self.verbose>0: print(f"Max episode length: {self.max_episode_length}")
        self.preference_buffer = PreferenceBuffer(
            self.preference_buffer_capacity*self.max_episode_length,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            features_dim=self.rl_policy.features_dim,
            dim_policy_traj=self.rl_policy.dim_policy_features,
            dim_model_traj=self.rl_policy.dim_model_features,
            int_rew_coef=self.int_rew_coef,
            int_rew_norm=self.int_rew_norm,
            int_rew_clip=self.int_rew_clip,
            int_rew_eps=self.int_rew_eps,
            int_rew_momentum=self.int_rew_momentum,
            gru_layers=self.rl_policy.gru_layers,
            use_status_predictor=self.rl_policy.use_status_predictor,
        )

    def _setup_synthetic_teacher(self, config: Optional[dict]=None) -> None:
        self.teacher = {
            "beta": -1.0 if config is None else config["teacher_beta"],
            "gamma": 1.0 if config is None else config["teacher_gamma"],
            "eps_mistake": 0.0 if config is None else config["teacher_eps_mistake"],
            "eps_equal": 0.0 if config is None else config["teacher_eps_equal"],
            "thres_equal": 0.01 if config is None else config["teacher_thres_equal"],
        }
        
    def _setup_sampling_strategy(self, pair_num: int) -> None:
        strategy = {}
        if self.sampling_strategy == "Uniform":
            strategy["name"] = "Uniform"    
            strategy["n_pairs"] = pair_num
        elif self.sampling_strategy == "SwissInfoGain":
            strategy["name"] = "SwissInfoGain"
            strategy["n_pairs"] = -1  # adaptive number of pairs
        return strategy
    
    def set_teacher_thres_equal(self, new_margin):
        self.teacher["thres_equal"] = new_margin * self.teacher["eps_equal"]

    def on_training_start(self):
        if isinstance(self._last_obs, Dict):
            self._last_obs = self._last_obs["rgb"]

        if self.env_source == EnvSrc.MiniGrid:
            # Set advanced options for MiniGrid envs
            self.env.can_see_walls = self.can_see_walls
            self.env.image_noise_scale = self.image_noise_scale
            self.env.image_rng = np.random.default_rng(seed=self.run_id + 1313)                    
            
            self._last_obs = self.env.reset()

            # Init variables for logging
            self.width = self.env.get_attr('width')[0]
            self.height = self.env.get_attr('height')[0]
            self.global_visit_counts = np.zeros([self.width, self.height], dtype=np.int32)
            self.global_reward_map_maxs = np.zeros([self.width, self.height], dtype=np.float64)
            self.global_reward_map_sums = np.zeros([self.width, self.height], dtype=np.float64)
            self.global_reward_map_nums = np.zeros([self.width, self.height], dtype=np.int32)
            self.global_value_map_sums = np.zeros([self.width, self.height], dtype=np.float64)
            self.global_value_map_nums = np.zeros([self.width, self.height], dtype=np.int32)

        self.global_episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.global_episode_intrinsic_rewards = np.zeros(self.n_envs, dtype=np.float32)

        self.global_episode_unique_states = np.zeros(self.n_envs, dtype=np.float32)
        self.global_episode_visited_states = [dict() for _ in range(self.n_envs)]
        self.global_lifelong_unique_states = 0
        self.global_lifelong_visited_states = dict()
        self.global_episode_visited_positions = [dict() for _ in range(self.n_envs)]
        self.global_episode_visited_pos_sum = np.zeros(self.n_envs, dtype=np.float32)

        self.global_episode_steps = np.zeros(self.n_envs, dtype=np.int32)
        if self.rl_policy.use_status_predictor:
            self.global_has_keys = np.zeros(self.n_envs, dtype=np.int32)
            self.global_open_doors = np.zeros(self.n_envs, dtype=np.int32)
            self.curr_key_status = np.zeros(self.n_envs, dtype=np.float32)
            self.curr_door_status = np.zeros(self.n_envs, dtype=np.float32)
            self.curr_agent_pos = np.zeros((self.n_envs, 2), dtype=np.float32)
            self.curr_target_dists = np.zeros((self.n_envs, 3), dtype=np.float32)
        else:
            self.global_has_keys = None
            self.global_open_doors = None
            self.curr_key_status = None
            self.curr_door_status = None
            self.curr_agent_pos = None
            self.curr_target_dists = None

        self.episodic_obs_emb_history = [None for _ in range(self.n_envs)]
        self.episodic_trj_emb_history = [None for _ in range(self.n_envs)]

        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            self.rl_policy.int_rew_model.init_obs_queue(self._last_obs)

        def float_zeros(tensor_shape):
            return th.zeros(tensor_shape, device=self.device, dtype=th.float32)

        self._last_policy_mems = float_zeros([self.n_envs, self.rl_policy.gru_layers, self.rl_policy.dim_policy_features])
        self._last_model_mems = float_zeros([self.n_envs, self.rl_policy.gru_layers, self.rl_policy.dim_model_features])

        if self.int_rew_source in [ModelType.AEGIS, ModelType.AEGIS_alt, ModelType.AEGIS_global_only, ModelType.AEGIS_local_only]:
            self.rl_policy.int_rew_model.init_obs_queue(self._last_obs)
            # last_obs_tensor = obs_as_tensor(self._last_obs, self.device)
            # self.rl_policy.int_rew_model.init_novel_experience_memory(last_obs_tensor, 
            #                                                        self._last_model_mems,
            #                                                        device=self.device)
            self._last_global_novelty = np.zeros(self.n_envs, dtype=np.float32)
        
    def init_on_rollout_start(self):
        # Log statistics data per each rollout
        self.rollout_stats = StatisticsLogger(mode='rollout')
        self.rollout_done_episodes = 0
        self.rollout_done_episode_steps = 0
        self.rollout_sum_rewards = 0
        self.rollout_episode_unique_states = 0
        self.rollout_done_episode_unique_states = 0
    
    def log_before_transition(self, values):
        if self.env_source == EnvSrc.MiniGrid:
            self._last_state_hash_vals = self.env.env_method('hash')

        # Update Key and Door Status
        agent_positions = None
        if self.rl_policy.use_status_predictor:
            agent_positions = np.array(self.env.get_attr('agent_pos'))
            agent_carryings = self.env.get_attr('carrying')
            env_grids = self.env.get_attr('grid')

            self.curr_door_pos = np.zeros((self.n_envs, 2), dtype=np.int32)
            self.curr_key_pos = np.copy(agent_positions).reshape(self.n_envs, 2)
            self.curr_goal_pos = np.zeros((self.n_envs, 2), dtype=np.int32)
            for env_id in range(self.n_envs):
                # The only possible carrying in DoorKey is the key
                self.global_has_keys[env_id] = int(isinstance(agent_carryings[env_id], Key))

                # Door, Key, Goal positions
                for env_id, grid in enumerate(env_grids[env_id].grid):
                    col = env_id % self.width
                    row = env_id // self.width
                    if isinstance(grid, Door):
                        self.curr_door_pos[env_id] = np.array((col, row))
                        self.global_open_doors[env_id] = int(grid.is_open)
                    elif isinstance(grid, Key):
                        self.curr_key_pos[env_id] = np.array((col, row))
                    elif isinstance(grid, Goal):
                        self.curr_goal_pos[env_id] = np.array((col, row))

            self.curr_key_status = np.copy(self.global_has_keys)
            self.curr_door_status = np.copy(self.global_open_doors)
            self.rollout_stats.add(
                key_status=np.mean(self.global_has_keys),
                door_status=np.mean(self.global_open_doors),
            )
        
        # Update agent position and visit count
        if self.rl_policy.use_status_predictor or self.enable_plotting:
            if agent_positions is None:
                agent_positions = np.array(self.env.get_attr('agent_pos'))
            for i in range(self.n_envs):
                c, r = agent_positions[i]
                self.global_visit_counts[r, c] += 1
                self.global_value_map_sums[r, c] += values[i].item()
                self.global_value_map_nums[r, c] += 1

            # Current agent position
            self.curr_agent_pos = np.copy(agent_positions)

            # Define the target of position prediction loss
            if self.rl_policy.use_status_predictor:
                # Manhattan Distance to the Door
                key_dists = np.abs(self.curr_agent_pos - self.curr_key_pos)
                key_dists = np.sum(key_dists, axis=1) / (self.width + self.height)
                door_dists = np.abs(self.curr_agent_pos - self.curr_door_pos)
                door_dists = np.sum(door_dists, axis=1) / (self.width + self.height)
                goal_dists = np.abs(self.curr_agent_pos - self.curr_goal_pos)
                goal_dists = np.sum(goal_dists, axis=1) / (self.width + self.height)
                self.curr_target_dists = np.stack([key_dists, door_dists, goal_dists], axis=1)

    def log_after_transition(self, rewards, intrinsic_rewards):
        self.global_episode_rewards += rewards
        self.global_episode_intrinsic_rewards += intrinsic_rewards
        self.global_episode_steps += 1

        # Logging episodic/lifelong visited states, reward map
        if self.log_explored_states:
            # 0 - Not to log
            # 1 - Log both episodic and lifelong states
            # 2 - Log episodic visited states only
            if self.env_source == EnvSrc.MiniGrid:
                agent_positions = np.array(self.env.get_attr('agent_pos'))
                for env_id in range(self.n_envs):
                    c, r = agent_positions[env_id]

                    # count the visited positions
                    pos = c * self.width + r
                    pos_visit_count = self.global_episode_visited_positions[env_id]
                    if pos not in pos_visit_count:
                        pos_visit_count[pos] = 1
                        self.global_episode_visited_pos_sum[env_id] += 1
                    else:
                        pos_visit_count[pos] += 1

                    env_hash = self._last_state_hash_vals[env_id]
                    if env_hash in self.global_episode_visited_states[env_id]:
                        self.global_episode_visited_states[env_id][env_hash] += 1
                    else:
                        self.global_episode_visited_states[env_id][env_hash] = 1
                        self.global_episode_unique_states[env_id] += 1
                        self.rollout_episode_unique_states += 1

                    if self.log_explored_states == 1:
                        if env_hash in self.global_lifelong_visited_states:
                            self.global_lifelong_visited_states[env_hash] += 1
                        else:
                            self.global_lifelong_visited_states[env_hash] = 1
                            self.global_lifelong_unique_states += 1

                    if self.enable_plotting:
                        self.global_reward_map_maxs[r, c] = np.maximum(
                            self.global_reward_map_maxs[r, c],
                            intrinsic_rewards[env_id]
                        )
                        self.global_reward_map_sums[r, c] += intrinsic_rewards[env_id]
                        self.global_reward_map_nums[r, c] += 1

            elif self.env_source == EnvSrc.ProcGen:
                for env_id in range(self.n_envs):
                    # In Procgen games, the counted "states" are observations
                    env_hash = tuple(self._last_obs[env_id].reshape(-1).tolist())

                    if env_hash in self.global_episode_visited_states[env_id]:
                        self.global_episode_visited_states[env_id][env_hash] += 1
                    else:
                        self.global_episode_visited_states[env_id][env_hash] = 1
                        self.global_episode_unique_states[env_id] += 1
                        self.rollout_episode_unique_states += 1

                    if self.log_explored_states == 1:
                        if env_hash in self.global_lifelong_visited_states:
                            self.global_lifelong_visited_states[env_hash] += 1
                        else:
                            self.global_lifelong_visited_states[env_hash] = 1
                            self.global_lifelong_unique_states += 1

    def clear_on_episode_end(self, dones, policy_mems, model_mems):
        for env_id in range(self.n_envs):
            if dones[env_id]:
                if policy_mems is not None: policy_mems[env_id] *= 0.0
                if model_mems is not None: model_mems[env_id] *= 0.0
                if self.int_rew_source in [ModelType.AEGIS, ModelType.AEGIS_alt, ModelType.AEGIS_global_only, ModelType.AEGIS_local_only]:
                    if self._last_global_novelty is not None: self._last_global_novelty[env_id] = 0.0
                self.episodic_obs_emb_history[env_id] = None
                self.episodic_trj_emb_history[env_id] = None
                self.rollout_sum_rewards += self.global_episode_rewards[env_id]
                self.rollout_done_episode_steps += self.global_episode_steps[env_id]
                self.rollout_done_episode_unique_states += self.global_episode_unique_states[env_id]
                self.rollout_done_episodes += 1
                self.global_episode_rewards[env_id] = 0
                self.global_episode_intrinsic_rewards[env_id] = 0
                self.global_episode_unique_states[env_id] = 0
                self.global_episode_visited_states[env_id] = dict()  # logging use
                self.global_episode_visited_positions[env_id] = dict()  # logging use
                self.global_episode_visited_pos_sum[env_id] = 0  # logging use
                self.global_episode_steps[env_id] = 0
                if self.rl_policy.use_status_predictor:
                    self.global_has_keys[env_id] = 0
                    self.global_open_doors[env_id] = 0
                    self.curr_key_status[env_id] = 0
                    self.curr_door_status[env_id] = 0

    def log_on_rollout_end(self, log_interval):
        if log_interval is not None and self.iteration % log_interval == 0:
            log_data = {
                "iterations": self.iteration,
                "time/fps": int(self.num_timesteps / (time.time() - self.start_time)),
                "time/time_elapsed": int(time.time() - self.start_time),
                "time/total_timesteps": self.num_timesteps,
                "rollout/ep_rew_mean": self.rollout_sum_rewards / (self.rollout_done_episodes + 1e-8),
                "rollout/ep_len_mean": self.rollout_done_episode_steps / (self.rollout_done_episodes + 1e-8),
                # unique states / positions
                "rollout/ep_unique_states": self.rollout_done_episode_unique_states / (
                            self.rollout_done_episodes + 1e-8),
                "rollout/ll_unique_states": self.global_lifelong_unique_states,
                "rollout/ep_unique_states_per_step": self.rollout_episode_unique_states / (
                            self.preference_buffer.buffer_size * self.n_envs),
                "rollout/ll_unique_states_per_step": self.global_lifelong_unique_states / self.num_timesteps,
                # intrinsic rewards
                "rollout/int_rew_coef": self.preference_buffer.int_rew_coef,
                "rollout/int_rew_buffer_mean": self.preference_buffer.int_rew_mean,
                "rollout/int_rew_buffer_std": self.preference_buffer.int_rew_std,
            }

            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                log_data.update({
                    "rollout/ep_info_true_rew_mean": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    "rollout/ep_info_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                })
            else:
                log_data.update({
                    "rollout/ep_info_rew_mean": 0.0,
                    "rollout/ep_info_len_mean": np.nan,
                })

            if self.int_rew_coef > 0:
                log_data.update({
                    "rollout/int_rew_mean": np.mean(self.preference_buffer.intrinsic_rewards),
                    "rollout/int_rew_std": np.std(self.preference_buffer.intrinsic_rewards),
                    "rollout/pos_int_rew_mean": np.maximum(self.preference_buffer.intrinsic_rewards, 0.0).mean(),
                    "rollout/neg_int_rew_mean": np.minimum(self.preference_buffer.intrinsic_rewards, 0.0).mean(),
                })

            # Update with other stats
            log_data.update(self.rollout_stats.to_dict())

            # Logging with wandb
            if self.use_wandb:
                wandb.log(log_data)
            # Logging with local logger
            if self.local_logger is not None:
                self.local_logger.write(log_data, log_type='rm_rollout')

    def create_intrinsic_rewards(self, new_obs, actions, dones):
        if self.int_rew_source == ModelType.NoModel:
            intrinsic_rewards = np.zeros([self.n_envs], dtype=float)
            model_mems = None
            return intrinsic_rewards, model_mems

        # Prepare input tensors for IR generation
        with th.no_grad():
            curr_obs_tensor = obs_as_tensor(self._last_obs, self.device)
            next_obs_tensor = obs_as_tensor(new_obs, self.device)
            curr_act_tensor = th.as_tensor(actions, dtype=th.int64, device=self.device)
            done_tensor = th.as_tensor(dones, dtype=th.int64, device=self.device)

            if self.rl_policy.use_model_rnn:
                last_model_mem_tensor = self._last_model_mems
                if self.int_rew_source in [ModelType.RND, ModelType.NGU, ModelType.NovelD]:
                    if self.rl_policy.rnd_use_policy_emb:
                        last_model_mem_tensor = self._last_policy_mems
            else:
                last_model_mem_tensor = None

            if self.rl_policy.use_status_predictor:
                key_status_tensor = th.as_tensor(self.curr_key_status, dtype=th.int64, device=self.device)
                door_status_tensor = th.as_tensor(self.curr_door_status, dtype=th.int64, device=self.device)
                target_dists_tensor = th.as_tensor(self.curr_target_dists, dtype=th.float32, device=self.device)
            else:
                key_status_tensor = None
                door_status_tensor = None
                target_dists_tensor = None

        # Aegis
        if self.int_rew_source in [ModelType.AEGIS, ModelType.AEGIS_alt, ModelType.AEGIS_global_only, ModelType.AEGIS_local_only]:
            intrinsic_rewards, model_mems, last_global_novelty = self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                obs_history=self.episodic_obs_emb_history,
                trj_history=self.episodic_trj_emb_history,
                last_global_novelty=self._last_global_novelty,
                stats_logger=self.rollout_stats,
                int_rew_source=self.int_rew_source
            )
            self._last_global_novelty = last_global_novelty # update global novelty for next step
            self.rl_policy.int_rew_model.update_obs_queue(
                iteration=self.iteration,
                intrinsic_rewards=intrinsic_rewards,
                ir_mean=self.preference_buffer.int_rew_stats.mean,
                new_obs=new_obs,
                stats_logger=self.rollout_stats
            )
        # DEIR / Plain discriminator model
        elif self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            intrinsic_rewards, model_mems = self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                obs_history=self.episodic_obs_emb_history,
                trj_history=self.episodic_trj_emb_history,
                plain_dsc=bool(self.int_rew_source == ModelType.PlainDiscriminator),
            )
            # Insert obs into the Discriminator's obs queue
            # Algorithm A2 in the Technical Appendix of DEIR paper
            self.rl_policy.int_rew_model.update_obs_queue(
                iteration=self.iteration,
                intrinsic_rewards=intrinsic_rewards,
                ir_mean=self.preference_buffer.int_rew_stats.mean,
                new_obs=new_obs,
                stats_logger=self.rollout_stats
            )
        # Plain forward / inverse model
        elif self.int_rew_source in [ModelType.PlainForward, ModelType.PlainInverse]:
            intrinsic_rewards, model_mems = self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                obs_history=self.episodic_obs_emb_history,
                key_status=key_status_tensor,
                door_status=door_status_tensor,
                target_dists=target_dists_tensor,
                stats_logger=self.rollout_stats
            )
        # ICM
        elif self.int_rew_source == ModelType.ICM:
            intrinsic_rewards, model_mems = self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                stats_logger=self.rollout_stats
            )
        # RND
        elif self.int_rew_source == ModelType.RND:
            intrinsic_rewards, model_mems = self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_dones=done_tensor,
                stats_logger=self.rollout_stats
            )
        # NGU
        elif self.int_rew_source == ModelType.NGU:
            intrinsic_rewards, model_mems = self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_act=curr_act_tensor,
                curr_dones=done_tensor,
                obs_history=self.episodic_obs_emb_history,
                stats_logger=self.rollout_stats
            )
        # NovelD
        elif self.int_rew_source == ModelType.NovelD:
            return self.rl_policy.int_rew_model.get_intrinsic_rewards(
                curr_obs=curr_obs_tensor,
                next_obs=next_obs_tensor,
                last_mems=last_model_mem_tensor,
                curr_dones=done_tensor,
                stats_logger=self.rollout_stats
            )
        else:
            raise NotImplementedError
        
        return intrinsic_rewards, model_mems

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        buffer: PreferenceBuffer,
        episode_num: int,
        strategy: Optional[dict] = None,
        verbose: int = 0,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        
        n_episodes = 0
        buffer.reset()

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.rl_policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        self.init_on_rollout_start()

        while n_episodes < episode_num:

            if self.use_sde and self.sde_sample_freq > 0 and n_episodes % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.rl_policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, _, policy_mems = \
                    self.rl_policy.forward(obs_tensor, self._last_policy_mems)
                actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Log before a transition
            self.log_before_transition(values)

            # Transition
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if isinstance(new_obs, Dict):
                new_obs = new_obs["rgb"]
            if self.env_render:
                env.render()

            # IR Generation
            intrinsic_rewards, model_mems = \
                self.create_intrinsic_rewards(new_obs, actions, dones)

            # Log after the transition and IR generation
            self.log_after_transition(rewards, intrinsic_rewards)

            # Clear episodic memories when an episode ends
            self.clear_on_episode_end(dones, policy_mems, model_mems)

            # Update global stats
            self.num_timesteps += self.n_envs
            self._update_info_buffer(infos)
            # Count episodes
            n_episodes += dones.sum().item()
            
            # self.verbose>0 and
            if verbose>0 and dones.sum().item() > 0:
                print(f"Finished {dones.sum().item()} at step {self.num_timesteps/self.n_envs}")

            # Add to buffer
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            buffer.add(
                self._last_obs,
                new_obs,
                self._last_policy_mems,
                self._last_model_mems,
                actions,
                rewards,
                intrinsic_rewards,
                self._last_episode_starts,
                dones,
                self.curr_key_status,
                self.curr_door_status,
                self.curr_target_dists,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            if policy_mems is not None:
                self._last_policy_mems = policy_mems.detach().clone()
            if model_mems is not None:
                self._last_model_mems = model_mems.detach().clone()
        
        buffer.compute_intrinsic_rewards()
        new_episodes = buffer.separate_episodes()
        pair_indices = buffer.sample_pairs(new_episodes, strategy)
        callback.on_rollout_end()
        return n_episodes, pair_indices

    def save_preference_data(self, data_path: str) -> None:
        # TODO: Save preference data to files [HUMAN FEEDBACK]
        # use dict with keys: 'pair_indices', 'pair_mask', 'pair_labels'
        pass

    def load_preference_data(self, data_path: str) -> None:
        # TODO: Load preference data from files [HUMAN FEEDBACK]
        # use dict with keys: 'pair_indices', 'pair_mask', 'pair_labels'
        pass

    def learn(self,
        episode_num: int,
        pair_num: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CustomOnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        use_synthetic_teacher: bool = True,
        teacher_config: Optional[dict] = None,
        generate_new_rollouts: bool = True,
        init: bool = True,
    ) -> None:
        if init : self.iteration = 0
        val_ratio = 0.125 if episode_num > 100 else 0.25
        assert episode_num > 0, "`episode_num` must be greater than 0"
        assert episode_num <= self.preference_buffer_capacity, "`episode_num` must be less than or equal to `preference_buffer_capacity`"
        assert episode_num * val_ratio >= 2, "Not enough episodes for validation rollouts."
        
        episode_num, callback = self._setup_learn(
                                            total_timesteps = episode_num, 
                                            callback=callback, 
                                            reset_num_timesteps=reset_num_timesteps, 
                                            tb_log_name=tb_log_name
                                        )
        self.episode_num = episode_num
        callback.on_training_start(locals(), globals())
        self.on_training_start()

        if use_synthetic_teacher:
            self._setup_synthetic_teacher(teacher_config)

        collect_start_time = time.time()
        if generate_new_rollouts or use_synthetic_teacher:
            # print('Collecting rollouts ...')
            train_strategy = self._setup_sampling_strategy(pair_num)
            n_episodes, train_pair_indices = self.collect_rollouts(
                self.env,
                callback,
                self.preference_buffer,
                episode_num=episode_num,
                strategy=train_strategy,
                verbose=self.verbose,
            )

        self.val_buffer = PreferenceBuffer(
            int(self.preference_buffer_capacity * val_ratio)*self.max_episode_length,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            features_dim=self.rl_policy.features_dim,
            dim_policy_traj=self.rl_policy.dim_policy_features,
            dim_model_traj=self.rl_policy.dim_model_features,
            int_rew_coef=self.int_rew_coef,
            int_rew_norm=self.int_rew_norm,
            int_rew_clip=self.int_rew_clip,
            int_rew_eps=self.int_rew_eps,
            gru_layers=self.rl_policy.gru_layers,
            int_rew_momentum=self.int_rew_momentum,
            use_status_predictor=self.rl_policy.use_status_predictor,
        )

        if generate_new_rollouts or use_synthetic_teacher:
            # print('Collecting validation rollouts ...')
            self.num_timesteps = 0
            self._episode_num = 0
            print(f'Collecting validation pairs: {int(train_pair_indices.shape[0] * val_ratio)} pairs')
            val_strategy = self._setup_sampling_strategy(int(train_pair_indices.shape[0] * val_ratio))
            _, val_pair_indices = self.collect_rollouts(
                self.env,
                callback,
                self.val_buffer,
                episode_num=int(episode_num * val_ratio),
                strategy=val_strategy,
                verbose=1,
            )
        collect_end_time = time.time()

        if generate_new_rollouts or use_synthetic_teacher:
            if self.verbose>0: print('Preparing preference pair data ...')
            # Prepare pair data for training and validation
            self.preference_buffer.prepare_pair_data(train_pair_indices, teacher=self.teacher)
            self.val_buffer.prepare_pair_data(val_pair_indices, teacher=self.teacher)
            if not use_synthetic_teacher:
                # Save the collected preference data for future use [HUMAN FEEDBACK]
                self.save_preference_data('path_to_save_data')  # TODO: specify the path
        else:
            print('Using saved preference pair data ...')
            train_data, val_data = self.load_preference_data('path_to_saved_data')  # TODO: specify the path
            self.preference_buffer.prepare_pair_data(pair_indices=train_data['pair_indices'], 
                                                     pair_mask=train_data['pair_mask'], 
                                                     pair_labels=train_data['pair_labels']
                                                     )
            self.val_buffer.prepare_pair_data(pair_indices=val_data['pair_indices'], 
                                               pair_mask=val_data['pair_mask'], 
                                               pair_labels=val_data['pair_labels']
                                               )
        
        # Uploading rollout infos
        self.iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, episode_num)
        self.log_on_rollout_end(log_interval)
        
        # Train the reward model
        train_start_time = time.time()
        epochs_completed, best_val_loss = self.train()
        train_end_time = time.time()
        
        # Print to the console
        rews = [ep_info["r"] for ep_info in self.ep_info_buffer]
        rew_mean = 0.0 if len(rews) == 0 else np.mean(rews)
        print(f'--RM-- '
              f'run: {self.run_id:2d}  '
              f'iters: {self.iteration}  '
              f'frames: {self.num_timesteps}  '
              f'eps: {n_episodes} '
              f'epochs: {epochs_completed}/{self.reward_epochs} '
              f'bestval: {best_val_loss:.6f}  '
              f'rollout: {collect_end_time - collect_start_time:.3f} sec  '
              f'train: {train_end_time - train_start_time:.3f} sec')
        callback.on_training_end()
        return self

    def add_rl_policy_models(self, rl_policy: ActorCriticPolicy, use_model_rnn: bool) -> None:
        rl_policy.eval()
        self.rl_policy = rl_policy
        # self.policy.add_encoder_models(rl_policy.model_cnn_extractor, 
        #                                    rl_policy.model_rnn_extractor, 
        #                                    use_model_rnn)

    def train(self) -> None:
        # Log training stats per each iteration
        self.training_stats = StatisticsLogger(mode='train')
    
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 5  # you can make this configurable
        best_model_state = None

        # Train Reward model
        for epoch in range(self.reward_epochs):
            # --- Training phase ---
            for traj_data in self.preference_buffer.get(self.reward_batch_size):
                self.policy.optimize(traj_data=traj_data, stats_logger=self.training_stats)
            train_losses=self.training_stats.data['rew_model_loss']
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # --- Validation phase ---
            val_losses = []
            for val_traj_data in self.val_buffer.get(self.reward_batch_size):  # separate validation buffer
                val_loss, _, _ = self.policy.evaluate(val_traj_data)
                val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

            if self.verbose>0: 
                print(f"[Epoch {epoch+1}/{self.reward_epochs}] Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

            # --- Early stopping check ---
            if avg_val_loss < best_val_loss - 1e-5:  # small threshold to avoid noise
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = {
                    k: v.detach().clone().cpu() for k, v in self.policy.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if self.verbose>0:
                        print(f"   Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    if best_model_state is not None:
                        self.policy.load_state_dict(best_model_state)
                    break

            # --- Logging ---
            log_data = {
                "time/epochs": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
            }
            # Update with other stats
            log_data.update(self.training_stats.to_dict())
            # Logging with wandb
            if self.use_wandb:
                wandb.log(log_data)
            # Logging with local logger
            if self.local_logger is not None:
                self.local_logger.write(log_data, log_type='rm_train')
            return epoch+1, best_val_loss

