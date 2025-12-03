import numpy as np
import torch as th
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium.spaces import Dict
from typing import Generator, Optional, Union

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize

from src.algo.buffers.type_aliases import PreferenceBufferSamples, TrajectoryData
from src.algo.reward_models.sampling_strategies.uniform import uniform_sampling
from src.utils.common_func import normalize_rewards
from src.utils.running_mean_std import RunningMeanStd

def pad_to_max_length(arr_list, pad_value=0.0):
    max_len = max(a.shape[0] for a in arr_list)
    # Determine full shape for padding
    sample_shape = arr_list[0].shape[1:]  # all dims except time
    padded = np.full((len(arr_list), max_len, *sample_shape), pad_value, dtype=np.float32)
    mask = np.zeros((len(arr_list), max_len), dtype=bool)
    for i, a in enumerate(arr_list):
        l = a.shape[0]
        padded[i, :l] = a
        mask[i, :l] = True
    return padded, mask

def discounted_sum(rewards, mask, gamma):
    """
    Compute discounted returns for padded reward sequences.
    rewards: (batch, max_len)
    mask: (batch, max_len) — True where valid
    """
    discounts = gamma ** np.arange(rewards.shape[1])
    discounted = rewards * discounts  # broadcasted across episodes
    # zero out padding
    discounted *= mask
    return discounted.sum(axis=1)

class PreferenceBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        features_dim: int = 0,
        dim_policy_traj: int = 0,
        dim_model_traj: int = 0,
        int_rew_coef: float = 1.0,
        ext_rew_coef: float = 1.0,
        ext_rew_pretrain_coef: float = 0.0,
        int_rew_norm: int = 0,
        int_rew_clip: float = 0.0,
        int_rew_eps: float = 1e-8,
        gru_layers: int = 1,
        int_rew_momentum: Optional[float] = None,
        use_status_predictor: int = 0,
        synthetic_teacher: bool = True,
        ):
        if isinstance(observation_space, Dict):
            observation_space = list(observation_space.values())[0]
        super(PreferenceBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs)
        self.int_rew_coef = int_rew_coef
        self.int_rew_norm = int_rew_norm
        self.int_rew_clip = int_rew_clip
        self.ext_rew_coef = ext_rew_coef
        self.ext_rew_pretrain_coef = ext_rew_pretrain_coef
        self.features_dim = features_dim
        self.dim_policy_traj = dim_policy_traj
        self.dim_model_traj = dim_model_traj
        self.int_rew_eps = int_rew_eps
        self.use_status_predictor = use_status_predictor
        self.gru_layers = gru_layers
        self.int_rew_momentum = int_rew_momentum
        self.int_rew_stats = RunningMeanStd(momentum=self.int_rew_momentum)
        self.synthetic_teacher = synthetic_teacher

        self.generator_ready = False
        self.first_update = True
        self.reset()


    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.last_policy_mems = np.zeros((self.buffer_size, self.n_envs, self.gru_layers, self.dim_policy_traj), dtype=np.float32)
        self.last_model_mems = np.zeros((self.buffer_size, self.n_envs, self.gru_layers, self.dim_model_traj), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if self.use_status_predictor:
            self.curr_key_status = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.curr_door_status = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.curr_target_dists = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)
        # self.temp_observations = np.zeros((self.buffer_size * self.n_envs,) + self.obs_shape, dtype=np.float32)
        # self.temp_new_observations = np.zeros((self.buffer_size * self.n_envs,) + self.obs_shape, dtype=np.float32)
        # self.temp_last_policy_mems = np.zeros((self.buffer_size * self.n_envs, self.gru_layers, self.dim_policy_traj), dtype=np.float32)
        # self.temp_last_model_mems = np.zeros((self.buffer_size * self.n_envs, self.gru_layers, self.dim_model_traj), dtype=np.float32)
        # self.temp_actions = np.zeros((self.buffer_size * self.n_envs, self.action_dim), dtype=np.float32)
        # self.temp_rewards = np.zeros((self.buffer_size * self.n_envs, 1), dtype=np.float32)
        # self.temp_intrinsic_rewards = np.zeros((self.buffer_size * self.n_envs, 1), dtype=np.float32)
        # self.temp_episode_starts = np.zeros((self.buffer_size * self.n_envs, 1), dtype=np.float32)
        # self.temp_episode_dones = np.zeros((self.buffer_size * self.n_envs, 1), dtype=np.float32)
        # if self.use_status_predictor:
        #     self.temp_curr_key_status = np.zeros((self.buffer_size * self.n_envs, 1), dtype=np.int32)
        #     self.temp_curr_door_status = np.zeros((self.buffer_size * self.n_envs, 1), dtype=np.int32)
        #     self.temp_curr_target_dists = np.zeros((self.buffer_size * self.n_envs, 3), dtype=np.float32)
        # self.generator_ready = False
        super(PreferenceBuffer, self).reset()
    

    def add(
        self,
        obs: np.ndarray,
        new_obs: np.ndarray,
        last_policy_mem: th.Tensor,
        last_model_mem: th.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        intrinsic_reward: np.ndarray,
        episode_start: np.ndarray,
        episode_done: np.ndarray,
        curr_key_status: Optional[np.ndarray],
        curr_door_status: Optional[np.ndarray],
        curr_target_dist: Optional[np.ndarray],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.new_observations[self.pos] = np.array(new_obs).copy()
        self.last_policy_mems[self.pos] = last_policy_mem.clone().cpu().numpy()
        self.last_model_mems[self.pos] = last_model_mem.clone().cpu().numpy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.intrinsic_rewards[self.pos] = np.array(intrinsic_reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.episode_dones[self.pos] = np.array(episode_done).copy()
        if self.use_status_predictor:
            self.curr_key_status[self.pos] = np.array(curr_key_status).copy()
            self.curr_door_status[self.pos] = np.array(curr_door_status).copy()
            self.curr_target_dists[self.pos] = np.array(curr_target_dist).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def get(self, batch_size: Optional[int] = None) -> Generator[PreferenceBufferSamples, None, None]:
        # print(f'Collected {self.pos}/{self.buffer_size} steps. {self.episode_dones.sum().item()} episodes.')
        num_generated_pairs = self.labels_tensor.shape[0]

        if batch_size is None:
            batch_size = min(num_generated_pairs // 4, 64) # 64 pairs per batch or 4 minibatches

        indices = np.random.permutation(num_generated_pairs)
        
        start_idx = 0
        end_idx = num_generated_pairs

        while start_idx < end_idx:
            # print(f'    Start_idx={start_idx}, end_idx={start_idx + batch_size}')
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> PreferenceBufferSamples:
        data1 = TrajectoryData( 
            curr_obs=self.obs1_tensor[batch_inds],
            actions=self.acts1_tensor[batch_inds],
            last_mems=self.mems1_tensor[batch_inds],
        )

        data2 = TrajectoryData(
            curr_obs=self.obs2_tensor[batch_inds],
            actions=self.acts2_tensor[batch_inds],
            last_mems=self.mems2_tensor[batch_inds],
        )

        labels = self.labels_tensor[batch_inds]

        return PreferenceBufferSamples(data1=data1, data2=data2, labels=labels)


    def compute_intrinsic_rewards(self) -> None:
        # Normalize intrinsic rewards per rollout buffer
        self.int_rew_stats.update(self.intrinsic_rewards.reshape(-1))
        self.int_rew_mean = self.int_rew_stats.mean
        self.int_rew_std = self.int_rew_stats.std
        self.intrinsic_rewards = normalize_rewards(
            norm_type=self.int_rew_norm,
            rewards=self.intrinsic_rewards,
            mean=self.int_rew_mean,
            std=self.int_rew_std,
            eps=self.int_rew_eps,
        )

        # Rescale by IR coef
        self.intrinsic_rewards *= self.int_rew_coef

        # Clip after normalization
        if self.int_rew_clip > 0:
            self.intrinsic_rewards = np.clip(self.intrinsic_rewards, -self.int_rew_clip, self.int_rew_clip)

    
    def separate_episodes(self) -> None:
        """
        Identify complete episodes in the collected rollout buffer
        and record their indices for fast trajectory-level access.
        Discards incomplete episodes at the end of collection.
        """
        dones = self.episode_dones[:self.pos]  # (steps, envs)
        starts = self.episode_starts[:self.pos]
        n_steps, n_envs = dones.shape

        # Flatten into a single axis
        flat_dones = dones.reshape(-1)
        flat_starts = starts.reshape(-1)
        step_ids = np.repeat(np.arange(n_steps), n_envs)
        env_ids = np.tile(np.arange(n_envs), n_steps)
        
        # Filter all starts and dones with their env+step
        start_mask = flat_starts.astype(bool)
        done_mask = flat_dones.astype(bool)

        start_envs = env_ids[start_mask]
        start_steps = step_ids[start_mask]
        done_envs = env_ids[done_mask]
        done_steps = step_ids[done_mask]

        # Sort by environment then step
        start_order = np.lexsort((start_steps, start_envs))
        done_order = np.lexsort((done_steps, done_envs))

        start_envs = start_envs[start_order]
        start_steps = start_steps[start_order]
        done_envs = done_envs[done_order]
        done_steps = done_steps[done_order]

        # Split by environment
        env_start_splits = np.split(start_steps, np.flatnonzero(np.diff(start_envs)) + 1)
        env_done_splits = np.split(done_steps, np.flatnonzero(np.diff(done_envs)) + 1)
        env_ids_unique = np.unique(start_envs)

        self.episode_index_dict = getattr(self, "episode_index_dict", {})
        next_id = getattr(self, "next_episode_id", 0)
        new_episodes = 0

        # Vectorized matching: first done after each start
        for env_idx, starts_env, dones_env in zip(env_ids_unique, env_start_splits, env_done_splits):
            # For each start, find first done > start
            # Broadcasting comparison -> (n_starts, n_dones)
            after_mask = dones_env[None, :] > starts_env[:, None]
            # Get index of first True per row (if any)
            first_done_idx = after_mask.argmax(axis=1)
            valid_mask = after_mask.any(axis=1)
            for s_idx, d_idx, valid in zip(starts_env, first_done_idx, valid_mask):
                if not valid:
                    continue  # incomplete episode
                e_idx = dones_env[d_idx]
                self.episode_index_dict[next_id] = {
                    "env": int(env_idx),
                    "start": int(s_idx),
                    "end": int(e_idx) + 1,
                }
                next_id += 1
                new_episodes += 1

        self.next_episode_id = next_id
        
        # Build arrays for fast indexing
        episode_starts = [v["start"] for v in self.episode_index_dict.values()]
        episode_ends   = [v["end"]   for v in self.episode_index_dict.values()]
        episode_envs   = [v["env"]   for v in self.episode_index_dict.values()]

        # extend arrays if already exist
        self.episode_ids = getattr(self, "episode_ids", np.array([], dtype=np.int32))
        self.episode_ids = np.concatenate((self.episode_ids, np.array(list(self.episode_index_dict.keys()), dtype=np.int32)), axis=0)
        # print("episode_index = ", np.array(list(zip(episode_starts, episode_ends)), dtype=np.int32).shape)
        self.episode_index = getattr(self, "episode_index", np.empty((0,2), dtype=np.int32))
        self.episode_index = np.concatenate((self.episode_index, np.array(list(zip(episode_starts, episode_ends)), dtype=np.int32)), axis=0)
        # print("episode_index = ", self.episode_index.shape)
        self.episode_envs = getattr(self, "episode_envs", np.array([], dtype=np.int32))
        self.episode_envs = np.concatenate((self.episode_envs, np.array(episode_envs, dtype=np.int32)), axis=0)
        return new_episodes


    def sample_pairs(self, new_episodes, strategy: dict) -> None:
        """
        Sample pairs of complete episodes for preference labeling according to the specified strategy.
        Each pair is a tuple of episode indices (i, j). Can control total n_pairs.
        """
        if not hasattr(self, "episode_index"):
            raise RuntimeError("Episodes have not been separated yet. Call separate_episodes() first.")

        n_episodes = len(self.episode_index)
        if n_episodes < 2:
            raise RuntimeError(f"Not enough episodes to create pairs: {n_episodes}")
        
        # Apply sampling strategy to generate pairs
        if strategy["name"] == "Uniform":
            pair_indices = uniform_sampling(self.episode_ids, new_episodes, strategy["n_pairs"])
        else:
            raise NotImplementedError(f"Sampling strategy '{strategy}' not implemented yet.")

        return pair_indices


    def prepare_pair_data(self, 
        pair_indices: np.ndarray,
        pair_mask: Optional[np.ndarray] = None,
        pair_labels: Optional[np.ndarray] = None, 
        teacher: Optional[dict] = None,
        debug: bool = False,
        ) -> None:
        """
        For each pair of episode indices, extract the episode data (observations, actions, last policy mems, rewards)
        and pad them to the maximum episode length in the batch.
        Then, generate preference labels using either a synthetic teacher model or real preferences.
        Store all data as tensors on the specified device.
        """
        # --- Extract episode rewards for each pair ---
        obs1_list, obs2_list = [], []
        acts1_list, acts2_list = [], []
        mems1_list, mems2_list = [], []
        r_t_1, r_t_2 = [], []
        pair_ids = []

        for idx1, idx2 in pair_indices:
            start1, end1 = self.episode_index[idx1]
            start2, end2 = self.episode_index[idx2]
            env1 = self.episode_envs[idx1]
            env2 = self.episode_envs[idx2]

            # Observations
            obs1_list.append(self.observations[start1:end1, env1])
            obs2_list.append(self.observations[start2:end2, env2])

            # Actions
            acts1_list.append(self.actions[start1:end1, env1])
            acts2_list.append(self.actions[start2:end2, env2])

            # Last policy memories
            mems1_list.append(self.last_policy_mems[start1:end1, env1])
            mems2_list.append(self.last_policy_mems[start2:end2, env2])

            # Rewards
            r_t_1.append(self.rewards[start1:end1, env1])
            r_t_2.append(self.rewards[start2:end2, env2])
            
            pair_ids.append((idx1, idx2))

        # Pad to longest episode and convert to arrays
        obs1_padded, _ = pad_to_max_length(obs1_list)
        obs2_padded, _ = pad_to_max_length(obs2_list)
        acts1_padded, _ = pad_to_max_length(acts1_list)
        acts2_padded, _ = pad_to_max_length(acts2_list)
        mems1_padded, _ = pad_to_max_length(mems1_list)
        mems2_padded, _ = pad_to_max_length(mems2_list)
        r_t_1, mask1 = pad_to_max_length(r_t_1)
        r_t_2, mask2 = pad_to_max_length(r_t_2)
        pair_ids = np.array(pair_ids)

        # Initialize labels as -1 for all new pairs
        labels = np.full(len(pair_ids), -1, dtype=float)

        if self.synthetic_teacher:
            # --- Generate synthetic preferences only for unlabeled pairs ---
            pair_mask, pair_labels = self._generate_synthetic_preferences(pair_ids, r_t_1, r_t_2, mask1, mask2, labels, teacher, debug=debug)
        
        labels[pair_mask] = pair_labels
        # --- Convert all to tensors on device ---
        self.obs1_tensor = th.tensor(obs1_padded, dtype=th.float32).to(self.device)
        self.obs2_tensor = th.tensor(obs2_padded, dtype=th.float32).to(self.device)
        self.acts1_tensor = th.tensor(acts1_padded, dtype=th.float32).to(self.device)
        self.acts2_tensor = th.tensor(acts2_padded, dtype=th.float32).to(self.device)
        self.mems1_tensor = th.tensor(mems1_padded, dtype=th.float32).to(self.device)
        self.mems2_tensor = th.tensor(mems2_padded, dtype=th.float32).to(self.device)
        self.r_t_1_tensor = th.tensor(r_t_1, dtype=th.float32).to(self.device)
        self.r_t_2_tensor = th.tensor(r_t_2, dtype=th.float32).to(self.device)
        self.labels_tensor = th.tensor(labels, dtype=th.float32).to(self.device)

    def _generate_synthetic_preferences(self, pair_ids, r_t_1, r_t_2, mask1, mask2, labels, teacher: dict, debug: bool=False) -> None:
        """
        filter ids of trajectories that already have preferences label=0, 1, 0.5, 
        keeping only those with label=-1. Then label them according to the synthetic teacher model.
        1. gamma: discounting factor that makes the teacher pay attention to the later part of the trajectory
        2. beta: rationality parameter
        3. eps_mistake: chance of making a mistake
        4. thres_equal: threshold for considering two trajectories equally preferable
        """
        pair_mask = (labels == -1)
        pair_ids = pair_ids[pair_mask]
        r_t_1 = r_t_1[pair_mask]
        r_t_2 = r_t_2[pair_mask]

        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < teacher["thres_equal"]).reshape(-1)
        sum_r_t_1 = discounted_sum(r_t_1, mask1, teacher["gamma"])
        sum_r_t_2 = discounted_sum(r_t_2, mask2, teacher["gamma"])
        
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if teacher["beta"] > 0: # Bradley-Terry rational model
            r_hat = th.cat([th.Tensor(sum_r_t_1), 
                            th.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*teacher["beta"]
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = th.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
    
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= teacher["eps_mistake"]
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels = labels.astype(float)
        labels[margin_index] = 0.5

        # --- Sanity checks ---
        if debug:
            print("\n[Sanity Check — Synthetic Preferences]")
            print(margin_index.sum(), " pairs labeled as equally preferable (0.5) due to threshold.")
            print(f"r_t_1 shape: {np.shape(r_t_1)}, r_t_2 shape: {np.shape(r_t_2)}")
            print(f"mask1 shape: {np.shape(mask1)}, mask2 shape: {np.shape(mask2)}")
            print(f"labels shape: {np.shape(labels)}  (unique values: {np.unique(labels, return_counts=True)})")

            # Verify padding consistency
            valid_steps1 = mask1.sum(axis=1)
            valid_steps2 = mask2.sum(axis=1)
            print(f"Valid steps per episode: r_t_1 mean={valid_steps1.mean():.1f}, r_t_2 mean={valid_steps2.mean():.1f}")

            # Check for NaNs or extreme values
            for name, arr in [("r_t_1", r_t_1), ("r_t_2", r_t_2)]:
                if np.isnan(arr).any():
                    print(f"⚠️ Warning: {name} contains NaN values")
                print(f"{name} min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")

            # Check label balance
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("Label distribution:")
            for lbl, cnt in zip(unique_labels, counts):
                print(f"  Label {lbl}: {cnt} ({100*cnt/len(labels):.1f}%)")
        return pair_mask, labels
        
