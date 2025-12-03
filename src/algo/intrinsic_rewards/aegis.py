import gymnasium as gym
from typing import Dict, Any
import time
import numpy as np
from gymnasium import spaces
from numpy.random import Generator
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor

from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType, ModelType
from src.utils.running_mean_std import RunningMeanStd

class AEGIS(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        # Method-specific params
        obs_rng: Optional[Generator] = None,
        lmdp_obs_queue_len: int = 0,
        log_lmdp_verbose: int = 0,

        aegis_knn_k: int = 5,
        aegis_nem_capacity: int = 512,
        aegis_dst_momentum: float = 0.9,
        aegis_l_coef: float = 0.5,
        aegis_g_coef: float = 0.5,
        aegis_novelty_alpha: float = 0.5,
        aegis_novelty_beta: float = 0.0,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)
        
        self.obs_rng = obs_rng
        self.lmdp_obs_queue_len = lmdp_obs_queue_len
        self.log_lmdp_verbose = log_lmdp_verbose

        self.aegis_knn_k = aegis_knn_k
        self.aegis_nem_capacity = aegis_nem_capacity
        self.aegis_dst_momentum = aegis_dst_momentum
        self.aegis_moving_avg_dists = RunningMeanStd(momentum=self.aegis_dst_momentum)
        self.novelty_alpha = aegis_novelty_alpha
        self.novelty_beta = aegis_novelty_beta
        self._init_obs_queue()

        self._build()
        self._init_modules()
        self._init_optimizers()


    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()

        # Build MLP
        self.model_mlp = AegisModelOutputHeads(
            features_dim=self.model_features_dim,
            latents_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            action_num=self.action_num,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
        )


    def _get_fake_obs(self, curr_obs, next_obs):    
        """
        In order to prepare negative samples for the discriminative model's training,
        this method randomly selects two fake observations from the observation queue
        and returns the one that differs from the positive training sample. If both are
        identical to a positive sample, then `obs_diff` is 0 and can be used as a signal
        to invalidate that sample when calculating training losses.
        """
        queue_len = min(self.obs_queue_filled, self.lmdp_obs_queue_len)
        batch_size = curr_obs.shape[0]

        # Randomly select two fake observations from the queue
        random_idx1 = self.obs_rng.integers(low=0, high=queue_len, size=batch_size, dtype=int)
        random_idx2 = self.obs_rng.integers(low=0, high=queue_len, size=batch_size, dtype=int)
        random_obs1 = obs_as_tensor(self.obs_queue[random_idx1], curr_obs.device)
        random_obs2 = obs_as_tensor(self.obs_queue[random_idx2], curr_obs.device)

        # `obs_diff{1,2}`: whether the ture observation at t+1 (`next_obs`)
        #                  differs from the {1st,2nd} fake sample (`random_obs{1,2}`)
        obs_diff1 = th.abs(next_obs - random_obs1).sum((1, 2, 3))
        obs_diff2 = th.abs(next_obs - random_obs2).sum((1, 2, 3))
        obs_diff1 = th.gt(obs_diff1, th.zeros_like(obs_diff1)).long().view(-1, 1, 1, 1)
        obs_diff2 = th.gt(obs_diff2, th.zeros_like(obs_diff2)).long().view(-1, 1, 1, 1)
        obs_diff = th.logical_or(obs_diff1, obs_diff2).long().view(-1, 1, 1, 1)

        # return `random_obs1` when `next_obs` differs from `random_obs1`, otherwise `random_obs2`
        rand_obs = random_obs1 * obs_diff1 + random_obs2 * (1 - obs_diff1)
        return rand_obs, obs_diff
    

    def _init_obs_queue(self):
        self.obs_shape = get_obs_shape(self.observation_space)
        self.obs_queue_filled = 0
        self.obs_queue_pos = 0
        self.obs_queue = np.zeros((self.lmdp_obs_queue_len,) + self.obs_shape, dtype=float)


    def _get_lmdp_embeddings(self, curr_obs, next_obs, last_mems, device=None):
        if not isinstance(curr_obs, Tensor):
            curr_obs = obs_as_tensor(curr_obs, device)
            next_obs = obs_as_tensor(next_obs, device)

        # Get CNN embeddings
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)

        # If RNN enabled
        if self.use_model_rnn:
            curr_mems = self._get_rnn_embeddings(last_mems, curr_cnn_embs, self.model_rnns)
            next_mems = self._get_rnn_embeddings(curr_mems, next_cnn_embs, self.model_rnns)
            curr_rnn_embs = th.squeeze(curr_mems[:, -1, :])
            next_rnn_embs = th.squeeze(next_mems[:, -1, :])
            return curr_cnn_embs, next_cnn_embs, curr_rnn_embs, next_rnn_embs, curr_mems

        # If RNN disabled
        return curr_cnn_embs, next_cnn_embs, curr_cnn_embs, next_cnn_embs, None


    def _get_training_losses(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor,
        obs_diff: Tensor, labels: Tensor,
        key_status: Optional[Tensor],
        door_status: Optional[Tensor],
        target_dists: Optional[Tensor],
        int_rew_source=ModelType.AEGIS
    ):
        # Count valid samples in a batch. A transition (o_t, a_t, o_t+1) is deemed invalid if:
        # 1) an episode ends at t+1, or 2) the ture sample is identical to the fake sample selected at t+1
        n_half_batch = curr_dones.shape[0] // 2
        valid_pos_samples = (1 - curr_dones[n_half_batch:].view(-1)).long()
        valid_neg_samples = th.logical_and(valid_pos_samples, obs_diff.view(-1)).long()
        n_valid_pos_samples = valid_pos_samples.sum().long().item()
        n_valid_neg_samples = valid_neg_samples.sum().long().item()
        n_valid_samples = n_valid_pos_samples + n_valid_neg_samples
        pos_loss_factor = 1 / n_valid_pos_samples if n_valid_pos_samples > 0 else 0.0
        neg_loss_factor = 1 / n_valid_neg_samples if n_valid_neg_samples > 0 else 0.0

        # Get embeddings
        _, _, curr_embs, next_embs, _ = self._get_lmdp_embeddings(curr_obs, next_obs, last_mems)

        # Inverse model prediction
        pred_obs, pred_act, likelihoods = self.model_mlp(curr_embs, next_embs, curr_act)

        # Get likelihoods
        likelihoods = th.sigmoid(likelihoods.view(-1)).view(-1)

        # Discriminator loss
        pos_dsc_losses = F.binary_cross_entropy(likelihoods[:n_half_batch], labels[:n_half_batch], reduction='none')
        neg_dsc_losses = F.binary_cross_entropy(likelihoods[n_half_batch:], labels[n_half_batch:], reduction='none')
        pos_dsc_loss = (pos_dsc_losses.view(-1) * valid_pos_samples).sum() * pos_loss_factor
        neg_dsc_loss = (neg_dsc_losses.view(-1) * valid_neg_samples).sum() * neg_loss_factor

        # Balance positive and negative samples
        if 0 < n_valid_pos_samples < n_valid_neg_samples:
            pos_dsc_loss *= n_valid_neg_samples / n_valid_pos_samples
        if 0 < n_valid_neg_samples < n_valid_pos_samples:
            neg_dsc_loss *= n_valid_pos_samples / n_valid_neg_samples

        # Get contrastive loss
        contrastive_loss = (pos_dsc_loss + neg_dsc_loss) * 0.5
        
        curr_dones = curr_dones[:n_half_batch].view(-1)
        n_samples = (1 - curr_dones).sum()
        
        # Inverse loss
        inv_losses = F.cross_entropy(pred_act[:n_half_batch], curr_act[:n_half_batch], reduction='none') * (1 - curr_dones)
        inv_loss = inv_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        # Forward model
        fwd_losses = 0.5 * F.mse_loss(pred_obs[:n_half_batch], next_embs[:n_half_batch], reduction='none') \
                         * self.model_features_dim  # eta (scaling factor)
        fwd_losses = fwd_losses.mean(dim=1) * (1 - curr_dones)
        fwd_loss = fwd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        if int_rew_source == ModelType.AEGIS_local_only:
            lmdp_loss =  contrastive_loss
        elif int_rew_source == ModelType.AEGIS_global_only:
            lmdp_loss =  0.2*inv_loss + 0.2*fwd_loss
        else: # ModelType.AEGIS or other variation
            lmdp_loss = 0.2*inv_loss + 0.2*fwd_loss + contrastive_loss

        if self.log_lmdp_verbose:
            with th.no_grad():
                pos_avg_likelihood = (likelihoods[:n_half_batch].view(-1) * valid_pos_samples).sum() * pos_loss_factor
                neg_avg_likelihood = (likelihoods[n_half_batch:].view(-1) * valid_neg_samples).sum() * neg_loss_factor
                avg_likelihood = (pos_avg_likelihood + neg_avg_likelihood) * 0.5
                dsc_accuracy = 1 - th.abs(likelihoods - labels).sum() / likelihoods.shape[0]
        else:
            avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy = None, None, None, None

        if self.use_status_predictor:
            key_loss, door_loss, pos_loss, key_dist, door_dist, goal_dist = \
                self._get_status_prediction_losses(curr_embs, key_status, door_status, target_dists)
        else:
            key_loss, door_loss, pos_loss, key_dist, door_dist, goal_dist = [self.constant_zero] * 6

        return lmdp_loss, pos_dsc_loss, neg_dsc_loss, \
               key_loss, door_loss, pos_loss, \
               key_dist, door_dist, goal_dist, \
               n_valid_samples, n_valid_pos_samples, n_valid_neg_samples, \
               avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy
    

    def _add_obs(self, obs):
        """
        Add one new element into the episodic observation queue.
        """
        self.obs_queue[self.obs_queue_pos] = np.copy(obs)
        self.obs_queue_filled += 1
        self.obs_queue_pos += 1
        self.obs_queue_pos %= self.lmdp_obs_queue_len


    def init_obs_queue(self, obs_arr):
        """
        In order to ensure the observation queue is not empty on training start
        by adding all observations received at time step 0.
        """
        for obs in obs_arr:
            self._add_obs(obs)
 

    def update_obs_queue(self, iteration, intrinsic_rewards, ir_mean, new_obs, stats_logger):
        """
        Update the observation queue after generating the intrinsic rewards for
        the current RL rollout.
        """
        for env_id in range(new_obs.shape[0]):
            if iteration == 0 or intrinsic_rewards[env_id] >= ir_mean:
                obs = new_obs[env_id]
                self._add_obs(obs)
                stats_logger.add(obs_insertions=1)
            else:
                stats_logger.add(obs_insertions=0)


    def get_intrinsic_rewards(self,
        curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history, trj_history, last_global_novelty, stats_logger, int_rew_source=ModelType.AEGIS
    ):
        with th.no_grad():
            curr_cnn_embs, next_cnn_embs, \
            curr_rnn_embs, next_rnn_embs, model_mems = \
                self._get_lmdp_embeddings(
                    curr_obs, next_obs, last_mems
                )
            # Inverse model prediction
            pred_obs, pred_act, _ = self.model_mlp(curr_rnn_embs, next_rnn_embs, curr_act)

            # # Inverse loss
            # curr_dones = curr_dones.view(-1)
            # inv_losses = F.cross_entropy(pred_act, curr_act, reduction='none') * (1 - curr_dones)
            # inv_loss = inv_losses.clone().cpu().numpy()

            # Forward loss
            fwd_losses = 0.5 * F.mse_loss(pred_obs, next_rnn_embs, reduction='none') \
                            * self.model_features_dim  # eta (scaling factor)
            fwd_losses = fwd_losses.mean(dim=1) * (1 - curr_dones)
            fwd_loss = fwd_losses.clone().cpu().numpy()

            # global_novelty = inv_loss
            # global_novelty = 0.5 * (inv_loss + fwd_loss)
            global_novelty = fwd_loss

        batch_size = curr_obs.shape[0]
        int_rews = np.zeros(batch_size, dtype=np.float32)
        local_rewards = np.zeros(batch_size, dtype=np.float32)
        global_rewards = np.zeros(batch_size, dtype=np.float32)
        for env_id in range(batch_size):
            # Update the episodic history of observation embeddings
            curr_obs_emb = curr_cnn_embs[env_id].view(1, -1)
            next_obs_emb = next_cnn_embs[env_id].view(1, -1)
            obs_embs = obs_history[env_id]
            new_embs = [curr_obs_emb, next_obs_emb] if obs_embs is None else [obs_embs, next_obs_emb]
            obs_embs = th.cat(new_embs, dim=0)
            obs_history[env_id] = obs_embs
            obs_dists = self.calc_euclidean_dists(obs_embs[:-1], obs_embs[-1])
            
            # Update the episodic history of trajectory embeddings
            if curr_rnn_embs is not None:
                curr_trj_emb = curr_rnn_embs[env_id].view(1, -1)
                next_trj_emb = next_rnn_embs[env_id].view(1, -1)
                trj_embs = trj_history[env_id]
                new_embs = [th.zeros_like(curr_trj_emb), curr_trj_emb, next_trj_emb] if trj_embs is None else [trj_embs, next_trj_emb]
                trj_embs = th.cat(new_embs, dim=0)
                trj_history[env_id] = trj_embs
                trj_dists = self.calc_euclidean_dists(trj_embs[:-2], trj_embs[-2])
            else:
                trj_dists = th.ones_like(obs_dists)

            # Generate local intrinsic reward
            # AEGIS: Equation 4 in the main paper
            mi_dists = th.pow(obs_dists, 2.0) / (trj_dists + 1e-6)
            local_rewards[env_id] += mi_dists.min().item()

            # Generate global intrinsic reward
            global_rewards[env_id] = max(global_novelty[env_id] - last_global_novelty[env_id] * self.novelty_alpha, self.novelty_beta) + 1.0

            # AEGIS: Equation 9 in the main paper
            if int_rew_source == ModelType.AEGIS_local_only:
                int_rews[env_id] += local_rewards[env_id]
            elif int_rew_source == ModelType.AEGIS_global_only:
                int_rews[env_id] += global_rewards[env_id]
            else:
                int_rews[env_id] += local_rewards[env_id] * global_rewards[env_id]

        # Logging
        stats_logger.add(
            int_rews=int_rews,
            local_rewards=local_rewards,
            global_rewards=global_rewards,
        )
        return int_rews, model_mems, global_novelty


    def optimize(self, rollout_data, stats_logger, int_rew_source=ModelType.AEGIS):
        # Prepare input data
        with th.no_grad():
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()
            label_ones = th.ones_like(rollout_data.episode_dones)
            label_zeros = th.zeros_like(rollout_data.episode_dones)
            pred_labels = th.cat([label_ones, label_zeros], dim=0)
            curr_obs = rollout_data.observations
            next_obs = rollout_data.new_observations
            fake_obs, obs_differences = self._get_fake_obs(curr_obs, next_obs)
            key_status, door_status, target_dists = None, None, None
            if self.use_status_predictor:
                key_status = rollout_data.curr_key_status
                door_status = rollout_data.curr_door_status
                target_dists = rollout_data.curr_target_dists

        lmdp_loss, pos_dsc_loss, neg_dsc_loss, \
        key_loss, door_loss, pos_loss, \
        key_dist, door_dist, goal_dist, \
        n_valid_samples, n_valid_pos_samples, n_valid_neg_samples, \
        avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy = \
            self._get_training_losses(
                curr_obs.tile((2, 1, 1, 1)),
                th.cat([next_obs, fake_obs], dim=0),
                rollout_data.last_model_mems.tile((2, 1, 1)),
                actions.tile(2),
                rollout_data.episode_dones.tile((2, 1)).long().view(-1),
                obs_differences,
                pred_labels.float().view(-1),
                key_status,
                door_status,
                target_dists,
                int_rew_source=int_rew_source
            )
        
        self.model_optimizer.zero_grad()
        lmdp_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()

        # Train the status predictor(s) for analysis experiments
        if self.use_status_predictor:
            predictor_loss = key_loss + door_loss + pos_loss
            self.predictor_optimizer.zero_grad()
            predictor_loss.backward()
            self.predictor_optimizer.step()

        # Logging
        stats_logger.add(
            lmdp_loss=lmdp_loss,
            pos_dsc_loss=pos_dsc_loss,
            neg_dsc_loss=neg_dsc_loss,
            avg_likelihood=avg_likelihood,
            pos_avg_likelihood=pos_avg_likelihood,
            neg_avg_likelihood=neg_avg_likelihood,
            dsc_accuracy=dsc_accuracy,
            key_loss=key_loss,
            door_loss=door_loss,
            pos_loss=pos_loss,
            key_dist=key_dist,
            door_dist=door_dist,
            goal_dist=goal_dist,
            n_valid_samples=n_valid_samples,
            n_valid_pos_samples=n_valid_pos_samples,
            n_valid_neg_samples=n_valid_neg_samples,
        )