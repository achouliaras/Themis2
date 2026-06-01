from typing import Type, Tuple, Optional, Dict, Any, List
from src.algo.reward_models.base_reward_model import BaseRewardModel
from gymnasium import spaces

import torch as th
from torch import Tensor, nn
from torch.nn import GRUCell
import torch.nn.functional as F
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor
from src.utils.enum_types import NormType
from src.algo.reward_models.mlps import RewardMemberModel
from src.utils.common_func import init_module_with_name
from src.utils.enum_types import NormType

class MLPEnsemble(BaseRewardModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
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
        # Method-specific params
        ensemble_size: int = 3,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
            optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
            model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
            model_features_dim, model_latents_dim, model_mlp_norm, model_cnn_norm,
            model_gru_norm, use_model_rnn, model_mlp_layers, gru_layers)
        self.ensemble_size = ensemble_size

        self._build()
        self._init_modules()
        self._init_optimizers()


    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()

        self.ensemble_nn = nn.ModuleList()
        for i in range(self.ensemble_size):
            member = RewardMemberModel(
                model_features_dim=self.model_features_dim,
                model_latents_dim=self.model_latents_dim,
                activation_fn=self.activation_fn,
                action_num=self.action_num,
                model_mlp_norm=self.model_mlp_norm,
                model_mlp_layers=self.model_mlp_layers,
            )
            self.ensemble_nn.append(member)

    def init_encoder_from_rl_policy(self, rl_policy) -> None:
        self.model_cnn_extractor.load_state_dict(rl_policy.features_extractor.state_dict())
        if self.use_model_rnn:
            for i, module in enumerate(self.model_rnns):
                self.model_rnns[i].load_state_dict(rl_policy.policy_rnns[i].state_dict())
    
    def freeze_encoders(self) -> None:
        # Freeze the CNN encoder weights
        for param in self.model_cnn_extractor.parameters():
            param.requires_grad = False

        # Freeze the RNN weights (if applicable)
        if self.use_model_rnn:
            for rnn in self.model_rnns:
                for param in rnn.parameters():
                    param.requires_grad = False
            
    
    def _get_embedding(self, 
        curr_obs: Tensor, 
        last_mems: Optional[Tensor] = None, 
        device: Optional[th.device] = None
    ) -> Tensor:
        if not isinstance(curr_obs, Tensor):
            curr_obs = obs_as_tensor(curr_obs, device)

        with th.no_grad():
            # Get CNN embeddings
            curr_cnn_embs = self._get_cnn_embeddings(curr_obs)

        # If RNN enabled
        if self.use_model_rnn:
            with th.no_grad():
                curr_mems = self._get_rnn_embeddings(last_mems, curr_cnn_embs, self.model_rnns)
                curr_rnn_embs = th.squeeze(curr_mems[:, -1, :])
            return curr_cnn_embs, curr_rnn_embs, curr_mems

        # If RNN disabled
        return curr_cnn_embs, None, None


    def r_hat(self, curr_obs: Tensor, curr_act: Tensor, last_mems: Optional[Tensor] = None, get_disagreement: bool = False) -> Tensor:
        # Return the average rewards from all ensemble members for a single (obs, act) pair. 
        # If get_disagreement is True, also return the variance across members as an uncertainty measure.
        curr_cnn_embs, curr_rnn_embs, _ = self._get_embedding(curr_obs, last_mems)
        curr_embs = curr_rnn_embs if self.use_model_rnn else curr_cnn_embs
        
        r_hat_all = th.stack([m(curr_embs, curr_act) for m in self.ensemble_nn])
        mean_r_hat = r_hat_all.mean(dim=0).squeeze(-1) # [batch_size]
        mean_r_hat_squashed = None
        mean_r_hat_squashed = th.sigmoid(mean_r_hat)  # Squash to [0, 1] for reward scaling

        r_hat = mean_r_hat_squashed if mean_r_hat_squashed is not None else mean_r_hat
        if get_disagreement:
            disagreement = r_hat_all.var(dim=0)
            return r_hat, disagreement
        return r_hat  # mean over ensemble members


    def r_hat_members(self, curr_obs: Tensor, curr_act: Tensor, last_mems: Optional[Tensor] = None) -> Tensor:
        curr_cnn_embs, curr_rnn_embs, _ = self._get_embedding(curr_obs, last_mems)
        curr_embs = curr_rnn_embs if self.use_model_rnn else curr_cnn_embs
        return th.stack([m(curr_embs, curr_act) for m in self.ensemble_nn])
        
    
    def _get_training_losses(self, 
        curr_obs1: Tensor, curr_obs2: Tensor,
        curr_act1: Tensor, curr_act2: Tensor,
        labels: Tensor,
        last_mems1: Optional[Tensor] = None, last_mems2: Optional[Tensor] = None
    ) -> Tensor:
        batch_size, ep_len = curr_obs1.shape[:2]
                
        # Flatten for network forward
        batch_obs1 = curr_obs1.reshape(batch_size * ep_len, *curr_obs1.shape[2:]).to(labels.device)
        batch_act1 = curr_act1.reshape(batch_size * ep_len, *curr_act1.shape[2:]).to(labels.device)
        batch_obs2 = curr_obs2.reshape(batch_size * ep_len, *curr_obs2.shape[2:]).to(labels.device)
        batch_act2 = curr_act2.reshape(batch_size * ep_len, *curr_act2.shape[2:]).to(labels.device)
        batch_mems1 = last_mems1.reshape(batch_size * ep_len, *last_mems1.shape[2:]).to(labels.device) if last_mems1 is not None else None
        batch_mems2 = last_mems2.reshape(batch_size * ep_len, *last_mems2.shape[2:]).to(labels.device) if last_mems2 is not None else None

        # print(f"Batch Curr Obs: {batch_obs1.shape}, {batch_obs2.shape}")
        # print(f"Batch Actions: {batch_act1.shape}, {batch_act2.shape}")
        # print(f"Batch Last Mems: {batch_mems1.shape}, {batch_mems2.shape}")
        # print(f"Labels: {labels.shape}")

        # Forward through reward model
        r_hat1 = self.r_hat_members(batch_obs1, batch_act1, batch_mems1) # [ensemble_size, batch_size * ep_len]
        r_hat2 = self.r_hat_members(batch_obs2, batch_act2, batch_mems2) # [ensemble_size, batch_size * ep_len]
        r_hat1 = r_hat1.view(self.ensemble_size, batch_size, ep_len) # [ensemble_size, batch_size, ep_len]
        r_hat2 = r_hat2.view(self.ensemble_size, batch_size, ep_len) # [ensemble_size, batch_size, ep_len]
        r_hat1_mean = r_hat1.mean(dim=0) # [batch_size, ep_len]
        r_hat2_mean = r_hat2.mean(dim=0) # [batch_size, ep_len]
        
        r_hat1_traj = r_hat1_mean.sum(dim=1) # [batch]
        r_hat2_traj = r_hat2_mean.sum(dim=1) # [batch]

        # Compute difference and loss
        r_diff = r_hat1_traj - r_hat2_traj  # [batch_size]
        main_loss = F.binary_cross_entropy_with_logits(
            r_diff,
            labels,
            reduction='mean'
        )
        
        # === Diagnostics: per-member BCEs ===
        loss_per_member = []
        for i in range(self.ensemble_size):
            r_diff_i = r_hat1[i].sum(dim=(1)) - r_hat2[i].sum(dim=(1))  # [batch_size]
            l_i = F.binary_cross_entropy_with_logits(r_diff_i, labels, reduction="mean")
            loss_per_member.append(l_i)
        loss_per_member = th.stack(loss_per_member)  # [ensemble_size]

        # === Ensemble diversity metric (variance of per-member predictions) ===
        diversity_penalty = (
            (r_hat1.var(dim=0).mean() + r_hat2.var(dim=0).mean())
        )

        # You can optionally regularize with it:
        # loss = main_loss + alpha * diversity_penalty
        loss = main_loss
        return loss, loss_per_member, diversity_penalty


    def optimize(self, traj_data, stats_logger):
        traj_data1 = traj_data.data1
        traj_data2 = traj_data.data2

        actions1, actions2 = traj_data1.actions, traj_data2.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions1 = traj_data1.actions.long().flatten()
            actions2 = traj_data2.actions.long().flatten()

        rew_model_loss, loss_per_member, rew_model_diversity = self._get_training_losses(
                traj_data1.curr_obs, traj_data2.curr_obs,
                actions1, actions2,
                traj_data.labels,
                traj_data1.last_mems, traj_data2.last_mems,
            )
        
        self.model_optimizer.zero_grad()
        rew_model_loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.model_optimizer.step()

        stats_logger.add(rew_model_loss=rew_model_loss.item())
        for i, member in enumerate(self.ensemble_nn):
            stats_logger.add(
                **{f'rew_model_loss_member_{i}': loss_per_member[i].item()}
            )
        stats_logger.add(rew_model_diversity=rew_model_diversity.item())


    def evaluate(self, traj_data) -> Tensor:
        traj_data1 = traj_data.data1
        traj_data2 = traj_data.data2

        actions1, actions2 = traj_data1.actions, traj_data2.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions1 = traj_data1.actions.long().flatten()
            actions2 = traj_data2.actions.long().flatten()

        with th.no_grad():
            rew_model_loss, loss_per_member, rew_model_diversity = self._get_training_losses(
                traj_data1.curr_obs, traj_data2.curr_obs,
                actions1, actions2,
                traj_data.labels,
                traj_data1.last_mems, traj_data2.last_mems,
            )
        return rew_model_loss, loss_per_member, rew_model_diversity


    def save(self, path: str) -> None:
        super().save(path, additional_data={"ensemble_size": self.ensemble_size})

    @classmethod
    def load(cls, path: str, device=None, ensemble_size=None) -> None:
        return super(MLPEnsemble, cls).load(path, device=device, additional_data={"ensemble_size": ensemble_size})
    