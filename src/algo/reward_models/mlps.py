import torch as th

from torch import Tensor, nn
import torch.nn.functional as F
from typing import Type, Tuple
from src.utils.enum_types import NormType
import torch

class RewardMemberModel(nn.Module):
    def __init__(self,
                model_features_dim: int,
                model_latents_dim: int = 128,
                activation_fn: Type[nn.Module] = nn.ReLU,
                action_num: int = 0,
                model_mlp_norm: NormType = NormType.NoNorm,
                model_mlp_layers: int = 1,
    ):
        super().__init__()
        self.model_features_dim = model_features_dim
        self.model_latents_dim = model_latents_dim
        self.activation_fn = activation_fn
        self.action_num = action_num
        self.model_mlp_norm = model_mlp_norm
        self.model_mlp_layers = model_mlp_layers

        modules = [
            nn.Linear(model_features_dim + action_num, model_latents_dim),
            NormType.get_norm_layer_1d(model_mlp_norm, model_latents_dim),
            activation_fn(),
        ]
        for _ in range(1, model_mlp_layers):
            modules += [
                nn.Linear(model_latents_dim, model_latents_dim),
                NormType.get_norm_layer_1d(model_mlp_norm, model_latents_dim),
                activation_fn(),
            ]
        modules.append(nn.Linear(model_latents_dim, 1))
        self.nn = nn.Sequential(*modules)

    def forward(self, curr_emb: Tensor, curr_act: Tensor) -> Tensor:
        one_hot_actions = F.one_hot(curr_act, num_classes=self.action_num)
        inputs = th.cat([curr_emb, one_hot_actions], dim=1)
        return self.nn(inputs)