import gymnasium as gym
import numpy as np

from torch.nn import GRUCell
from typing import Dict, Any, List

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor

from src.algo.common_models.gru_cell import CustomGRUCell
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType
from src.utils.common_func import init_module_with_name


class BaseRewardModel(nn.Module):
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
    ):
        super().__init__()
        if isinstance(observation_space, gym.spaces.Dict):
            observation_space = observation_space["rgb"]
        self.observation_space = observation_space
        self.normalize_images = normalize_images
        self.action_space = action_space
        self.action_num = action_space.n
        self.max_grad_norm = max_grad_norm
        self.model_features_dim = model_features_dim
        self.model_latents_dim = model_latents_dim
        self.model_learning_rate = model_learning_rate
        self.model_mlp_norm = model_mlp_norm
        self.model_cnn_norm = model_cnn_norm
        self.model_gru_norm = model_gru_norm
        self.model_mlp_layers = model_mlp_layers
        self.gru_layers = gru_layers
        self.model_gru_cell = GRUCell if self.model_gru_norm == NormType.NoNorm else CustomGRUCell
        self.use_model_rnn = use_model_rnn
        self.model_cnn_features_extractor_class = model_cnn_features_extractor_class
        self.model_cnn_features_extractor_kwargs = model_cnn_features_extractor_kwargs
        self.activation_fn = activation_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.model_rnn_kwargs = dict(
            input_size=self.model_features_dim,
            hidden_size=self.model_features_dim,
        )
        if self.model_gru_norm != NormType.NoNorm:
            self.model_rnn_kwargs.update(dict(
                norm_type=self.model_gru_norm,
            ))

        self.constant_zero = th.zeros(1, dtype=th.float)
        self.constant_one = th.ones(1, dtype=th.float)


    def _build(self) -> None:
        self.model_cnn_features_extractor_kwargs.update(dict(
            features_dim=self.model_features_dim,
        ))
        self.model_cnn_extractor = \
            self.model_cnn_features_extractor_class(
                self.observation_space,
                **self.model_cnn_features_extractor_kwargs
            )

        # Build RNNs
        self.model_rnns = []
        if self.use_model_rnn:
            for l in range(self.gru_layers):
                name = f'model_rnn_layer_{l}'
                setattr(self, name, self.model_gru_cell(**self.model_rnn_kwargs))
                self.model_rnns.append(getattr(self, name))


    def _init_modules(self) -> None:
        assert hasattr(self, 'ensemble_nn'), "Be sure to define the MLP ensemble first"

        module_names = {
            self.model_cnn_extractor: 'model_cnn_extractor'
        }
        for i, member in enumerate(self.ensemble_nn):
            module_names.update({member: f'model_mlp_{i}'})
        if self.use_model_rnn:
            for l in range(self.gru_layers):
                name = f'model_rnn_layer_{l}'
                module = getattr(self, name)
                module_names.update({module: name})
        for module, name in module_names.items():
            init_module_with_name(name, module)


    def _init_optimizers(self) -> None:
        param_dicts = dict(self.named_parameters(recurse=True)).items()
        self.model_params = [
            param for param in param_dicts
        ]
        self.model_optimizer = self.optimizer_class(self.model_params, lr=self.model_learning_rate, **self.optimizer_kwargs)


    def _get_rnn_embeddings(self, hiddens: Optional[Tensor], inputs: Tensor, modules: List[nn.Module]):
        outputs = []
        for i, module in enumerate(modules):
            hidden_i = th.squeeze(hiddens[:, i, :])
            output_i = module(inputs, hidden_i)
            inputs = output_i
            outputs.append(output_i)
        outputs = th.stack(outputs, dim=1)
        return outputs


    def _get_cnn_embeddings(self, obs, module=None):
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        if module is None:
            return self.model_cnn_extractor(obs)
        return module(obs)
    
    def save(self, path: str, additional_data: Optional[Dict[str, Any]] = None, debug: bool = False) -> None:
        """
        Save the reward model configuration, parameters, and optimizer state.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": self.model_optimizer.state_dict(),
            "config": {
                "observation_space": self.observation_space,  # may need to serialize manually
                "action_space": self.action_space,  # may need to serialize manually
                "activation_fn": self.activation_fn.__name__,
                "normalize_images": self.normalize_images,
                "optimizer_class": self.optimizer_class.__name__,
                "optimizer_kwargs": self.optimizer_kwargs,
                "max_grad_norm": self.max_grad_norm,
                "model_learning_rate": self.model_learning_rate,
                "model_cnn_features_extractor_class": self.model_cnn_features_extractor_class.__name__,
                "model_cnn_features_extractor_kwargs": self.model_cnn_features_extractor_kwargs,
                "model_feature_dim": self.model_features_dim,
                "model_latent_dim": self.model_latents_dim,
                "model_mlp_norm": self.model_mlp_norm,
                "model_cnn_norm": self.model_cnn_norm,
                "model_gru_norm": self.model_gru_norm,
                "use_model_rnn": self.use_model_rnn,
                "model_mlp_layers": self.model_mlp_layers,
                "gru_layers": self.gru_layers,
            },
        }
        if additional_data is not None:
            checkpoint["config"].update(additional_data)
        # torch spaces are not serializable, so handle them carefully
        # If your action_space is from gymnasium, save its type and shape
        if hasattr(self.action_space, "n"):
            checkpoint["config"]["action_space"] = {
                "type": "Discrete",
                "n": self.action_space.n,
            }
        elif hasattr(self.action_space, "shape"):
            checkpoint["config"]["action_space"] = {
                "type": "Box",
                "shape": self.action_space.shape,
            }
        th.save(checkpoint, path)
        if debug:
            print(f"Reward model saved to {path}")

    @classmethod
    def load(cls, path: str, additional_data: Optional[Dict[str, Any]] = None, device=None, debug: bool = False) -> None:
        """
        Load a reward model and return an initialized instance with weights and optimizer restored.
        """
        checkpoint = th.load(path, map_location=device or "cpu")
        cfg = checkpoint["config"]

        # Rebuild the observation space
        from gymnasium import spaces
        if hasattr(cfg["observation_space"], "shape"):
            observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=cfg["observation_space"].shape,
                dtype=th.float32,
            )
        else:
            observation_space = cfg["observation_space"]
        
        # Rebuild the action space
        if isinstance(cfg["action_space"], dict):
            space_cfg = cfg["action_space"]
            if space_cfg["type"] == "Discrete":
                action_space = spaces.Discrete(space_cfg["n"])
            elif space_cfg["type"] == "Box":
                action_space = spaces.Box(low=-1, high=1, shape=space_cfg["shape"])
            else:
                raise ValueError(f"Unknown action space type {space_cfg['type']}")
        else:
            action_space = cfg["action_space"]

        if additional_data is not None:
            kwargs = additional_data
        else:
            kwargs = {}
        # Rebuild the model
        model = cls(
            observation_space=observation_space,
            action_space=action_space,
            activation_fn=getattr(nn, cfg["activation_fn"]),
            normalize_images=cfg["normalize_images"],
            optimizer_class=getattr(th.optim, cfg["optimizer_class"]),
            optimizer_kwargs=cfg["optimizer_kwargs"],
            max_grad_norm=cfg["max_grad_norm"],
            model_learning_rate=cfg["model_learning_rate"],
            model_cnn_features_extractor_class=getattr(th.nn, cfg["model_cnn_features_extractor_class"]),
            model_cnn_features_extractor_kwargs=cfg["model_cnn_features_extractor_kwargs"],
            model_feature_dim=cfg["model_feature_dim"],
            model_latent_dim=cfg["model_latent_dim"],
            model_mlp_norm=cfg["model_mlp_norm"],
            model_cnn_norm=cfg["model_cnn_norm"],
            model_gru_norm=cfg["model_gru_norm"],
            use_model_rnn=cfg["use_model_rnn"],
            model_mlp_layers=cfg["model_mlp_layers"],
            **kwargs,
        )
        # Load weights and optimizer
        model.load_state_dict(checkpoint["state_dict"])
        model.model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.to(device or "cpu")
        if debug:
            print(f"Reward model loaded from {path}")
        return model