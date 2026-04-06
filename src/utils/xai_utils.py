import numpy as np
import torch as th
import torch.nn as nn
import cv2
from stable_baselines3.common.utils import obs_as_tensor
from src.utils.enum_types import XplainMethod
from captum.attr import (
    Saliency, InputXGradient, GuidedBackprop, Deconvolution,
    IntegratedGradients, DeepLift, GradientShap, DeepLiftShap,
    LayerGradCam, NoiseTunnel, LayerAttribution, LRP
)

class ValueNetworkWrapper(nn.Module):
    def __init__(self, sb3_model, explain_model='value'):
        super(ValueNetworkWrapper, self).__init__()
        # Store the SB3 model as a submodule so Captum can find its layers
        self.sb3_model = sb3_model
        self.explain_model = explain_model
        
    def forward(self, obs_in, policy_mems_in, actions_in=None):
        # 1. Pass the observation through the feature extractor
        latent_pi, latent_vf, _ = self.sb3_model.policy.extract_features(obs_in, policy_mems_in)
        # 2. Predict the Value of the current state and action distribution (logits/probs)
        if self.explain_model == 'value':
            values = self.sb3_model.policy.value_net(latent_vf)
            return values.squeeze(-1)
        elif self.explain_model == 'action':
            distribution = self.sb3_model.policy._get_action_dist_from_latent(latent_pi)
            log_probs = distribution.log_prob(actions_in)
            return log_probs
        else:
            raise ValueError(f"Unsupported explanation model: {self.explain_model}")
    
def fetch_captum_explainer(enum_method, wrapper, model=None, kwargs=None):
    kwargs = kwargs or {}
    if enum_method == XplainMethod.Saliency:
        return Saliency(wrapper)
    elif enum_method == XplainMethod.SmoothSaliency:
        return NoiseTunnel(Saliency(wrapper))
    elif enum_method == XplainMethod.InputXGradient:
        return InputXGradient(wrapper)
    elif enum_method == XplainMethod.GuidedBackprop:
        return GuidedBackprop(wrapper)
    elif enum_method == XplainMethod.Deconvolution:
        return Deconvolution(wrapper)
    elif enum_method == XplainMethod.LRP:
        return LRP(wrapper)
    elif enum_method == XplainMethod.IntegratedGradients:
        return IntegratedGradients(wrapper)
    elif enum_method == XplainMethod.SmoothIntegratedGradients:
        return NoiseTunnel(IntegratedGradients(wrapper))
    elif enum_method == XplainMethod.DeepLift:
        return DeepLift(wrapper)
    elif enum_method == XplainMethod.GradientSHAP:
        return GradientShap(wrapper)
    elif enum_method == XplainMethod.DeepLiftShap:
        return DeepLiftShap(wrapper)
    elif enum_method == XplainMethod.GradCAM:
        target_layer_num = kwargs.get('target_layer', -2)
        target_layer = model.policy.features_extractor.cnn[target_layer_num]
        return LayerGradCam(wrapper, target_layer)
    else:
        raise ValueError(f"Unsupported explanation method: {enum_method}")

def fetch_attribution(xai_method: XplainMethod, xplainer, obs_tensor, policy_mems, actions_tensor, target=None):
    baseline = th.zeros_like(obs_tensor)
    baseline_dist = th.zeros(5, *obs_tensor.shape[1:], device=obs_tensor.device)
    if xai_method in [XplainMethod.Saliency, XplainMethod.InputXGradient, XplainMethod.GuidedBackprop, XplainMethod.Deconvolution, XplainMethod.LRP]:
        return xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), target=target)
    elif xai_method in [XplainMethod.SmoothSaliency]:
        return xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), target=target, nt_samples=5, nt_type='smoothgrad')
    elif xai_method == XplainMethod.IntegratedGradients:
        return xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), baselines=baseline, target=target, n_steps=15)
    elif xai_method == XplainMethod.SmoothIntegratedGradients:
        return xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), baselines=baseline, target=target, n_steps=15, nt_samples=5, nt_type='smoothgrad')
    elif xai_method in [XplainMethod.DeepLift]:
        return xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), baselines=baseline, target=target)
    elif xai_method in [XplainMethod.GradientSHAP, XplainMethod.DeepLiftShap]:
        return xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), baselines=baseline_dist, target=target)
    elif xai_method == XplainMethod.GradCAM:
        layer_grads = xplainer.attribute(obs_tensor, additional_forward_args=(policy_mems, actions_tensor), relu_attributions=True, target=target)
        # Upsample back to (Batch, C, H, W)
        return LayerAttribution.interpolate(layer_grads, obs_tensor.shape[2:])
    else:
        raise ValueError(f"Unsupported explanation method: {xai_method}")

def project_attribution_to_global_fast(attribution_map, agent_pos, agent_dir, env_width, env_height, tile_size=8):
    """
    Fast, vectorized projection of a local attribution map of arbitrary size onto the global environment coordinate space.
    It rotates the entire array and doing block-wise assignment.
    Dynamically handles non-square environments and custom agent view sizes.
    It is used in Minigrid.
    """
    view_h_pixels, view_w_pixels = attribution_map.shape
    view_h = view_h_pixels // tile_size
    view_w = view_w_pixels // tile_size
    ax, ay = agent_pos

    # 1. Rotate the entire attribution map at once and determine the new top-left tile anchor
    if agent_dir == 3: # Facing Up (-y)
        rot_k = 0
        start_x = ax - (view_w // 2)
        start_y = ay - (view_h - 1)
    elif agent_dir == 0: # Facing Right (+x)
        rot_k = -1
        start_x = ax
        start_y = ay - (view_w // 2)
    elif agent_dir == 1: # Facing Down (+y)
        rot_k = -2
        start_x = ax - (view_w // 2)
        start_y = ay
    elif agent_dir == 2: # Facing Left (-x)
        rot_k = 1
        start_x = ax - (view_h - 1)
        start_y = ay - (view_w // 2)
    else:
        raise ValueError(f"Invalid agent_dir: {agent_dir}. Must be in [0, 3].")
        
    rot_patch = np.rot90(attribution_map, k=rot_k)
    rot_h_pixels, rot_w_pixels = rot_patch.shape
    
    # 2. Convert top-left tile coordinates to pixel coordinates
    px_start_x = start_x * tile_size
    px_start_y = start_y * tile_size
    px_end_x = px_start_x + rot_w_pixels
    px_end_y = px_start_y + rot_h_pixels

    # 3. Calculate bounding box intersections (clipping)
    # Determine where the patch lands on the global canvas
    g_x1 = max(0, px_start_x)
    g_y1 = max(0, px_start_y)
    g_x2 = min(env_width * tile_size, px_end_x)
    g_y2 = min(env_height * tile_size, px_end_y)

    # Determine which part of the local patch to slice (crop if out of global bounds)
    p_x1 = max(0, -px_start_x)
    p_y1 = max(0, -px_start_y)
    p_x2 = rot_w_pixels - max(0, px_end_x - env_width * tile_size)
    p_y2 = rot_h_pixels - max(0, px_end_y - env_height * tile_size)

    # 4. Initialize canvas and paste the block in one shot
    global_saliency = np.zeros((env_height * tile_size, env_width * tile_size), dtype=np.float32)
    
    # Only paste if the view is actually within the environment bounds
    if g_x1 < g_x2 and g_y1 < g_y2:
        global_saliency[g_y1:g_y2, g_x1:g_x2] = rot_patch[p_y1:p_y2, p_x1:p_x2]

    return global_saliency

def generate_attribution_map(observations, policy_mems, actions, env, device, xplainer, frames, xai_method):
    obs_tensor = obs_as_tensor(observations, device=device).float().clone().detach()
    obs_tensor.requires_grad_()
    
    # Convert the rollout actions to a tensor so we can evaluate them
    actions_tensor = th.as_tensor(actions, device=device)

    # Because our wrapper returns a 1D tensor of log_probs (one specific scalar per batch item),
    # Captum knows exactly what to differentiate. We no longer need to pass 'target'.
    grads = fetch_attribution(xai_method, xplainer, obs_tensor, policy_mems, actions_tensor)

    # Convert to numpy (Batch, Channels, Height, Width)
    # grads_np = np.abs(grads.cpu().detach().numpy())
    grads_np = grads.cpu().detach().numpy()
    
    # Collapse channels by taking the max gradient per pixel -> (Batch, Height, Width)
    # saliency_batch = np.max(grads_np, axis=1)
    saliency_batch = np.sum(grads_np, axis=1)
    
    # 4. Fetch vectorized environment attributes
    # Standard way to get variables from a stable-baselines3 or gym3 VecEnv
    try:
        agent_positions = env.get_attr('agent_pos')
        agent_dirs = env.get_attr('agent_dir')
        grid_widths = env.get_attr('width')
        grid_heights = env.get_attr('height')
        agent_view_sizes = env.get_attr('agent_view_size')
    except AttributeError:
        # Fallback if you are using a custom VecEnv
        agent_positions = [e.unwrapped.agent_pos for e in env.envs]
        agent_dirs = [e.unwrapped.agent_dir for e in env.envs]
        grid_widths = [e.unwrapped.width for e in env.envs]
        grid_heights = [e.unwrapped.height for e in env.envs]
        agent_view_sizes = [e.unwrapped.agent_view_size for e in env.envs]

    blended_frames = []

    # 5. Process each environment in the batch
    for i in range(len(observations)):
        local_saliency = saliency_batch[i]
        frame = frames[i]
        render_h, render_w, _ = frame.shape
        
        # The height of the local attribution array divided by the agent's view distance
        obs_pixels_h = local_saliency.shape[0]
        view_tiles_h = agent_view_sizes[i]

        # Use integer division. (Fallback to 1 if using non-image, purely symbolic 7x7 grids)
        tile_size = max(1, obs_pixels_h // view_tiles_h)

        # Project local 7x7 (or VxV) attribution to the global coordinate space
        global_saliency = project_attribution_to_global_fast(
            attribution_map=local_saliency, 
            agent_pos=agent_positions[i], 
            agent_dir=agent_dirs[i], 
            env_width=grid_widths[i], 
            env_height=grid_heights[i], 
            tile_size=tile_size # Fallback to 8 if not in config
        )
        
        # # POSITIVE NORMALIZATION to [0, 255]
        # if global_saliency.max() > 0:
        #     global_saliency = (global_saliency - global_saliency.min()) / (global_saliency.max() - global_saliency.min() + 1e-8)
        # global_saliency = np.uint8(global_saliency * 255)
        
        # SYMMETRIC NORMALIZATION (Keeps 0 at 0)
        # Find the absolute maximum so we can scale everything between -1.0 and 1.0
        max_abs_val = np.max(np.abs(global_saliency))
        if max_abs_val > 1e-8:
            global_saliency = global_saliency / max_abs_val
        else:
            global_saliency = global_saliency

        # Scale up to the high-res render dimensions
        saliency_resized = cv2.resize(global_saliency, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # # Apply colormap
        # heatmap = cv2.applyColorMap(saliency_resized, cv2.COLORMAP_JET)
        
        # # Mask and Blend
        # mask = saliency_resized > 0

        # # Original video frames underneath.
        # blended = frame_bgr.copy()
        # # Blend only where the agent is looking (80% original, 20% heatmap)
        # blended[mask] = cv2.addWeighted(frame_bgr, 0.8, heatmap, 0.2, 0)[mask]
        
        # Alpha is the absolute magnitude (0.0 to 1.0). Scale by 0.6 for max opacity.
        alpha = np.abs(saliency_resized)[..., np.newaxis] * 0.5
        
        # Create solid BGR canvases
        red_canvas = np.zeros_like(frame_bgr)
        red_canvas[:] = [0, 0, 255]   # Pure Red for positive
        
        blue_canvas = np.zeros_like(frame_bgr)
        blue_canvas[:] = [255, 0, 0]  # Pure Blue for negative
        
        # Create boolean masks for positive and negative regions
        pos_mask = (saliency_resized > 0)[..., np.newaxis]
        neg_mask = (saliency_resized < 0)[..., np.newaxis]
        
        # Blend mathematically
        blended = frame_bgr * (1.0 - alpha)                           # Darken original frame where alpha is high
        blended += np.where(pos_mask, red_canvas * alpha, 0)          # Add red to positive areas
        blended += np.where(neg_mask, blue_canvas * alpha, 0)         # Add blue to negative areas

        # Convert back to RGB for video writers like imageio or wandb
        blended_rgb = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB)
        blended_frames.append(blended_rgb)
    return blended_frames