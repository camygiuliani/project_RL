from html import parser
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import cv2
from datetime import datetime


from utils import load_config
from wrappers import make_env
from ppo import ActorCriticCNN
from ddqn import DDQN_Agent,DDQNCNN
from sac import SACDiscrete_Agent


def _get_fill_value(obs_uint8, mode="mean"):
    # here we decide how to fill the occluded patch
    
    """
    obs_uint8: (H,W,4) uint8
    mode:
      - "mean": mean for channel (default)
      - "zero": black (zeros)
    returns: fill value (1,1,4) uint8
    """
    if mode == "zero":
        return 0
    # mean for channel (shape (1,1,4)) then broadcast
    fill = obs_uint8.mean(axis=(0, 1), keepdims=True)
    return fill.astype(obs_uint8.dtype)


@torch.no_grad()  #function is more efficient without gradient tracking
def q_values_batch(agent, obs_uint8_batch):
    #compute Q values for a batch of observations
    """
    obs_uint8_batch: (N,H,W,4) uint8
    returns: (N,A) numpy float
    """
    x = torch.from_numpy(obs_uint8_batch).to(agent.device)
    x = x.permute(0, 3, 1, 2).float() / 255.0  # NCHW
    q = agent.q(x)  # (N,A)
    return q.detach().cpu().numpy()

#heart of the SARFA-like method
#in here we  generate the heatmap
def sarfa_heatmap(agent, obs_input, patch=8, stride=4, 
                  fill_mode="mean", batch_size=32, 
                  use_advantage=True, clamp_positive=True,
                  use_action_flip=True, flip_weight=2.0):
    
    obs = obs_input
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()

    # If obs is CHW (4,84,84), convert to HWC (84,84,4)
    if obs.ndim == 3:
        if obs.shape[0] in (1, 4) and obs.shape[-1] not in (1, 4):
            obs = np.transpose(obs, (1, 2, 0))  # (H,W,C)
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
        
    # input preparation
    if obs.dtype != np.uint8:
        if obs.max() <= 1.0: 
            obs = (obs * 255).astype(np.uint8)
        else: 
            obs = obs.astype(np.uint8)
        
    # Gym returns (H, W, C), PyTorch wants (C, H, W)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    obs_tensor = obs_tensor.permute(2, 0, 1) 
    obs_tensor = obs_tensor.unsqueeze(0) / 255.0
    
    # Baseline
    with torch.no_grad():
        q_values = agent.q(obs_tensor) 
      
        base_action = torch.argmax(q_values, dim=1).item()
        base_q = q_values[0, base_action].item()

    # grid settings
    H, W, C = obs.shape
    coords = [(x, y) for y in range(0, H-patch+1, stride) for x in range(0, W-patch+1, stride)]
    heatmap = np.zeros((H, W), dtype=np.float32)

    if fill_mode == "mean": fill_val = np.mean(obs)
    else: fill_val = 0

    # batch loop
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i+batch_size]
        batch_np = np.repeat(obs[None, ...], len(chunk), axis=0).copy()

        for idx, (x, y) in enumerate(chunk):
            batch_np[idx, y:y+patch, x:x+patch, :] = fill_val 

        batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=agent.device)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2) 
        batch_tensor = batch_tensor / 255.0

        with torch.no_grad():
            q_perturbed = agent.q(batch_tensor) 
            

        # score computation
        q_of_base_action = q_perturbed[:, base_action].cpu().numpy()
        delta = base_q - q_of_base_action

        if use_action_flip:
            new_actions = torch.argmax(q_perturbed, dim=1).cpu().numpy()
            flip_mask = (new_actions != base_action)
            delta[flip_mask] = np.maximum(delta[flip_mask], 0.1) * flip_weight

        for idx, (x, y) in enumerate(chunk):
            val = delta[idx]
            if clamp_positive and val < 0: val = 0
            heatmap[y:y+patch, x:x+patch] += val

    return heatmap, base_action



# =========================
# PPO version 
# =========================

@torch.no_grad()
def ppo_logprobs_batch(actor_critic_net, device, obs_uint8_batch, actions=None):
    """
    actor_critic_net: your ActorCriticCNN (from ppo_agent.py), already on device
    obs_uint8_batch: (N,H,W,4) uint8
    actions: optional (N,) int actions. If None, we use argmax of policy logits.

    Returns:
      logp: (N,) float  log pi(a|s) for chosen action
      chosen_actions: (N,) int
      best_actions: (N,) int  argmax_a pi(a|s)
    """
    x = torch.from_numpy(obs_uint8_batch).to(device)
    x = x.permute(0, 3, 1, 2).float() / 255.0  # NCHW

    out = actor_critic_net(x)
    if isinstance(out, (tuple, list)):
        logits = out[0]   # PPO: (logits, value)
    else:
        logits = out      # SAC: logits only

    dist = torch.distributions.Categorical(logits=logits)

    best_actions = torch.argmax(logits, dim=1)

    if actions is None:
        chosen_actions = best_actions
    else:
        chosen_actions = torch.as_tensor(actions, device=device, dtype=torch.long)

    logp = dist.log_prob(chosen_actions)  # (N,)
    return (
        logp.detach().cpu().numpy(),
        chosen_actions.detach().cpu().numpy(),
        best_actions.detach().cpu().numpy()
    )

def blur_heatmap(heat, k=7):
    # box blur kxk
    pad = k // 2
    h = np.pad(heat, ((pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(heat, dtype=np.float32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = h[y:y+k, x:x+k].mean()
    return out

def sarfa_heatmap_policy_logp(
    policy_net, device, obs_input,
    patch=8, stride=4, fill_mode="mean", batch_size=32,
    clamp_positive=True, use_action_flip=True, flip_weight=2.0,
    occlude_only_last_frame=True
):
    obs = obs_input
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    # normalize layout to HWC
    if obs.ndim == 3:
        # CHW -> HWC
        if obs.shape[0] in (1, 4) and obs.shape[-1] not in (1, 4):
            obs = np.transpose(obs, (1, 2, 0))
    elif obs.ndim == 2:
        # HW -> HWC (grayscale)
        obs = obs[..., None]
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
    if obs.dtype != np.uint8:
        obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)

    H, W, C = obs.shape
    assert C == 4, "Expected (84,84,4) stacked frames"

    # base action + base logp
    base_logp, _, base_best = ppo_logprobs_batch(
        policy_net, device, obs[None, ...], actions=None
    )
    base_action = int(base_best[0])  # explain argmax action
    base_logp = float(
        ppo_logprobs_batch(
            policy_net, device, obs[None, ...], actions=np.array([base_action], dtype=np.int64)
        )[0][0]
    )

    coords = [(x, y) for y in range(0, H - patch + 1, stride)
                    for x in range(0, W - patch + 1, stride)]
    heatmap = np.zeros((H, W), dtype=np.float32)

    fill = _get_fill_value(obs, mode=fill_mode)  # (1,1,4) or 0

    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        n = len(chunk)
        batch_np = np.repeat(obs[None, ...], n, axis=0).copy()

        for j, (x, y) in enumerate(chunk):
            if occlude_only_last_frame:
                # occlude only last frame to avoid destroying motion info
                if isinstance(fill, np.ndarray):
                    batch_np[j, y:y+patch, x:x+patch, -1] = fill[0, 0, -1]
                else:
                    batch_np[j, y:y+patch, x:x+patch, -1] = fill
            else:
                batch_np[j, y:y+patch, x:x+patch, :] = fill

        logp_masked, _, best_masked = ppo_logprobs_batch(
            policy_net, device, batch_np,
            actions=np.full((n,), base_action, dtype=np.int64)
        )

        delta = base_logp - logp_masked  # higher = more important

        if use_action_flip:
            flips = (best_masked != base_action).astype(np.float32)
            delta = delta * (1.0 + flip_weight * flips)

        if clamp_positive:
            delta = np.maximum(delta, 0.0)

        for (x, y), s in zip(chunk, delta):
            heatmap[y:y+patch, x:x+patch] += float(s)

    return heatmap, base_action


#original patch=8 stride=8

def sarfa_heatmap_ppo(model, device, obs_input, patch=8, stride=4, 
                              fill_mode="mean", batch_size=32, 
                              clamp_positive=True, use_action_flip=True, flip_weight=2.0):
    
    obs = obs_input
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()

    if obs.ndim == 3:
        # CHW case (common with FrameStack): (4,84,84)
        if obs.shape[0] in (1, 4) and obs.shape[-1] not in (1, 4):
            obs = np.transpose(obs, (1, 2, 0))  # -> (H,W,C)
        # else assume it's already HWC
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    # initial input processing
    if obs.dtype != np.uint8:
        if obs.max() <= 1.0: 
            obs = (obs * 255).astype(np.uint8)
        else: 
            obs = obs.astype(np.uint8)

    #  (H,W,C -> C,H,W)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.permute(2, 0, 1) # (4, 84, 84)
    obs_tensor = obs_tensor.unsqueeze(0) / 255.0 # (1, 4, 84, 84)
    

    with torch.no_grad():
        # TODO controllare questo pezzo
        # --- FIX ATTRIBUTE ERROR ---
        # Il modello restituisce (logits, values) oppure (distribution, values).
        # Controlliamo cosa abbiamo ricevuto.
        output, _ = model(obs_tensor)
        
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output # this is already the logits

        base_probs = torch.softmax(logits, dim=-1)
        base_action = torch.argmax(base_probs, dim=-1).item()
        base_prob_val = base_probs[0, base_action].item()
        # ---------------------------

    # grid coordinates
    H, W, C = obs.shape
    y_range = range(0, H - patch + 1, stride)
    x_range = range(0, W - patch + 1, stride)
    coords = [(x, y) for y in y_range for x in x_range]

    heatmap = np.zeros((H, W), dtype=np.float32)

    if fill_mode == "mean":
        fill_val = np.mean(obs)
    else:
        fill_val = 0 

    # batch loop
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i+batch_size]
        n_chunk = len(chunk)

        # creating numpy batch (Batch, H, W, C)
        
        batch_np = np.repeat(obs[None, ...], n_chunk, axis=0).copy()

        # applying masks (deep masking on all channels)
        for idx, (x, y) in enumerate(chunk):
            batch_np[idx, y:y+patch, x:x+patch, :] = fill_val 

        
        batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2) # (Batch, 4, 84, 84)
        batch_tensor = batch_tensor / 255.0
       

        with torch.no_grad():
            # --- FIX ATTRIBUTE ERROR (Anche qui nel loop) ---
            output_p, _ = model(batch_tensor)
            
            if hasattr(output_p, 'logits'):
                logits_p = output_p.logits
            else:
                logits_p = output_p
            
            probs_p = torch.softmax(logits_p, dim=-1)
            # -----------------------------------------------
        
        # computing sarfa scores
        new_probs_of_base_action = probs_p[:, base_action]
        delta = base_prob_val - new_probs_of_base_action.cpu().numpy()

        if use_action_flip:
            new_actions = torch.argmax(probs_p, dim=-1).cpu().numpy()
            changed_mask = (new_actions != base_action)
            delta[changed_mask] *= flip_weight

        for idx, (x, y) in enumerate(chunk):
            score = delta[idx]
            if clamp_positive and score < 0: score = 0
            heatmap[y:y+patch, x:x+patch] += score

    return heatmap, base_action

@torch.no_grad()
def sac_policy_logits(agent, obs_batch):
    # obs_batch può essere np.ndarray o torch.Tensor
    if isinstance(obs_batch, torch.Tensor):
        x = obs_batch.to(agent.device)
    else:
        x = torch.from_numpy(obs_batch).to(agent.device)

    # Convert to float
    if x.dtype != torch.float32:
        x = x.float()

    # --- Detect layout and fix ---
    # Case A: already NCHW (N,4,84,84)
    if x.ndim == 4 and x.shape[1] == 4:
        pass  # ok

    # Case B: NHWC (N,84,84,4)
    elif x.ndim == 4 and x.shape[-1] == 4:
        x = x.permute(0, 3, 1, 2)

    # Case C: weird (N,84,4,84) -> fix to (N,4,84,84)
    elif x.ndim == 4 and x.shape[2] == 4:
        x = x.permute(0, 2, 1, 3)

    else:
        raise ValueError(f"Unexpected obs shape for SAC actor: {tuple(x.shape)}")

    # normalize
    x = x / 255.0

    logits = agent.actor(x)  # (N, A)
    return logits

def _save_side_by_side(image_paths, out_path):
    imgs = []
    for p in image_paths:
        im = cv2.imread(p)  # BGR
        if im is None:
            raise RuntimeError(f"Could not read image: {p}")
        imgs.append(im)

    # resize all to same height
    h = min(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        w = int(im.shape[1] * (h / im.shape[0]))
        resized.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))

    combo = cv2.hconcat(resized)
    cv2.imwrite(out_path, combo)

def _extract_ale_rgb(env):
    # Try to get ALE screen RGB through wrappers
    ale = None
    current_env = env
    while hasattr(current_env, "env"):
        if hasattr(current_env, "ale"):
            ale = current_env.ale
            break
        current_env = current_env.env
    if ale is None and hasattr(current_env, "ale"):
        ale = current_env.ale

    if ale is not None:
        try:
            return ale.getScreenRGB()
        except Exception:
            pass

    # fallback
    try:
        return env.render()
    except Exception:
        return None


def _collect_snapshots(env, seed, snap_steps):
    """
    Returns list of tuples: (obs_uint8, rgb_bg) for each step in snap_steps.
    Uses random actions to advance the env.
    """
    obs, _ = env.reset(seed=seed)
    snaps = []
    max_step = max(snap_steps)
    target_set = set(snap_steps)

    for t in range(1, max_step + 1):
        a = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()

        if t in target_set:
            # ensure uint8
            obs_u = obs
            if obs_u.max() <= 1.0:
                obs_u = (obs_u * 255).astype(np.uint8)
            else:
                obs_u = obs_u.astype(np.uint8)

            rgb = _extract_ale_rgb(env)
            snaps.append((t, obs_u, rgb))

    return snaps


def _save_grid_3x3(grid_paths, snap_order, algo_order, out_path):
    """
    grid_paths: dict like grid_paths[snap_name][algo] = png_path
    snap_order: ["early","mid","late"]
    algo_order: ["ddqn","ppo","sac"]
    """
    fig, axes = plt.subplots(len(snap_order), len(algo_order), figsize=(15, 15))

    for i, snap in enumerate(snap_order):
        for j, algo in enumerate(algo_order):
            ax = axes[i, j] if len(snap_order) > 1 else axes[j]
            p = grid_paths.get(snap, {}).get(algo, None)
            if p is None or not os.path.exists(p):
                ax.axis("off")
                ax.set_title("MISSING")
                continue

            im_bgr = cv2.imread(p)               # BGR
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(im_rgb)
            ax.axis("off")

            # Column titles (top row)
            if i == 0:
                ax.set_title(algo.upper(), fontsize=16, pad=12)

            # Row labels (left column)
            if j == 0:
                ax.set_ylabel(snap.upper(), fontsize=16, rotation=90, labelpad=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close()

''' 
def record_sarfa_video(env, agent, algo_name, args, cfg, device, out_path, max_steps=500):
    """
    Runs an episode and records a video with the SARFA heatmap overlaid.
    """
    print(f"--- Starting Video Recording for {algo_name} ---")
    
    # 1. Setup Video Writer
    obs, _ = env.reset(seed=args.seed)
    
    # Get initial RGB frame to determine video dimensions
    rgb_frame = _extract_ale_rgb(env)
    if rgb_frame is None:
        print("Error: Could not extract RGB frame for video.")
        return

    height, width, layers = rgb_frame.shape
    # Upscale factor for better visibility
    scale = 4
    video_size = (width * scale, height * scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, video_size)

    try:
        for step in range(max_steps):
            if step % 10 == 0:
                print(f"Rendering frame {step}/{max_steps}...")

            # 2. Compute Heatmap & Action based on Algo
            # We use the existing sarfa functions which return (heatmap, best_action)
            
            # Prepare observation (uint8 check)
            obs_in = obs
            if obs_in.max() <= 1.0:
                obs_in = (obs_in * 255).astype(np.uint8)
            else:
                obs_in = obs_in.astype(np.uint8)

            heatmap = None
            action = 0

            if algo_name == "ddqn":
                heatmap, action = sarfa_heatmap(
                    agent, obs_in,
                    patch=args.patch, stride=args.stride,
                    fill_mode="mean",
                    batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )
            elif algo_name == "ppo":
                heatmap, action = sarfa_heatmap_ppo(
                    agent, device, obs_in,
                    patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )
            elif algo_name == "sac":
                heatmap, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device, obs_in,
                    patch=args.patch, stride=args.stride
                )

            # 3. Process Heatmap for Video (using OpenCV for speed)
            # Blur
            heatmap = blur_heatmap(heatmap, k=7)
            
            # Normalize (0 to 1)
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Thresholding (clean up low noise)
            thr = np.mean(heatmap)
            heatmap[heatmap < thr] = 0.0

            # Resize heatmap to match RGB frame
            heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to Color Map (0-255)
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # 4. Blend Images
            # Get current RGB background
            bg_rgb = _extract_ale_rgb(env)
            # Convert RGB (Gym) to BGR (OpenCV)
            bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

            # Create mask for transparency (where heatmap is strong, show it; else show game)
            # Simple weighted add:
            # We want the heatmap to be transparent. 
            # Note: heatmap_color is BGR.
            
            # Multiply heatmap intensity to make the transparent parts darker in the overlay
            mask = heatmap_resized[..., None] # (H,W,1)
            
            # 0.0 = pure game, 1.0 = pure heatmap. Let's cap opacity at 0.6
            opacity = 0.6
            blended = (bg_bgr * (1.0 - (mask * opacity)) + heatmap_color * (mask * opacity)).astype(np.uint8)

            # 5. Add Text (Action Name)
            # Assuming cfg has ACTION_NAMES, otherwise use ID
            act_name = str(action)
            if "ACTION_NAMES" in cfg and action < len(cfg["ACTION_NAMES"]):
                act_name = cfg["ACTION_NAMES"][action]
            
            cv2.putText(blended, f"Step: {step} | Action: {act_name}", (2, 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 6. Upscale for video file
            final_frame = cv2.resize(blended, video_size, interpolation=cv2.INTER_NEAREST)
            out.write(final_frame)

            # 7. Step Environment
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print("Episode finished early.")
                break
                
    except KeyboardInterrupt:
        print("Video recording interrupted by user.")
    finally:
        out.release()
        print(f"Saved video to: {out_path}")
'''
def record_sarfa_video(env, agent, algo_name, args, cfg, device, out_path, max_steps=1000):
    print(f"--- Starting Video Recording for {algo_name} ---")
    
    # 1. Setup Video Writer
    obs, _ = env.reset(seed=args.seed)
    
    # Get initial RGB frame to determine video dimensions
    rgb_frame = _extract_ale_rgb(env)
    if rgb_frame is None:
        print("Error: Could not extract RGB frame for video.")
        return

    height, width, layers = rgb_frame.shape
    # Upscale factor for better visibility
    scale = 4
    video_size = (width * scale, height * scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, video_size)

    try:
        for step in range(max_steps):
            if step % 10 == 0:
                print(f"Rendering frame {step}/{max_steps}...")

            # 2. Compute Heatmap & Action based on Algo
            # We use the existing sarfa functions which return (heatmap, best_action)
            
            # Prepare observation (uint8 check)
            obs_in = obs
            if obs_in.max() <= 1.0:
                obs_in = (obs_in * 255).astype(np.uint8)
            else:
                obs_in = obs_in.astype(np.uint8)

            heatmap = None
            action = 0

            if algo_name == "ddqn":
                heatmap, action = sarfa_heatmap(
                    agent, obs_in,
                    patch=args.patch, stride=args.stride,
                    fill_mode="mean",
                    batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )
            elif algo_name == "ppo":
                heatmap, action = sarfa_heatmap_ppo(
                    agent, device, obs_in,
                    patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )
            elif algo_name == "sac":
                heatmap, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device, obs_in,
                    patch=args.patch, stride=args.stride
                )

            # 3. Process Heatmap for Video (using OpenCV for speed)
            # Blur
            heatmap = blur_heatmap(heatmap, k=7)
            
            # Normalize (0 to 1)
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Thresholding (clean up low noise)
            thr = np.mean(heatmap)
            heatmap[heatmap < thr] = 0.0

            # Resize heatmap to match RGB frame
            heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to Color Map (0-255)
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # 4. Blend Images
            # Get current RGB background
            bg_rgb = _extract_ale_rgb(env)
            # Convert RGB (Gym) to BGR (OpenCV)
            bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

            # Create mask for transparency (where heatmap is strong, show it; else show game)
            # Simple weighted add:
            # We want the heatmap to be transparent. 
            # Note: heatmap_color is BGR.
            
            # Multiply heatmap intensity to make the transparent parts darker in the overlay
            mask = heatmap_resized[..., None] # (H,W,1)
            
            # 0.0 = pure game, 1.0 = pure heatmap. Let's cap opacity at 0.6
            opacity = 0.6
            blended = (bg_bgr * (1.0 - (mask * opacity)) + heatmap_color * (mask * opacity)).astype(np.uint8)

            # 5. Add Text (Action Name)
            # Assuming cfg has ACTION_NAMES, otherwise use ID
            act_name = str(action)
            if "ACTION_NAMES" in cfg and action < len(cfg["ACTION_NAMES"]):
                act_name = cfg["ACTION_NAMES"][action]
            
            cv2.putText(blended, f"Step: {step} | Action: {act_name}", (2, 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 6. Upscale for video file
            final_frame = cv2.resize(blended, video_size, interpolation=cv2.INTER_NEAREST)
            out.write(final_frame)

            # 7. Step Environment
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print("Episode finished early.")
                break
                
    except KeyboardInterrupt:
        print("Video recording interrupted by user.")
    finally:
        out.release()
        print(f"Saved video to: {out_path}")


def record_sarfa_video_2(env, agent, algo_name, args, cfg, device, out_path, max_steps=1000):
    print(f"--- Starting Agent-View Video for {algo_name} ---")
    
    obs, _ = env.reset(seed=args.seed)
    
    # --- HELPER: Converti l'Osservazione in Immagine Visibile ---
    def obs_to_img(observation):
        # L'osservazione può essere (H, W, C) o (C, H, W) e float o uint8
        img = observation.copy()
        
        # 1. Se è un tensore Torch, portalo a Numpy
        if hasattr(img, 'cpu'): img = img.cpu().numpy()
        
        # 2. Gestione Canali (Canali primi vs Canali ultimi)
        # Se la dimensione piccola è all'inizio (es: 4, 84, 84), spostala alla fine
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.moveaxis(img, 0, -1)
            
        # 3. Gestione Frame Stacking (es. 4 frame impilati)
        # Prendiamo il MAX sui canali o l'ultimo canale per vedere tutto
        if img.shape[-1] > 3: 
            # Spesso i canali sono frame temporali. Il max aiuta a vedere proiettili che lampeggiano.
            img = np.max(img, axis=-1)
            
        # 4. Normalizzazione a 0-255
        if img.max() <= 1.0 + 1e-5:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # 5. Se è diventato scala di grigi (H, W), fallo diventare RGB (H, W, 3)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        return img
    # -----------------------------------------------------------

    # Prepariamo dimensioni basandoci sull'osservazione
    first_img = obs_to_img(obs)
    h, w, _ = first_img.shape
    
    # Upscale spinto perché l'obs è piccola (di solito 84x84)
    scale = 5
    final_h, final_w = h * scale, w * scale
    video_size = (final_w * 2, final_h) # Side by Side
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, video_size)

    try:
        for step in range(max_steps):
            if step % 50 == 0:
                print(f"Rendering step {step}/{max_steps}...")

            # 1. Prepara l'immagine "Reale" (Ciò che vede l'agente)
            # Usiamo obs PRE-step per coerenza con l'azione decisa
            agent_view_img = obs_to_img(obs)
            
            # Resize per il video
            left_img = cv2.resize(agent_view_img, (final_w, final_h), interpolation=cv2.INTER_NEAREST)

            # 2. Calcola Heatmap
            obs_in = obs
            # Assicuriamoci che obs_in sia nel formato giusto per sarfa_heatmap
            if hasattr(obs_in, 'max') and obs_in.max() <= 1.0:
                 obs_in_sarfa = (obs_in * 255).astype(np.uint8)
            else:
                 obs_in_sarfa = obs_in.astype(np.uint8)

            heatmap = None
            action = 0

            # Logica Agenti
            if algo_name == "ddqn":
                heatmap, action = sarfa_heatmap(
                    agent, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode="mean", batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )
            elif algo_name == "ppo":
                heatmap, action = sarfa_heatmap_ppo(
                    agent, device, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )
            elif algo_name == "sac":
                heatmap, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device, obs_in_sarfa, patch=args.patch, stride=args.stride
                )

            # 3. Elabora Lato Destro (Heatmap Overlay)
            # Blur
            heatmap = blur_heatmap(heatmap, k=5)
            # Normalize
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            # Threshold
            heatmap[heatmap < np.mean(heatmap)] = 0.0

            # Resize Heatmap sulle dimensioni dell'obs originale (84x84)
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Sovrapposizione sull'immagine dell'agente (non sul render rgb_array)
            # agent_view_img è già BGR perché obs_to_img lo converte
            alpha = 0.6
            mask = (heatmap_resized > 0)[..., None]
            
            overlay = agent_view_img.copy()
            # Applica colore solo dove c'è heatmap
            overlay = (overlay * (1 - mask * alpha) + heatmap_color * (mask * alpha)).astype(np.uint8)
            
            right_img = cv2.resize(overlay, (final_w, final_h), interpolation=cv2.INTER_NEAREST)

            # 4. Unisci e Salva
            combined = np.hstack([left_img, right_img])

            # Info Testo
            act_name = str(action)
            if "ACTION_NAMES" in cfg and action < len(cfg["ACTION_NAMES"]):
                act_name = cfg["ACTION_NAMES"][action]
            
            cv2.putText(combined, f"A: {act_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "Agent Input (Deflickered)", (10, final_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

            out.write(combined)

            # 5. Step
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print("Episode finished early.")
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        out.release()
        print(f"Saved Agent-View video to: {out_path}")

def record_sarfa_video_3(env, agent, algo_name, args, cfg, device, out_path, max_steps=1000):
    print(f"--- Starting High-Fidelity Video for {algo_name} ---")
    
    obs, _ = env.reset(seed=args.seed)
    
    # --- HELPER: Renderizza l'Osservazione (84x84) in Alta Qualità (HD) ---
    def process_agent_view(observation, target_h, target_w):
        # 1. Converti in Numpy (H, W) o (H, W, C)
        img = observation.copy()
        if hasattr(img, 'cpu'): img = img.cpu().numpy()
        
        # Gestione canali (se (C, H, W) -> (H, W, C))
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = np.moveaxis(img, 0, -1)
            
        # Frame Stacking: Prendiamo il MAX sui canali per vedere tutti i proiettili (anche quelli passati)
        if img.ndim == 3:
            img = np.max(img, axis=-1)
            
        # Normalizza 0-255
        if img.max() <= 1.0 + 1e-5:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # L'obs originale è solitamente 84x84 (Quadrata)
        # Il target è rettangolare (es. 320x420). 
        # Se facciamo resize diretto, si deforma. Facciamo Resize + Letterbox (barre nere).
        
        # Rendiamo l'immagine nitida (Nearest Neighbor) scalandola
        # L'Atari originale è circa 160x210. L'obs è 84x84.
        # Scaliamo l'84x84 per riempire la larghezza del target
        scale = target_w / img.shape[1]
        new_h = int(img.shape[0] * scale)
        
        resized = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Creiamo canvas nero della dimensione target
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Centriamo l'immagine ridimensionata nel canvas
        y_offset = (target_h - new_h) // 2
        # Clip per sicurezza se new_h > target_h
        if y_offset < 0:
            resized = resized[-y_offset : -y_offset+target_h, :]
            y_offset = 0
            
        end_y = min(y_offset + resized.shape[0], target_h)
        canvas[y_offset:end_y, :] = resized[:end_y-y_offset, :]
        
        # Convertiamo in BGR per video e applichiamo una tinta (Verde Matrix) per stile
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Tinta verde leggera per far capire che è la visione del computer
        canvas_bgr[..., 0] = 0 # Azzera blu
        canvas_bgr[..., 2] = 0 # Azzera rosso (lascia solo verde nel canale 1)
        # O lasciamo in scala di grigi pulita:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        return canvas_bgr, (0, y_offset, target_w, resized.shape[0]) # Ritorniamo anche l'area valida
    # --------------------------------------------------------------------------

    # Setup dimensioni Video
    # Sinistra: Gioco Reale (da render). Destra: Agente (da obs).
    # Usiamo dimensioni fisse standard HD verticale per chiarezza
    W_PANEL = 320
    H_PANEL = 420
    
    video_size = (W_PANEL * 2, H_PANEL)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, video_size)

    # Buffer per render RGB (Sinistra)
    try:
        dummy_render = env.render()
        if isinstance(dummy_render, list): dummy_render = dummy_render[0]
    except:
        dummy_render = np.zeros((210, 160, 3), dtype=np.uint8)
        
    render_buffer = [dummy_render] * 3 # Buffer di 3 frame per deflicker lato sinistro

    try:
        for step in range(max_steps):
            if step % 50 == 0:
                print(f"Rendering frame {step}/{max_steps}...")

            # === LATO SINISTRO: GIOCO REALE ===
            try:
                raw_render = env.render()
                if isinstance(raw_render, list): raw_render = raw_render[0]
            except:
                raw_render = np.zeros((210, 160, 3), dtype=np.uint8)
            
            # Aggiorna buffer e prendi il MAX (Deflicker visivo per umani)
            render_buffer.append(raw_render)
            render_buffer.pop(0)
            clean_render = np.max(np.stack(render_buffer), axis=0)
            
            clean_render_bgr = cv2.cvtColor(clean_render, cv2.COLOR_RGB2BGR)
            left_img = cv2.resize(clean_render_bgr, (W_PANEL, H_PANEL), interpolation=cv2.INTER_NEAREST)

            # === LATO DESTRO: VISIONE AGENTE + HEATMAP ===
            # 1. Processa Obs per renderlo bello
            agent_img, valid_area = process_agent_view(obs, H_PANEL, W_PANEL)
            
            # 2. Calcola Heatmap SARFA
            obs_in = obs
            if hasattr(obs_in, 'max') and obs_in.max() <= 1.0:
                 obs_in_sarfa = (obs_in * 255).astype(np.uint8)
            else:
                 obs_in_sarfa = obs_in.astype(np.uint8)

            heatmap = None
            action = 0
            
            # (Logica Agenti standard...)
            if algo_name == "ddqn":
                heatmap, action = sarfa_heatmap(
                    agent, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode="mean", batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )
            elif algo_name == "ppo":
                heatmap, action = sarfa_heatmap_ppo(
                    agent, device, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )
            elif algo_name == "sac":
                heatmap, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device, obs_in_sarfa, patch=args.patch, stride=args.stride
                )

            # 3. Applica Heatmap SOLO sull'area valida dell'immagine agente
            # Blur
            heatmap = blur_heatmap(heatmap, k=5)
            # Normalize
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            heatmap[heatmap < np.mean(heatmap)] = 0.0

            # L'area valida dove c'è il gioco dentro agent_img
            x_off, y_off, w_valid, h_valid = valid_area
            
            # Resiziamo la heatmap per coincidere con l'area di gioco ridimensionata, non tutto il pannello
            heatmap_resized = cv2.resize(heatmap, (w_valid, h_valid), interpolation=cv2.INTER_NEAREST)
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Estraiamo la parte di immagine agente dove sovrapporre
            roi = agent_img[y_off:y_off+h_valid, x_off:x_off+w_valid]
            
            # Blend
            alpha = 0.6
            mask = (heatmap_resized > 0)[..., None]
            roi_blended = (roi * (1 - mask * alpha) + heatmap_color * (mask * alpha)).astype(np.uint8)
            
            # Rimettiamo la ROI nel pannello destro
            agent_img[y_off:y_off+h_valid, x_off:x_off+w_valid] = roi_blended
            right_img = agent_img

            # === UNIONE ===
            combined = np.hstack([left_img, right_img])

            # Testo
            act_name = str(action)
            if "ACTION_NAMES" in cfg and action < len(cfg["ACTION_NAMES"]):
                act_name = cfg["ACTION_NAMES"][action]
            
            # Barra separatrice bianca
            cv2.line(combined, (W_PANEL, 0), (W_PANEL, H_PANEL), (255, 255, 255), 2)
            
            # Info
            cv2.putText(combined, "Game (Human View)", (10, H_PANEL - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(combined, "Agent Logic (Bullets Visible)", (W_PANEL + 10, H_PANEL - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(combined, f"Action: {act_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            out.write(combined)

            # Step
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print("Episode finished early.")
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        out.release()
        print(f"Saved High-Fidelity video to: {out_path}")
        
def record_sarfa_video_4(env, agent, algo_name, args, cfg, device, out_path, max_steps=None):
    # max_steps is now ignored or used only as a safety limit (e.g. 100k)
    print(f"--- Starting FULL EPISODE Video for {algo_name} ---")
    print("Note: This will run until Game Over. Press Ctrl+C to stop early.")
    
    # --- Helper: Robust RGB Extraction ---
    def _extract_ale_rgb(env):
        try:
            frame = env.render()
            if isinstance(frame, list): frame = frame[0]
            if frame is not None and frame.shape[-1] == 3: return frame
        except:
            pass
        try:
            return env.unwrapped.ale.getScreenRGB()
        except:
            return np.zeros((210, 160, 3), dtype=np.uint8)

    # --- Helper: Agent View (Max Projected) ---
    def _get_agent_obs_view(observation):
        img = observation.copy()
        if hasattr(img, 'cpu'): img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] < img.shape[1]: 
            img = np.moveaxis(img, 0, -1)
        if img.ndim == 3:
            img = np.max(img, axis=-1)
        if img.max() <= 1.0 + 1e-5:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img 
    # ---------------------------------------------------------

    obs, _ = env.reset(seed=args.seed)
    
    # Setup Dimensions
    rgb_ref = _extract_ale_rgb(env)
    h_rgb, w_rgb, _ = rgb_ref.shape 
    
    obs_ref = _get_agent_obs_view(obs)
    h_obs, w_obs = obs_ref.shape    
    
    TARGET_H = h_rgb * 2 
    FOOTER_H = 60
    FINAL_H = TARGET_H + FOOTER_H
    
    W_LEFT = w_rgb * 2
    W_RIGHT = w_obs * 5 
    TOTAL_W = W_LEFT + W_RIGHT
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (TOTAL_W, FINAL_H))

    render_buffer = []
    step = 0

    try:
        # CHANGED: Loop forever until done
        while True:
            if step % 50 == 0:
                print(f"Rendering frame {step}...")

            # --- 1. LEFT PANEL (RGB) ---
            raw_rgb = _extract_ale_rgb(env)
            render_buffer.append(raw_rgb)
            if len(render_buffer) > 2: render_buffer.pop(0)
            stable_rgb = np.max(np.stack(render_buffer), axis=0)
            bg_left = cv2.cvtColor(stable_rgb, cv2.COLOR_RGB2BGR)

            # --- 2. RIGHT PANEL (Agent) ---
            raw_obs_img = _get_agent_obs_view(obs)
            bg_right_gray = cv2.resize(raw_obs_img, (W_RIGHT, TARGET_H), interpolation=cv2.INTER_NEAREST)
            bg_right = cv2.cvtColor(bg_right_gray, cv2.COLOR_GRAY2BGR)

            # --- 3. SARFA CALC ---
            obs_in = obs
            if hasattr(obs_in, 'max') and obs_in.max() <= 1.0:
                 obs_in_sarfa = (obs_in * 255).astype(np.uint8)
            else:
                 obs_in_sarfa = obs_in.astype(np.uint8)

            heatmap = None
            action = 0

            if algo_name == "ddqn":
                heatmap, action = sarfa_heatmap(
                    agent, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode="mean", batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )
            elif algo_name == "ppo":
                heatmap, action = sarfa_heatmap_ppo(
                    agent, device, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )
            elif algo_name == "sac":
                heatmap, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device, obs_in_sarfa, patch=args.patch, stride=args.stride
                )

            # --- 4. VISUALS ---
            heatmap = blur_heatmap(heatmap, k=5)
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            heatmap[heatmap < 0.1] = 0.0 # Low threshold to show more

            # Resize heatmaps
            hm_left_small = cv2.resize(heatmap, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            hm_left_color = cv2.applyColorMap((hm_left_small * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            hm_right_large = cv2.resize(heatmap, (W_RIGHT, TARGET_H), interpolation=cv2.INTER_NEAREST)
            hm_right_color = cv2.applyColorMap((hm_right_large * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Blend Left
            mask_l = hm_left_small[..., None]
            final_left_small = (bg_left * (1.0 - mask_l*0.5) + hm_left_color * (mask_l*0.5)).astype(np.uint8)
            final_left = cv2.resize(final_left_small, (W_LEFT, TARGET_H), interpolation=cv2.INTER_NEAREST)

            # Blend Right (Smart Opacity)
            mask_r = cv2.resize(heatmap, (W_RIGHT, TARGET_H), interpolation=cv2.INTER_NEAREST)[..., None]
            is_bullet = bg_right_gray > 40
            alpha_map = np.full((TARGET_H, W_RIGHT), 0.6)
            alpha_map[is_bullet] = 0.1
            alpha_map = alpha_map[..., None]
            final_right = (bg_right * (1.0 - mask_r * alpha_map) + hm_right_color * (mask_r * alpha_map)).astype(np.uint8)

            # Stitch
            full_canvas = np.zeros((FINAL_H, TOTAL_W, 3), dtype=np.uint8)
            full_canvas[0:TARGET_H, 0:W_LEFT] = final_left
            full_canvas[0:TARGET_H, W_LEFT:TOTAL_W] = final_right
            cv2.line(full_canvas, (W_LEFT, 0), (W_LEFT, TARGET_H), (255, 255, 255), 2)
            
            # Text
            act_name = str(action)
            if "ACTION_NAMES" in cfg and action < len(cfg["ACTION_NAMES"]):
                act_name = cfg["ACTION_NAMES"][action]

            font = cv2.FONT_HERSHEY_SIMPLEX
            y_text = TARGET_H + 40
            cv2.putText(full_canvas, f"Step: {step}", (20, y_text), font, 0.8, (200, 200, 200), 2)
            cv2.putText(full_canvas, f"Action: {act_name}", (W_LEFT + 20, y_text), font, 0.8, (0, 255, 0), 2)
            cv2.putText(full_canvas, "Render", (W_LEFT - 100, TARGET_H - 10), font, 0.6, (255,255,255), 2)
            cv2.putText(full_canvas, "Agent+SARFA", (TOTAL_W - 160, TARGET_H - 10), font, 0.6, (255,255,255), 2)

            out.write(full_canvas)

            # --- 5. STEP ---
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            
            # Break if done
            if terminated or truncated:
                print(f"Episode finished at step {step}.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        out.release()
        print(f"Saved Full Episode video to: {out_path}")

def record_sarfa_video_5(env, agent, algo_name, args, cfg, device, out_path, max_steps=1000):

    print(f"--- Starting Final Video (Clean Score + More Heatmap) for {algo_name} ---")
    
    # --- Helper: Robust RGB Extraction ---
    def _extract_ale_rgb(env):
        try:
            frame = env.render()
            if isinstance(frame, list): frame = frame[0]
            if frame is not None and frame.shape[-1] == 3: return frame
        except:
            pass
        try:
            return env.unwrapped.ale.getScreenRGB()
        except:
            return np.zeros((210, 160, 3), dtype=np.uint8)

    # --- Helper: Agent View (Max Projected for Bullets) ---
    def _get_agent_obs_view(observation):
        img = observation.copy()
        if hasattr(img, 'cpu'): img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] < img.shape[1]: 
            img = np.moveaxis(img, 0, -1)
            
        # Max projection to see bullets from all stacked frames
        if img.ndim == 3:
            img = np.max(img, axis=-1)
            
        if img.max() <= 1.0 + 1e-5:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img 
    # ---------------------------------------------------------

    obs, _ = env.reset(seed=args.seed)
    
    # 1. Setup Dimensions
    rgb_ref = _extract_ale_rgb(env)
    h_rgb, w_rgb, _ = rgb_ref.shape # ~210x160
    
    obs_ref = _get_agent_obs_view(obs)
    h_obs, w_obs = obs_ref.shape    # ~84x84
    
    # Target Height (Upscale x2)
    TARGET_H = h_rgb * 2 
    
    # Dedicated Footer for Text (so we don't cover the Score)
    FOOTER_H = 60
    FINAL_H = TARGET_H + FOOTER_H
    
    # Widths
    W_LEFT = w_rgb * 2
    W_RIGHT = w_obs * 5 # Upscale 84 -> 420 approx
    TOTAL_W = W_LEFT + W_RIGHT
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (TOTAL_W, FINAL_H))

    # Buffer for Left Side smoothness
    render_buffer = []

    try:
        for step in range(max_steps):
            if step % 50 == 0:
                print(f"Rendering frame {step}/{max_steps}...")

            # --- LEFT PANEL (RGB) ---
            raw_rgb = _extract_ale_rgb(env)
            render_buffer.append(raw_rgb)
            if len(render_buffer) > 2: render_buffer.pop(0)
            stable_rgb = np.max(np.stack(render_buffer), axis=0)
            
            bg_left = cv2.cvtColor(stable_rgb, cv2.COLOR_RGB2BGR)

            # --- RIGHT PANEL (Agent) ---
            raw_obs_img = _get_agent_obs_view(obs)
            bg_right_gray = cv2.resize(raw_obs_img, (W_RIGHT, TARGET_H), interpolation=cv2.INTER_NEAREST)
            bg_right = cv2.cvtColor(bg_right_gray, cv2.COLOR_GRAY2BGR)

            # --- SARFA CALC ---
            obs_in = obs
            if hasattr(obs_in, 'max') and obs_in.max() <= 1.0:
                 obs_in_sarfa = (obs_in * 255).astype(np.uint8)
            else:
                 obs_in_sarfa = obs_in.astype(np.uint8)

            heatmap = None
            action = 0

            if algo_name == "ddqn":
                heatmap, action = sarfa_heatmap(
                    agent, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode="mean", batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )
            elif algo_name == "ppo":
                heatmap, action = sarfa_heatmap_ppo(
                    agent, device, obs_in_sarfa, patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )
            elif algo_name == "sac":
                heatmap, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device, obs_in_sarfa, patch=args.patch, stride=args.stride
                )

            # --- HEATMAP PROCESSING (Increased Visibility) ---
            heatmap = blur_heatmap(heatmap, k=5)
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # *** FIX: Show MORE heatmap ***
            # Previously: heatmap[heatmap < np.mean(heatmap)] = 0.0
            # Now: Only remove very low noise (bottom 10%)
            heatmap[heatmap < 0.1] = 0.0

            # Resize heatmaps
            hm_left_small = cv2.resize(heatmap, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            hm_left_color = cv2.applyColorMap((hm_left_small * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            hm_right_large = cv2.resize(heatmap, (W_RIGHT, TARGET_H), interpolation=cv2.INTER_NEAREST)
            hm_right_color = cv2.applyColorMap((hm_right_large * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # --- BLENDING ---
            # Left
            mask_l = hm_left_small[..., None]
            # Use 0.5 opacity so game is still very visible
            final_left_small = (bg_left * (1.0 - mask_l*0.5) + hm_left_color * (mask_l*0.5)).astype(np.uint8)
            final_left = cv2.resize(final_left_small, (W_LEFT, TARGET_H), interpolation=cv2.INTER_NEAREST)

            # Right (Agent View)
            # Smart opacity: if pixels are bright (bullets), make heatmap transparent
            mask_r = cv2.resize(heatmap, (W_RIGHT, TARGET_H), interpolation=cv2.INTER_NEAREST)[..., None]
            
            # Identify bullets in the gray image
            is_bullet = bg_right_gray > 40
            
            # Alpha map: default 0.6 (strong heatmap), but 0.1 on bullets
            alpha_map = np.full((TARGET_H, W_RIGHT), 0.6)
            alpha_map[is_bullet] = 0.1
            alpha_map = alpha_map[..., None]
            
            # Blend
            final_right = (bg_right * (1.0 - mask_r * alpha_map) + hm_right_color * (mask_r * alpha_map)).astype(np.uint8)

            # --- COMPOSITING ---
            # Create full canvas with Footer
            full_canvas = np.zeros((FINAL_H, TOTAL_W, 3), dtype=np.uint8)
            
            # Place images
            full_canvas[0:TARGET_H, 0:W_LEFT] = final_left
            full_canvas[0:TARGET_H, W_LEFT:TOTAL_W] = final_right
            
            # Draw Divider
            cv2.line(full_canvas, (W_LEFT, 0), (W_LEFT, TARGET_H), (255, 255, 255), 2)
            
            # --- FOOTER TEXT (Clean Score) ---
            act_name = str(action)
            if "ACTION_NAMES" in cfg and action < len(cfg["ACTION_NAMES"]):
                act_name = cfg["ACTION_NAMES"][action]

            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_text = TARGET_H + 40 # Position in footer
            
            cv2.putText(full_canvas, f"Step: {step}", (20, y_text), font, 0.8, (200, 200, 200), 2)
            cv2.putText(full_canvas, f"Action: {act_name}", (W_LEFT + 20, y_text), font, 0.8, (0, 255, 0), 2)
            
            # Labels
            cv2.putText(full_canvas, "Render", (W_LEFT - 100, TARGET_H - 10), font, 0.6, (255,255,255), 2)
            cv2.putText(full_canvas, "Agent+SARFA", (TOTAL_W - 160, TARGET_H - 10), font, 0.6, (255,255,255), 2)

            out.write(full_canvas)

            # Step
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print("Episode finished early.")
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        out.release()
        print(f"Saved video to: {out_path}")

def main():

    # load config from config.yaml
    cfg = load_config("config.yaml")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=int, default=cfg["sarfa"]["patch"])
    parser.add_argument("--stride", type=int, default=cfg["sarfa"]["stride"])
    parser.add_argument("--outdir", type=str, default=cfg["sarfa"]["outdir"])
    parser.add_argument("--seed", type=int, default=cfg["sarfa"]["seed"])
    parser.add_argument("--algo", type=str, default="ddqn", choices=["ddqn", "ppo", "sac", "all"])
    parser.add_argument("--ddqn_model", type=str, default=cfg["ddqn"]["path_best_model"])
    parser.add_argument("--ppo_model", type=str, default=cfg["ppo"]["path_best_model"])
    parser.add_argument("--sac_model", type=str, default=cfg["sac"]["path_best_model"])
    parser.add_argument("--snap_steps", type=int, nargs="+", default=[50, 600, 1500],
                    help="Environment steps (after reset) at which to take SARFA snapshots (early/mid/late).")
    parser.add_argument("--snap_names", type=str, nargs="+", default=["early", "mid", "late"],
                    help="Names for snapshots; must match snap_steps length.")

    # === NEW ARGUMENTS ===
    parser.add_argument("--video", action="store_true", help="If set, record a video instead of snapshots.")
    parser.add_argument("--video_length", type=int, default=200, help="Number of frames to record.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up environment

    try:
        env = make_env(env_id=cfg["env"]["id"], seed=args.seed, render_mode="rgb_array")
    except:
        env = make_env(env_id=cfg["env"]["id"], seed=args.seed)

    
    # --- Collect early/mid/late snapshots ---
    if len(args.snap_names) != len(args.snap_steps):
        raise ValueError("--snap_names length must match --snap_steps length")

    print(f"Collecting snapshots at steps: {list(zip(args.snap_names, args.snap_steps))}")
    snapshots = _collect_snapshots(env, seed=args.seed, snap_steps=args.snap_steps)

    # Map step -> name for nicer filenames
    step_to_name = {s: n for n, s in zip(args.snap_names, args.snap_steps)}

    
    print(f"Computing SARFA for patch={args.patch} on device={device}...")
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    outdir = os.path.join(args.outdir, date)
    os.makedirs(outdir, exist_ok=True)


    # === VIDEO GENERATION BLOCK ===
    if args.video:
        algos_to_run = ["ddqn", "ppo", "sac"] if args.algo == "all" else [args.algo]
        
        for algo in algos_to_run:
            video_path = os.path.join(outdir, f"video_sarfa_{algo}_{time}.mp4")
            
            # Load specific agent
            if algo == "ddqn":
                print("Loading DDQN...")
                agent = DDQN_Agent(env, obs_shape[0], obs_shape, n_actions, device, double_dqn=True)
                agent.load(args.ddqn_model)
                record_sarfa_video_4(env, agent, "ddqn", args, cfg, device, video_path, args.video_length)

            elif algo == "ppo":
                print("Loading PPO...")
                ckpt = torch.load(args.ppo_model, map_location=device)
                agent = ActorCriticCNN(in_channels=cfg["sarfa"]["ppo"]["in_channels"], n_actions=n_actions).to(device)
                agent.load_state_dict(ckpt["net"])
                agent.eval()
                record_sarfa_video_4(env, agent, "ppo", args, cfg, device, video_path, args.video_length)

            elif algo == "sac":
                print("Loading SAC...")
                agent = SACDiscrete_Agent(obs_shape=obs_shape, n_actions=n_actions, device=device, env_id=cfg["env"]["id"])
                agent.load(args.sac_model)
                record_sarfa_video_4(env, agent, "sac", args, cfg, device, video_path, args.video_length)
                
        env.close()
        return  # Exit after video to avoid running snapshot logic
    # ==============================

    # always run all 3 algos when doing snapshot analysis
    algos_to_run = ["ddqn", "ppo", "sac"] if args.algo in ["all"] else [args.algo]

    # store paths for final 3x3 grid: grid_paths[snap_name][algo] = png_path
    grid_paths = {}


    for (t, obs, rgb_bg) in snapshots:
        snap_name = step_to_name.get(t, f"t{t}")
        print(f"\n=== Snapshot {snap_name} (step={t}) ===")

        saved_paths = []

        for algo in algos_to_run:
            # --- your existing DDQN / PPO / SAC blocks ---
            if algo == "ddqn":
                print("Using DDQN agent for SARFA...")
                agent = DDQN_Agent(env, obs_shape[0], obs_shape, n_actions, device, double_dqn=True)
                agent.load(args.ddqn_model)

                heat, action = sarfa_heatmap(
                    agent, obs,
                    patch=args.patch,
                    stride=args.stride, 
                    fill_mode="mean",
                    batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                    use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                    clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                    use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                    flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                )

            elif algo == "ppo":
                print("Using PPO agent for SARFA...")
                ckpt = torch.load(args.ppo_model, map_location=device)
                actor_critic = ActorCriticCNN(in_channels=cfg["sarfa"]["ppo"]["in_channels"],
                                            n_actions=n_actions).to(device)
                actor_critic.load_state_dict(ckpt["net"])
                actor_critic.eval()

                heat, action = sarfa_heatmap_ppo(
                    actor_critic, device, obs,
                    patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )

            else:  # sac
                print("Using SAC agent for SARFA...")
                agent = SACDiscrete_Agent(obs_shape=obs.shape, n_actions=n_actions, device=device, env_id=cfg["env"]["id"])
                agent.load(args.sac_model)

                heat, action = sarfa_heatmap_policy_logp(
                    lambda x: sac_policy_logits(agent, x),
                    device,
                    obs,
                    patch=args.patch,
                    stride=args.stride
                )

            # --- your existing visualization code, but use rgb_bg from this snapshot ---
            heat_vis = blur_heatmap(heat, k=7)
            if heat_vis.max() > 0:
                heat_vis = (heat_vis - heat_vis.min()) / (heat_vis.max() - heat_vis.min() + 1e-8)
                thr = np.mean(heat_vis)
                heat_vis[heat_vis < thr] = 0.0

            rgb = rgb_bg
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.array(rgb)
            if len(rgb.shape) == 2:
                rgb = np.stack([rgb]*3, axis=-1)
            if rgb.shape[:2] != heat_vis.shape:
                heat_vis = cv2.resize(heat_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            cmap = plt.get_cmap("jet")
            overlay = cmap(heat_vis)
            overlay[..., 3] = heat_vis * 0.7

            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.imshow(overlay)
            plt.axis("off")
            plt.title(f"SARFA {snap_name} (Action: {cfg['ACTION_NAMES'][action]}) - {algo} - MaxHeat: {heat.max():.4f}")

            png_path = os.path.join(outdir, f"sarfa_{snap_name}_{algo}_{time}.png")
            plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"[SARFA] Saved: {png_path}")
            saved_paths.append(png_path)

            grid_paths.setdefault(snap_name, {})[algo] = png_path


        # side-by-side per snapshot
        if len(saved_paths) == 3:
            combo_path = os.path.join(outdir, f"sarfa_{snap_name}_ALL_{time}.png")
            _save_side_by_side(saved_paths, combo_path)
            print(f"[SARFA] Saved side-by-side: {combo_path}")

        # final 3x3 grid
        # final 3x3 grid
        # if user asked for all, save a single 3x3 grid (early/mid/late x ddqn/ppo/sac)
        if args.algo == "all":
            snap_order = args.snap_names  # ["early","mid","late"]
            algo_order = ["ddqn", "ppo", "sac"]
            grid_out = os.path.join(outdir, f"sarfa_GRID_{time}.png")
            _save_grid_3x3(grid_paths, snap_order, algo_order, grid_out)
            print(f"[SARFA] Saved 3x3 grid: {grid_out}")

            
        env.close()
    
    

if __name__ == "__main__":
    main()
