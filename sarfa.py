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
from wrappers import make_env_eval
from ppo import ActorCriticCNN
from ddqn import DDQN_Agent,DDQNCNN
from sac import SACDiscrete_Agent


def _get_fill_value(obs_uint8, mode="mean"):
    # here we decide how to fill the occluded patch
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

def sarfa_heatmap_DDQN(agent, obs_input, patch=8, stride=4, 
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

def sarfa_heatmap_SAC(
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

def sarfa_heatmap_PPO(model, device, obs_input, patch=8, stride=4, 
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
        output, _ = model(obs_tensor)
        
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output 

        base_probs = torch.softmax(logits, dim=-1)
        base_action = torch.argmax(base_probs, dim=-1).item()
        base_prob_val = base_probs[0, base_action].item()

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
            output_p, _ = model(batch_tensor)
            
            if hasattr(output_p, 'logits'):
                logits_p = output_p.logits
            else:
                logits_p = output_p
            
            probs_p = torch.softmax(logits_p, dim=-1)
        
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

""" def _extract_ale_rgb(env):
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
 """
def _extract_ale_rgb(env):
    """
    Returns the true game RGB frame (210x160x3) from ALE if possible,
    otherwise falls back to env.render().
    This avoids "double bullets"/ghosting artifacts.
    """
    # Unwrap to base env that has ALE
    e = env
    while hasattr(e, "env"):
        if hasattr(e, "ale"):
            break
        e = e.env

    # Try ALE first (most reliable for Atari details like bullets)
    try:
        if hasattr(e, "ale") and e.ale is not None:
            frame = e.ale.getScreenRGB()
            if frame is not None:
                return frame
    except Exception:
        pass

    # Fallback: gym render
    try:
        frame = env.render()
        if isinstance(frame, list):
            frame = frame[0]
        return frame
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

""" def _extract_ale_rgb(env):
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
"""

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

def overlay_sarfa_contours(frame_bgr, heatmap_84, thr=0.25):
    """
    frame_bgr: (210,160,3)
    heatmap_84: (84,84) in [0,1]
    Draws contours of salient regions so bullets/HUD remain visible.
    """
    h, w = frame_bgr.shape[:2]
    hm = cv2.resize(heatmap_84, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (hm >= thr).astype(np.uint8) * 255

    # clean small noise
    mask = cv2.medianBlur(mask, 3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = frame_bgr.copy()
    # draw thin contours (non invasivo)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)  # giallo
    return out

def overlay_sarfa_colormap(frame_bgr, heatmap_84, thr=0.15, alpha_max=0.45, preserve_bright=True):
    """
    frame_bgr: (H,W,3) uint8 BGR
    heatmap_84: (84,84) float in [0,1] (o simile)
    """
    h, w = frame_bgr.shape[:2]

    # resize heatmap to frame size
    hm = cv2.resize(heatmap_84.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    hm = np.clip(hm, 0.0, 1.0)

    # mask 2D: dove applicare overlay
    mask2d = (hm >= thr).astype(np.float32)  # 0/1
    mask = mask2d[..., None]                 # (H,W,1)

    # colormap
    hm_u8 = (hm * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO).astype(np.float32)  # (H,W,3)

    # alpha per pixel (H,W,1)
    alpha = (alpha_max * hm).astype(np.float32)[..., None]

    base = frame_bgr.astype(np.float32)

    # blend: base*(1-a) + cmap*a, ma SOLO dove mask=1
    blended = base * (1.0 - alpha) + cmap * alpha
    out = base * (1.0 - mask) + blended * mask
    out = np.clip(out, 0, 255).astype(np.uint8)

    # preserve bright pixels (bullets/HUD)
    if preserve_bright:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bright = gray > 200
        out[bright] = frame_bgr[bright]

    return out


        
def record_sarfa_video(env, agent, algo_name, args, cfg, device, out_path, max_steps=None):
    
    print(f"--- Starting FULL EPISODE Video for {algo_name} ---")
    print("Note: This will run until Game Over. Press Ctrl+C to stop early.")
    
    obs, _ = env.reset(seed=args.seed)
    
    rgb_ref = _extract_ale_rgb(env)
    h_rgb, w_rgb, _ = rgb_ref.shape  # 210x160

    SCALE = 4
    OUT_W, OUT_H = w_rgb * SCALE, h_rgb * SCALE  # 640x840

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (OUT_W, OUT_H))


    render_buffer = []
    step = 0

    
    ACTION_REPEAT = 4  # simula frame_skip=4 ma salvando OGNI frame

    step = 0
    terminated = False
    truncated = False

    try:
             while True:
                        if step % 50 == 0:
                                 print(f"Rendering frame {step}...")

            



                        # --- SARFA CALC on current obs ---
                        obs_in = obs
                        if hasattr(obs_in, 'max') and obs_in.max() <= 1.0:
                            obs_in_sarfa = (obs_in * 255).astype(np.uint8)
                        else:
                            obs_in_sarfa = obs_in.astype(np.uint8)

                        if algo_name == "ddqn":
                            heatmap, action = sarfa_heatmap_DDQN(
                                agent, obs_in_sarfa, patch=args.patch, stride=args.stride,
                                fill_mode="mean", batch_size=cfg["sarfa"]["ddqn"]["batch_size"],
                                use_advantage=cfg["sarfa"]["ddqn"]["use_advantage"],
                                clamp_positive=cfg["sarfa"]["ddqn"]["clamp_positive"],
                                use_action_flip=cfg["sarfa"]["ddqn"]["use_action_flip"],
                                flip_weight=cfg["sarfa"]["ddqn"]["flip_weight"]
                            )
                        elif algo_name == "ppo":
                            heatmap, action = sarfa_heatmap_PPO(
                                agent, device, obs_in_sarfa, patch=args.patch, stride=args.stride,
                                fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                                batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                            )
                        else:  # sac
                            heatmap, action = sarfa_heatmap_SAC(
                                lambda x: sac_policy_logits(agent, x),
                                device, obs_in_sarfa, patch=args.patch, stride=args.stride
                            )

                        # normalize heatmap 0..1
                        heatmap = blur_heatmap(heatmap, k=5)
                        if heatmap.max() > 0:
                            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                        # --- repeat same action 4 times, BUT save every frame ---
                        for _ in range(ACTION_REPEAT):
                            obs, _, terminated, truncated, _ = env.step(action)
                            step += 1

                            rgb = _extract_ale_rgb(env)   # <-- AFTER step!
                            if rgb is None:
                                terminated = True
                                break

                            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                            # overlay NON distruttivo: contorni (bullets restano)
                            #frame = overlay_sarfa_contours(frame, heatmap, thr=0.25)
                            frame = overlay_sarfa_colormap(frame, heatmap, thr=0.15, alpha_max=0.35, preserve_bright=True)


                            # upscale for readability
                            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)

                            out.write(frame)

                            if terminated or truncated:
                                break

                        if terminated or truncated:
                            print(f"Episode finished at step {step}.")
                            break

    except KeyboardInterrupt:
        print("Video recording interrupted by user.")



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
    parser.add_argument("--video", action="store_true", help="If set, record a video instead of snapshots.")
    parser.add_argument("--video_length", type=int, default=200, help="Number of frames to record.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up environment

    try:
        env = make_env_eval(env_id=cfg["env"]["id"], seed=args.seed, render_mode="rgb_array")
    except:
        env = make_env_eval(env_id=cfg["env"]["id"], seed=args.seed)

    if args.video:
    # IMPORTANT: frame_skip=1 for video so bullets are visible
        env = make_env_eval(env_id=cfg["env"]["id"], seed=args.seed, frame_skip=1, render_mode="rgb_array")

    
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
                record_sarfa_video(env, agent, "ddqn", args, cfg, device, video_path, args.video_length)

            elif algo == "ppo":
                print("Loading PPO...")
                ckpt = torch.load(args.ppo_model, map_location=device)
                agent = ActorCriticCNN(in_channels=cfg["sarfa"]["ppo"]["in_channels"], n_actions=n_actions).to(device)
                agent.load_state_dict(ckpt["net"])
                agent.eval()
                record_sarfa_video(env, agent, "ppo", args, cfg, device, video_path, args.video_length)

            elif algo == "sac":
                print("Loading SAC...")
                agent = SACDiscrete_Agent(obs_shape=obs_shape, n_actions=n_actions, device=device, env_id=cfg["env"]["id"])
                agent.load(args.sac_model)
                record_sarfa_video(env, agent, "sac", args, cfg, device, video_path, args.video_length)
                
        env.close()
        return
  
    algos_to_run = ["ddqn", "ppo", "sac"] if args.algo in ["all"] else [args.algo]
    grid_paths = {}


    for (t, obs, rgb_bg) in snapshots:
        snap_name = step_to_name.get(t, f"t{t}")
        print(f"\n=== Snapshot {snap_name} (step={t}) ===")

        saved_paths = []

        for algo in algos_to_run:
            if algo == "ddqn":
                print("Using DDQN agent for SARFA...")
                agent = DDQN_Agent(env, obs_shape[0], obs_shape, n_actions, device, double_dqn=True)
                agent.load(args.ddqn_model)

                heat, action = sarfa_heatmap_DDQN(
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

                heat, action = sarfa_heatmap_PPO(
                    actor_critic, device, obs,
                    patch=args.patch, stride=args.stride,
                    fill_mode=cfg["sarfa"]["ppo"]["fill_mode"],
                    batch_size=cfg["sarfa"]["ppo"]["batch_size"]
                )

            else:  # sac
                print("Using SAC agent for SARFA...")
                agent = SACDiscrete_Agent(obs_shape=obs.shape, n_actions=n_actions, device=device, env_id=cfg["env"]["id"])
                agent.load(args.sac_model)

                heat, action = sarfa_heatmap_SAC(
                    lambda x: sac_policy_logits(agent, x),
                    device,
                    obs,
                    patch=args.patch,
                    stride=args.stride
                )

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
