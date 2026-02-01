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

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up environment

    try:
        env = make_env(env_id=cfg["env"]["id"], seed=args.seed, render_mode="rgb_array")
    except:
        env = make_env(env_id=cfg["env"]["id"], seed=args.seed)

    # reset and workout part
    obs, _ = env.reset(seed=args.seed)
    
    print("Doing warmup with 50 step...")
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # we directly extract the screenshot from the ALE 
    ale = None
    current_env = env
    while hasattr(current_env, 'env'):
        if hasattr(current_env, 'ale'):
            ale = current_env.ale
            break
        current_env = current_env.env
    if ale is None and hasattr(current_env, 'ale'):
        ale = current_env.ale

    rgb_bg = None
    if ale is not None:
        try:
            rgb_bg = ale.getScreenRGB()
        except:
            pass
    if rgb_bg is None:
        rgb_bg = env.render()

    # computing SARFA
    print(f"Computing Sarfa for patch={args.patch} on device={device}...")
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # check if input is normalized
    if obs.max() <= 1.0:
        print("[WARNING] observation is normalized (0-1). Multiplying by 255.")
        obs = (obs * 255).astype(np.uint8)

    algos_to_run = ["ddqn", "ppo", "sac"] if args.algo == "all" else [args.algo]
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    outdir = os.path.join(args.outdir, date)
    os.makedirs(outdir, exist_ok=True)
    
    saved_paths = []

    for algo in algos_to_run:

        if algo == "ddqn":
            print("Using DDQN agent for SARFA...")
            agent = DDQN_Agent(env, obs_shape[2], n_actions, device, double_dqn=True)   
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
                                        n_actions=n_actions)
            actor_critic = actor_critic.to(device)
            actor_critic.load_state_dict(ckpt["net"])
            actor_critic.eval()
            heat, action = sarfa_heatmap_ppo(actor_critic,
                                            device,
                                            obs,
                                            patch=args.patch,
                                            stride=args.stride,
                                            fill_mode=cfg["sarfa"]["ppo"]["fill_mode"], 
                                            batch_size=cfg["sarfa"]["ppo"]["batch_size"])
        
        else:  # sac
            print("Using SAC agent for SARFA...")
            ckpt = torch.load(args.sac_model, map_location=device)

            agent = SACDiscrete_Agent(  obs_shape=obs.shape,          # (84,84,4)
                                        n_actions=n_actions,
                                        device=device,
                                        env_id=cfg["env"]["id"]
                                    )

            agent.load(args.sac_model)

            heat, action = sarfa_heatmap_policy_logp(
                lambda x: sac_policy_logits(agent, x),
                device,
                obs,
                patch=args.patch,
                stride=args.stride
            )
        



        # DEBUG: Controlliamo se la heatmap è vuota
        #print(f"[DEBUG] Heatmap Max Value: {heat.max():.6f}")
        #print(f"[DEBUG] Heatmap Mean Value: {heat.mean():.6f}")

        if heat.max() == 0:
            print("[ERROR] Heatmap is all zeros!")
            print(" -> Try reducing the patch size or check if the model always predicts the same action.")

        # robust visualization with blurring and normalization
        heat_vis = blur_heatmap(heat, k=7)
        
        # removing aggressive thresholding if values are too low
        # and using a min-max standard normalization
        if heat_vis.max() > 0:
            heat_vis = (heat_vis - heat_vis.min()) / (heat_vis.max() - heat_vis.min() + 1e-8)
            
            
            #applying a softer threshold: we show everything above mean
            thr = np.mean(heat_vis) 
            heat_vis[heat_vis < thr] = 0.0
        
        # background image loading....
        if hasattr(rgb_bg, 'detach'): rgb_bg = rgb_bg.detach().cpu().numpy()
        rgb_bg = np.array(rgb_bg)
        if len(rgb_bg.shape) == 2:
            rgb_bg = np.stack([rgb_bg]*3, axis=-1)

        # Resize Heatmap
        if rgb_bg.shape[:2] != heat_vis.shape:
            heat_vis = cv2.resize(heat_vis, (rgb_bg.shape[1], rgb_bg.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay
        cmap = plt.get_cmap('jet')
        overlay = cmap(heat_vis)
        
        # adaptive transparency the more is hot, the more is opaque (up to 0.7)
        overlay[..., 3] = heat_vis * 0.7 

        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_bg)
        plt.imshow(overlay)
        
        plt.axis("off")
        plt.title(f"SARFA (Action: {action}) - {algo} - MaxHeat: {heat.max():.4f}")
   

        png_path = os.path.join(outdir, f"sarfa_{algo}_{time}.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[SARFA] Saved: {png_path}")
        saved_paths.append(png_path)

    if args.algo == "all" and len(saved_paths) == 3:
        combo_path = os.path.join(outdir, f"sarfa_ALL_{time}.png")
        _save_side_by_side(saved_paths, combo_path)
        print(f"[SARFA] Saved side-by-side: {combo_path}")
        
    env.close()
    

if __name__ == "__main__":
    main()
