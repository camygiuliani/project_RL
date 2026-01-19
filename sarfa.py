import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import gymnasium as gym
from datetime import datetime



from wrappers import make_env
from ppo_agent import ActorCriticCNN
from dqn_agent import DQN_Agent,DQNCNN
from sac_discrete import SACDiscreteAgent


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
    # mean per canale (shape (1,1,4)) poi broadcasta
    fill = obs_uint8.mean(axis=(0, 1), keepdims=True)
    return fill.astype(obs_uint8.dtype)

#deactivate gradients for this function, more efficient
@torch.no_grad()
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
    
    # Rinominiamo
    obs = obs_input
    
    # 1. Preparazione Input
    if obs.dtype != np.uint8:
        if obs.max() <= 1.0: 
            obs = (obs * 255).astype(np.uint8)
        else: 
            obs = obs.astype(np.uint8)
        
    # Gym restituisce (H, W, C), PyTorch vuole (C, H, W)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    obs_tensor = obs_tensor.permute(2, 0, 1) 
    obs_tensor = obs_tensor.unsqueeze(0) / 255.0
    
    # 2. Baseline
    with torch.no_grad():
        # --- FIX NOME VARIABILE ---
        # Nel tuo agente la rete si chiama 'q', non 'q_net'
        q_values = agent.q(obs_tensor) 
        # --------------------------
        
        base_action = torch.argmax(q_values, dim=1).item()
        base_q = q_values[0, base_action].item()

    # 3. Setup Griglia
    H, W, C = obs.shape
    coords = [(x, y) for y in range(0, H-patch+1, stride) for x in range(0, W-patch+1, stride)]
    heatmap = np.zeros((H, W), dtype=np.float32)

    if fill_mode == "mean": fill_val = np.mean(obs)
    else: fill_val = 0

    # 4. Ciclo Batch
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i+batch_size]
        batch_np = np.repeat(obs[None, ...], len(chunk), axis=0).copy()

        for idx, (x, y) in enumerate(chunk):
            batch_np[idx, y:y+patch, x:x+patch, :] = fill_val 

        batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=agent.device)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2) 
        batch_tensor = batch_tensor / 255.0

        with torch.no_grad():
            # --- FIX NOME VARIABILE ---
            q_perturbed = agent.q(batch_tensor) 
            # --------------------------

        # 5. Calcolo Score
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
# PPO version (SCRATCH)
# =========================

@torch.no_grad()
def ppo_scratch_logprobs_batch(actor_critic_net, device, obs_uint8_batch, actions=None):
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

    logits, _ = actor_critic_net(x)  # logits: (N,A)
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

#original patch=8 stride=8

def sarfa_heatmap_ppo_scratch(model, device, obs_input, patch=8, stride=4, 
                              fill_mode="mean", batch_size=32, 
                              clamp_positive=True, use_action_flip=True, flip_weight=2.0):
    
    # Rinominiamo subito per sicurezza
    obs = obs_input

    # 1. Preparazione Input Base
    if obs.dtype != np.uint8:
        if obs.max() <= 1.0: 
            obs = (obs * 255).astype(np.uint8)
        else: 
            obs = obs.astype(np.uint8)

    # --- FIX DIMENSIONI (H,W,C -> C,H,W) ---
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.permute(2, 0, 1) # (4, 84, 84)
    obs_tensor = obs_tensor.unsqueeze(0) / 255.0 # (1, 4, 84, 84)
    # ---------------------------------------

    with torch.no_grad():
        # --- FIX ATTRIBUTE ERROR ---
        # Il modello restituisce (logits, values) oppure (distribution, values).
        # Controlliamo cosa abbiamo ricevuto.
        output, _ = model(obs_tensor)
        
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output # È già il tensore dei logits

        base_probs = torch.softmax(logits, dim=-1)
        base_action = torch.argmax(base_probs, dim=-1).item()
        base_prob_val = base_probs[0, base_action].item()
        # ---------------------------

    # 2. Coordinate della griglia
    H, W, C = obs.shape
    y_range = range(0, H - patch + 1, stride)
    x_range = range(0, W - patch + 1, stride)
    coords = [(x, y) for y in y_range for x in x_range]

    heatmap = np.zeros((H, W), dtype=np.float32)

    if fill_mode == "mean":
        fill_val = np.mean(obs)
    else:
        fill_val = 0 

    # 3. Ciclo su Batch
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i+batch_size]
        n_chunk = len(chunk)

        # Creiamo il batch numpy: (Batch, H, W, C)
        batch_np = np.repeat(obs[None, ...], n_chunk, axis=0).copy()

        # Applichiamo le maschere (Deep Masking su tutti i canali)
        for idx, (x, y) in enumerate(chunk):
            batch_np[idx, y:y+patch, x:x+patch, :] = fill_val 

        # --- FIX DIMENSIONI BATCH ---
        batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2) # (Batch, 4, 84, 84)
        batch_tensor = batch_tensor / 255.0
        # ----------------------------

        with torch.no_grad():
            # --- FIX ATTRIBUTE ERROR (Anche qui nel loop) ---
            output_p, _ = model(batch_tensor)
            
            if hasattr(output_p, 'logits'):
                logits_p = output_p.logits
            else:
                logits_p = output_p
            
            probs_p = torch.softmax(logits_p, dim=-1)
            # -----------------------------------------------
        
        # 4. Calcolo Punteggio SARFA
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/dqn_step_200000.pt")
    parser.add_argument("--env", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--ppo_model", type=str, default="runs/ppo_16_jan/ppo_final.pt")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SETUP AMBIENTE
    try:
        env = make_env(env_id=args.env, seed=args.seed, render_mode="rgb_array")
    except:
        env = make_env(env_id=args.env, seed=args.seed)

    # 2. RESET E WARMUP
    obs, _ = env.reset(seed=args.seed)
    
    print("Eseguo warmup (50 step)...")
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # 3. ESTRAZIONE DIRETTA SCREENSHOT
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

    # 4. CALCOLO SARFA
    print(f"Calcolo SARFA con patch={args.patch} su device={device}...")
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Assicuriamoci che l'input sia corretto
    if obs.max() <= 1.0:
        print("[WARNING] L'osservazione sembra normalizzata (0-1). Moltiplico per 255.")
        obs = (obs * 255).astype(np.uint8)

    if args.algo == "dqn":
        agent = DQN_Agent(args.env, obs_shape[2], n_actions, device, double_dqn=False)   
        agent.load(args.model)
        
        heat, action = sarfa_heatmap(
            agent, obs,
            patch=args.patch, stride=args.stride,
            fill_mode="mean", batch_size=64,
            use_advantage=True, clamp_positive=True,
            use_action_flip=True, flip_weight=2.0
        )
    else:
        ckpt = torch.load(args.ppo_model, map_location=device)
        actor_critic = ActorCriticCNN(in_channels=4, n_actions=n_actions).to(device)
        actor_critic.load_state_dict(ckpt["net"])
        actor_critic.eval()
        heat, action = sarfa_heatmap_ppo_scratch(actor_critic, device, obs, patch=args.patch, stride=args.stride, fill_mode="mean", batch_size=64)

    # DEBUG: Controlliamo se la heatmap è vuota
    print(f"[DEBUG] Heatmap Max Value: {heat.max():.6f}")
    print(f"[DEBUG] Heatmap Mean Value: {heat.mean():.6f}")

    if heat.max() == 0:
        print("[ERRORE] La heatmap è completamente vuota (tutti zeri)!")
        print(" -> Prova a ridurre la dimensione della patch o controllare se il modello predice sempre la stessa azione.")

    # 5. VISUALIZZAZIONE ROBUSTA
    heat_vis = blur_heatmap(heat, k=7)
    
    # Rimuoviamo il threshold aggressivo se i valori sono bassi
    # Usiamo una normalizzazione Min-Max standard
    if heat_vis.max() > 0:
        heat_vis = (heat_vis - heat_vis.min()) / (heat_vis.max() - heat_vis.min() + 1e-8)
        
        # Applichiamo un threshold più morbido: mostriamo tutto ciò che è sopra la media
        thr = np.mean(heat_vis) 
        heat_vis[heat_vis < thr] = 0.0
    
    # Preparazione immagine background
    if hasattr(rgb_bg, 'detach'): rgb_bg = rgb_bg.detach().cpu().numpy()
    rgb_bg = np.array(rgb_bg)
    if len(rgb_bg.shape) == 2:
        rgb_bg = np.stack([rgb_bg]*3, axis=-1)

    # Resize Heatmap
    if rgb_bg.shape[:2] != heat_vis.shape:
        heat_vis = cv2.resize(heat_vis, (rgb_bg.shape[1], rgb_bg.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    cmap = plt.get_cmap('jet') # O 'hot' per vedere meglio il rosso
    overlay = cmap(heat_vis)
    
    # Trasparenza adattiva: più è caldo, più è opaco (fino a 0.7)
    overlay[..., 3] = heat_vis * 0.7 

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_bg)
    plt.imshow(overlay)
    
    plt.axis("off")
    plt.title(f"SARFA (Action: {action}) - {args.algo} - MaxHeat: {heat.max():.4f}")
    
    timestamp = datetime.now().strftime("%H_%M_%S")
    png_path = os.path.join(args.outdir, f"sarfa_{args.algo}_{timestamp}.png")
    
    plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    env.close()
    print(f"[SARFA] Saved: {png_path}")

if __name__ == "__main__":
    main()
