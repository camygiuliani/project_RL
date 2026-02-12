from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config
from wrappers import make_env_eval
import torch
import csv
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import sarfa



# =========================
# Helpers
# =========================

def summarize(results):
    out= {}
    by= {}
    for r in results:
        by.setdefault(r.condition, []).append(r)

    for cond, rr in by.items():
        rets = np.array([x.return_sum for x in rr], dtype=np.float32)
        lens = np.array([x.length for x in rr], dtype=np.float32)
        out[cond] = {
            "n": float(len(rr)),
            "mean_return": float(rets.mean()) if len(rr) else 0.0,
            "std_return": float(rets.std(ddof=1)) if len(rr) > 1 else 0.0,
            "mean_len": float(lens.mean()) if len(rr) else 0.0,
            "std_len": float(lens.std(ddof=1)) if len(rr) > 1 else 0.0,
        }
    return out

def _extract_ale_rgb(env):
    """Returns the true game RGB frame from ALE if possible."""
    e = env
    while hasattr(e, "env"):
        if hasattr(e, "ale"):
            break
        e = e.env
    try:
        if hasattr(e, "ale") and e.ale is not None:
            frame = e.ale.getScreenRGB()
            if frame is not None:
                return frame
    except Exception:
        pass
    # Fallback
    try:
        frame = env.render()
        if isinstance(frame, list): frame = frame[0]
        return frame
    except Exception:
        return None

def to_hwc(x: np.ndarray) -> Tuple[np.ndarray, str]:
    x = np.asarray(x)
    if x.ndim == 2: return x[:, :, None], "HW"
    if x.ndim != 3: raise ValueError(f"Expected 2D/3D array, got shape={x.shape}")
    if x.shape[-1] in (1, 3, 4): return x, "HWC"
    if x.shape[0] in (1, 3, 4): return np.transpose(x, (1, 2, 0)), "CHW"
    return x, "HWC"

def from_hwc(x_hwc: np.ndarray, tag: str) -> np.ndarray:
    if tag == "CHW": return np.transpose(x_hwc, (2, 0, 1))
    if tag == "HW": return x_hwc[:, :, 0]
    return x_hwc

def normalize_heatmap(heat_hw: np.ndarray) -> np.ndarray:
    h = np.asarray(heat_hw, dtype=np.float32)
    if h.ndim == 3: h = h.mean(axis=-1)
    mn, mx = float(h.min()), float(h.max())
    if mx - mn < 1e-8: return np.zeros_like(h, dtype=np.float32)
    return (h - mn) / (mx - mn)

@dataclass
class OcclusionConfig:
    patch: int = 8
    k: int = 10
    mode: str = "zero"
    seed: int = 0

def grid_shape(H: int, W: int, patch: int) -> Tuple[int, int]:
    return math.ceil(H / patch), math.ceil(W / patch)

def random_patch_indices(H: int, W: int, patch: int, k: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    gy, gx = grid_shape(H, W, patch)
    all_idx = [(py, px) for py in range(gy) for px in range(gx)]
    k = min(k, len(all_idx))
    chosen = rng.choice(len(all_idx), size=k, replace=False)
    return [all_idx[i] for i in chosen]

def topk_patch_indices_from_heat(heat_hw: np.ndarray, patch: int, k: int) -> List[Tuple[int, int]]:
    heat = normalize_heatmap(heat_hw)
    H, W = heat.shape
    gy, gx = grid_shape(H, W, patch)
    scored = []
    for py in range(gy):
        for px in range(gx):
            y0, x0 = py * patch, px * patch
            y1, x1 = min(y0 + patch, H), min(x0 + patch, W)
            s = float(heat[y0:y1, x0:x1].mean())
            scored.append(((py, px), s))
    scored.sort(key=lambda t: t[1], reverse=True)
    k = min(k, len(scored))
    return [scored[i][0] for i in range(k)]

def occlude_patches_hwc(obs_hwc: np.ndarray, patch: int, patch_indices: List[Tuple[int, int]], mode: str) -> np.ndarray:
    out = obs_hwc.copy()
    H, W, C = out.shape

    # Determine fill value
    if mode == "mean":
        fill = out.mean(axis=(0, 1), keepdims=True).astype(out.dtype)
    elif mode == "gray":
        val = 127 if out.dtype == np.uint8 else 0.5
        fill = np.full((1, 1, C), val, dtype=out.dtype)
    elif mode == "zero":
        fill = 0
    elif mode == "random":
        fill = None # Handled inside loop
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if patch_indices:
        for (py, px) in patch_indices:
            y0, x0 = py * patch, px * patch
            y1, x1 = min(y0 + patch, H), min(x0 + patch, W)
            
            if mode == "random":
                if out.dtype == np.uint8:
                    noise = np.random.randint(0, 256, size=(y1-y0, x1-x0, C), dtype=np.uint8)
                else:
                    noise = np.random.rand(y1-y0, x1-x0, C).astype(out.dtype)
                out[y0:y1, x0:x1, :] = noise
            else:
                out[y0:y1, x0:x1, :] = fill
    return out

def draw_high_res_occlusion(img_high_res_bgr, patch_indices, agent_h=84, agent_w=84, patch_size=8, mode="gray"):
    """
    Draws the occlusion boxes on the High-Res image.
    """
    H_hr, W_hr = img_high_res_bgr.shape[:2]
    out = img_high_res_bgr.copy()
    
    scale_y = H_hr / agent_h
    scale_x = W_hr / agent_w

    # Define color for solid modes
    fill_color = None
    if mode == "gray":
        fill_color = (127, 127, 127)
    elif mode == "zero":
        fill_color = (0, 0, 0) # Note: Black blocks might be invisible on black space background
    elif mode == "mean":
        # Calculate mean color of the frame (B, G, R)
        mean_bgr = img_high_res_bgr.mean(axis=(0, 1)).astype(int)
        fill_color = tuple(int(x) for x in mean_bgr)
    
    for (py, px) in patch_indices:
        # Convert agent grid coords to pixel coords
        y0 = int(py * patch_size * scale_y)
        x0 = int(px * patch_size * scale_x)
        y1 = int(min((py + 1) * patch_size, agent_h) * scale_y)
        x1 = int(min((px + 1) * patch_size, agent_w) * scale_x)
        
        if mode == "random":
            noise = np.random.randint(0, 256, (y1-y0, x1-x0, 3), dtype=np.uint8)
            out[y0:y1, x0:x1] = noise
        elif fill_color is not None:
            cv2.rectangle(out, (x0, y0), (x1, y1), fill_color, -1)
            
    return out

def save_csv(results: List[EpisodeResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "condition", "return_sum", "length"])
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"Saved results to {path}")


def save_summary_csv(summary, path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = set()
    for s in summary.values():
        keys |= set(s.keys())
    fieldnames = ["condition"] + sorted(keys)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cond, s in summary.items():
            row = {"condition": cond, **s}
            w.writerow(row)
    print(f"Saved summary to {path}")

def plot_summary(summary, outpath, title) -> None:
    if plt is None:
        print("[robustness] matplotlib not available, skipping plot.")
        return
    conds = list(summary.keys())
    means = [summary[c]["mean_return"] for c in conds]
    stds = [summary[c]["std_return"] for c in conds]

    plt.figure()
    plt.bar(conds, means, yerr=stds)
    plt.ylabel("Return (mean ± std)")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved summary plot to {outpath}")

@dataclass
class EpisodeResult:
    episode: int
    condition: str 
    return_sum: float
    length: int

def run_eval(env, policy_fn, sarfa_heatmap_fn, episodes, occ, condition, seed_offset=0, save_dir=None, save_video=False):
    results = []
    rng = np.random.default_rng((occ.seed if occ else 0) + seed_offset)
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    condition_str = f"sarfa_{condition}" if sarfa_heatmap_fn is not None else condition

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        
        raw_rgb = _extract_ale_rgb(env)
        if raw_rgb is None: raw_rgb = obs
        
        H_v, W_v = raw_rgb.shape[:2]
        out_shape = (W_v * 2, H_v * 2)
        
        video_writer = None
        if save_video and ep == 0 and save_dir:
            v_path = os.path.join(save_dir, f"video_{condition_str}_ep0.mp4")
            video_writer = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, out_shape)
           
            print(f"[{condition_str}] Recording video to {v_path}...")

        done = False
        total = 0.0
        t = 0

        while not done:
            # 1. Prepare Agent Data
            obs_hwc, tag = to_hwc(obs) # Agent view (84, 84, C)
            obs_used = obs
            
            # 2. Determine Occlusion patches (on 84x84 grid)
            patches_to_hide = []
            
            if condition in ("mean", "zero", "gray", "random") and sarfa_heatmap_fn is None:
                patches_to_hide = random_patch_indices(obs_hwc.shape[0], obs_hwc.shape[1], occ.patch, occ.k, rng)

            elif condition in ("mean", "zero", "gray", "random") and sarfa_heatmap_fn is not None:
                a_clean = policy_fn(obs)
                heat = sarfa_heatmap_fn(obs, a_clean) 
                heat_hwc, _ = to_hwc(heat)
                heat_hw = heat_hwc[:, :, 0]
                patches_to_hide = topk_patch_indices_from_heat(heat_hw, occ.patch, occ.k)

            # 3. Apply Occlusion to Agent's Observation
            if patches_to_hide:
                occ_obs_hwc = occlude_patches_hwc(obs_hwc, occ.patch, patches_to_hide, occ.mode)
                obs_used = from_hwc(occ_obs_hwc, tag)
            
            # 4. Visualization
            if (video_writer is not None and ep == 0) or (ep == 0 and t == 50 and condition in ("mean", "zero", "gray", "random")):
                vis_base_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
                
                vis_overlaid = vis_base_bgr
                if patches_to_hide:
                    vis_final = draw_high_res_occlusion(vis_overlaid, patches_to_hide, 
                                                      agent_h=obs_hwc.shape[0], 
                                                      agent_w=obs_hwc.shape[1], 
                                                      patch_size=occ.patch, 
                                                      mode=occ.mode if condition != "clean" else "gray")
                else:
                    vis_final = vis_overlaid

                frame_out = cv2.resize(vis_final, out_shape, interpolation=cv2.INTER_NEAREST)
                
                if (video_writer is not None and ep == 0):
                    video_writer.write(frame_out)

                if t == 50 and condition in ("mean", "zero", "gray", "random"):
                    if sarfa_heatmap_fn is not None:
                        s_path = os.path.join(save_dir, f"snapshot_{condition_str}_ep{ep}_step{t}.png")
                    else:
                        s_path = os.path.join(save_dir, f"snapshot_{condition_str}_ep{ep}_step{t}.png")
                    cv2.imwrite(s_path, frame_out)
                    print(f"[{condition_str}] Saved snapshot: {s_path}")

            # Step Agent
            a = policy_fn(obs_used)
            obs, r, term, trunc, _ = env.step(a)
            raw_rgb = _extract_ale_rgb(env) 
            done = term or trunc
            total += float(r)
            t += 1

        if video_writer is not None:
            video_writer.release()
            print(f"[{condition_str}] Video saved.")

        results.append(EpisodeResult(ep, condition_str, total, t))

    return results

def build_policy(algo, env_id, ckpt, device, cfg):
    dev = torch.device(device)
    print(f"Building policy for {algo}...")
    tmp_env = make_env_eval(env_id=env_id, seed=0, frame_skip=4, render_mode=None)
    obs_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    if algo == "ddqn":
        from ddqn import DDQN_Agent
        agent = DDQN_Agent(env=env_id, n_channels=cfg['ddqn']['n_channels'], obs_shape=obs_shape, n_actions=n_actions, device=dev) 
        agent.load(ckpt)
        return lambda obs: agent.act(obs, eps=0.0), "DDQN", agent
    
    elif algo == "ppo":
        from ppo import PPO_Agent
        agent = PPO_Agent(obs_shape=obs_shape, n_actions=n_actions, device=dev)
        agent.load(ckpt)
        def policy_fn(obs):
            if obs.ndim == 3: obs = np.expand_dims(obs, 0)
            return int(agent.act(obs)[0][0])
        return policy_fn, "PPO", agent
    
    elif algo == "sac":
        from sac import SACDiscrete_Agent
        agent = SACDiscrete_Agent(obs_shape=obs_shape, n_actions=n_actions, device=dev)
        agent.load(ckpt)
        return lambda obs: agent.act(obs, deterministic=True), "SAC", agent
    
    raise ValueError(f"Unknown algo: {algo}")

def build_sarfa_heatmap_fn(agent, algo_name):
    if agent is None: return None

    if algo_name == "DDQN":
        return lambda obs, a=None: sarfa.sarfa_heatmap_DDQN(agent, obs)[0]
    elif algo_name == "PPO":
        return lambda obs, a=None: sarfa.sarfa_heatmap_PPO(agent, obs)[0]
    elif algo_name == "SAC":
        return lambda obs: sarfa.sarfa_heatmap_SAC(agent, obs)[0]
    
    return None

def main():
    cfg = load_config("config.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", "-env_id", type=str, default="ALE/SpaceInvaders-v5")
    ap.add_argument("--algo", "-algo", type=str, required=True, choices=["ddqn", "ppo", "sac"])
    ap.add_argument("--ckpt", "-ckpt", type=str, default=None) 
    ap.add_argument("--episodes", "-episodes", type=int, default=1)
    ap.add_argument("--video", "-video", action="store_true")
    ap.add_argument("--patch", "-patch", type=int, default=8)
    ap.add_argument("--k", "-k", type=int, default=10)
    ap.add_argument("--mode", "-mode", type=str, default="zero", choices=["zero", "gray", "mean", "random", "all"])
    ap.add_argument("--seed", "-seed", type=int, default=0)
    ap.add_argument("--frame_skip", "-frame_skip", type=int, default=4)
    ap.add_argument("--device", "-device", type=str, default="cuda")
    ap.add_argument("--run_sarfa", "-run_sarfa", action="store_true")

    args = ap.parse_args()

    env = make_env_eval(env_id=args.env_id, seed=args.seed, frame_skip=args.frame_skip, render_mode=None)
    outdir = os.path.join(cfg["robustness"]["outdir"], args.algo)
    os.makedirs(outdir, exist_ok=True)  

    if args.ckpt is None:
        ckpt = cfg[args.algo]["path_best_model"]
    else:
        ckpt = args.ckpt
    
    policy_fn, algo_name, agent = build_policy(args.algo, args.env_id, ckpt, args.device, cfg)

    results: List[EpisodeResult] = []

    results += run_eval(env, policy_fn, sarfa_heatmap_fn=None, episodes=args.episodes, occ=None, 
                        condition="clean", seed_offset=args.seed, save_dir=outdir, save_video=args.video)
    
    if args.mode == "all":
        modes_to_run = ["gray", "mean", "zero", "random"]
    else:
        modes_to_run = [args.mode]
    
    sem = 0

    for mode in modes_to_run:
        
        print(f"Running occlusion mode: {mode}...")
        occ = OcclusionConfig(patch=args.patch, k=args.k, mode=mode, seed=args.seed)
        results += run_eval(env, policy_fn, sarfa_heatmap_fn=None, episodes=args.episodes, occ=occ, 
                            condition=mode, seed_offset=args.seed + 10_000, save_dir=outdir, save_video=args.video)
        
        if sem == 0 and args.run_sarfa:
            print(f"Running SARFA occlusion for mode: {mode}...")
            sarfa_heatmap_fn = build_sarfa_heatmap_fn(agent=agent, algo_name=algo_name)
            results += run_eval(env, policy_fn, sarfa_heatmap_fn=sarfa_heatmap_fn, episodes=args.episodes, occ=occ, 
                                condition=mode, seed_offset=args.seed + 20_000, save_dir=outdir, save_video=args.video)
            sem= 1
    
    env.close()

    summ = summarize(results)

    if args.run_sarfa:
        results_path = os.path.join(outdir, f"results_{args.mode}_sarfa.csv")
        summary_path = os.path.join(outdir, f"summary_{args.mode}_sarfa.csv")
        plot_path = os.path.join(outdir, f"summary_{args.mode}_sarfa.png")
    else:
        results_path = os.path.join(outdir, f"results_{args.mode}.csv")
        summary_path = os.path.join(outdir, f"summary_{args.mode}.csv")
        plot_path = os.path.join(outdir, f"summary_{args.mode}.png")
    
    save_csv(results, results_path)
    save_summary_csv(summ, summary_path)
    plot_summary(summ, plot_path, 
                title=f"{algo_name} robustness ({args.env_id}) p={args.patch}, k={args.k}, ep={args.episodes}")
    
    print(f"Done. Saved to {outdir}")
    print(summ)

    

if __name__ == "__main__":
    main()