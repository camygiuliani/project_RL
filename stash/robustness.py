from __future__ import annotations
from datetime import datetime
import os
from unittest import case
import numpy as np
from tqdm import tqdm, trange
from utils import load_config
from wrappers import make_env_eval
import torch
import os
import csv
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple
from ddqn import DDQN_Agent
from ppo import PPO_Agent
from sac import SACDiscrete_Agent
import cv2
import sarfa

# matplotlib optional
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from wrappers import make_env_eval  # your wrappers.py :contentReference[oaicite:2]{index=2}


def _extract_ale_rgb(env):
    """
    Returns the true game RGB frame from ALE if possible.
    """
    # Unwrap to base env that has ALE
    e = env
    while hasattr(e, "env"):
        if hasattr(e, "ale"):
            break
        e = e.env

    # Try ALE first (most reliable for Atari details)
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
    
def to_hwc(x: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert observation (or heatmap) to HWC.
    Returns (x_hwc, tag) where tag is used to restore.
    Supports: (H,W), (H,W,C), (C,H,W)
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x[:, :, None], "HW"
    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape={x.shape}")

    # likely HWC
    if x.shape[-1] in (1, 3, 4):
        return x, "HWC"

    # maybe CHW
    if x.shape[0] in (1, 3, 4):
        return np.transpose(x, (1, 2, 0)), "CHW"

    # fallback
    return x, "HWC"


def from_hwc(x_hwc: np.ndarray, tag: str) -> np.ndarray:
    if tag == "CHW":
        return np.transpose(x_hwc, (2, 0, 1))
    if tag == "HW":
        return x_hwc[:, :, 0]
    return x_hwc


def normalize_heatmap(heat_hw: np.ndarray) -> np.ndarray:
    h = np.asarray(heat_hw, dtype=np.float32)
    if h.ndim == 3:
        h = h.mean(axis=-1)
    if h.ndim != 2:
        raise ValueError(f"Heatmap must be 2D (H,W) or 3D reducible, got {h.shape}")
    mn, mx = float(h.min()), float(h.max())
    if mx - mn < 1e-8:
        return np.zeros_like(h, dtype=np.float32)
    return (h - mn) / (mx - mn)


# =========================
# Occlusion mechanics
# =========================

@dataclass
class OcclusionConfig:
    patch: int = 8
    k: int = 10
    mode: str = "zero"   # "zero" | "gray" | "mean" | "random"
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

    scored: List[Tuple[Tuple[int, int], float]] = []
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

    if mode == "mean":
        fill = out.mean(axis=(0, 1), keepdims=True).astype(out.dtype)
    elif mode == "random":
        fill = None
    elif mode == "gray":
        val = 127 if out.dtype == np.uint8 else 0.5
        fill = np.full((1, 1, C), val, dtype=out.dtype)
    elif mode == "zero":
        fill = 0
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")

    for (py, px) in patch_indices:
        y0, x0 = py * patch, px * patch
        y1, x1 = min(y0 + patch, H), min(x0 + patch, W)

        if mode == "random":
            if out.dtype == np.uint8:
                noise = np.random.randint(0, 256, size=(y1 - y0, x1 - x0, C), dtype=np.uint8)
            else:
                noise = np.random.rand(y1 - y0, x1 - x0, C).astype(out.dtype)
            out[y0:y1, x0:x1, :] = noise
        else:
            out[y0:y1, x0:x1, :] = fill

    return out


# =========================
# Evaluation
# =========================

@dataclass
class EpisodeResult:
    episode: int
    condition: str   # clean | random | sarfa
    return_sum: float
    length: int

def save_high_res_vis(clean_obs, occluded_obs, save_path, heatmap=None, scale=5):
    """
    Saves a high-res, colored visualization similar to sarfa.py.
    """
    def prepare_frame(obs):
        # 1. Handle dimensions (H,W) -> (H,W,1)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        
        # 2. Handle Channels (Grayscale -> RGB)
        # We repeat the grayscale channel 3 times so we can draw colored overlays on it
        if obs.shape[-1] == 1:
            obs = np.repeat(obs, 3, axis=-1)
        elif obs.shape[-1] > 3:
            obs = obs[:, :, :3]  # Take first 3 frames if stacked

        # 3. Normalize to uint8 [0, 255]
        if obs.dtype != np.uint8:
            obs = (np.clip(obs, 0, 1) * 255).astype(np.uint8)

        # 4. Upscale (Nearest Neighbor preserves sharp pixels)
        h, w = obs.shape[:2]
        bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Prepare base images
    img_clean = prepare_frame(clean_obs)
    img_occ = prepare_frame(occluded_obs)

    if heatmap is not None:
        # Resize heatmap to match the upscaled image dimensions
        h_high, w_high = img_clean.shape[:2]
        hm_resized = cv2.resize(heatmap, (w_high, h_high), interpolation=cv2.INTER_NEAREST)
        hm_resized = np.clip(hm_resized, 0, 1)

        # Apply the "Turbo" colormap (or cv2.COLORMAP_JET)
        hm_uint8 = (hm_resized * 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_TURBO)

        # Create a mask to only color meaningful regions (optional, keeps background cleaner)
        mask = (hm_resized > 0.15).astype(np.float32)[..., None]
        
        # Blend: Original Image * (1-alpha) + Heatmap * alpha
        # We only blend where the mask is active
        alpha = 0.5
        overlay = img_clean.astype(np.float32) * (1 - mask * alpha) + colored_map.astype(np.float32) * (mask * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Stack: Clean (Gray) | Overlay (Color) | Occluded (Gray+Block)
        combined = np.hstack((img_clean, overlay, img_occ))
    else:
        # Stack: Clean | Occluded
        combined = np.hstack((img_clean, img_occ))

    # Save to disk
    cv2.imwrite(save_path, combined)

def summarize(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    by: Dict[str, List[EpisodeResult]] = {}
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


def add_deltas(summary: Dict[str, Dict[str, float]], base: str = "clean") -> Dict[str, Dict[str, float]]:
    base_mean = summary.get(base, {}).get("mean_return", None)
    if base_mean is None:
        return summary
    for cond, s in summary.items():
        s["delta_return"] = s["mean_return"] - base_mean
        s["delta_return_pct"] = (s["delta_return"] / base_mean * 100.0) if abs(base_mean) > 1e-8 else 0.0
    return summary


def save_csv(results: List[EpisodeResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "condition", "return_sum", "length"])
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def save_summary_csv(summary: Dict[str, Dict[str, float]], path: str) -> None:
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


def plot_summary(summary: Dict[str, Dict[str, float]], outpath: str, title: str) -> None:
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

def run_eval(env, policy_fn: Callable[[np.ndarray], int],sarfa_heatmap_fn: Optional[Callable[[np.ndarray, int], np.ndarray]],
            episodes: int, occ: Optional[OcclusionConfig],condition: str, seed_offset: int = 0, save_dir : Optional[str] = None,
            save_video: bool = False) -> List[EpisodeResult]:
    
    results: List[EpisodeResult] = []
    rng = np.random.default_rng((occ.seed if occ else 0) + seed_offset)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        
        raw_rgb = _extract_ale_rgb(env)
        
        if save_video and ep == 0 and save_dir:
            v_path = os.path.join(save_dir, f"video_{condition}_ep0.mp4")
            # 420x420 is the 5x scaled size (84 * 5)
            # FPS=15 because typical Atari skip is 4 (60Hz / 4 = 15)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(v_path, fourcc, 15.0, (420, 420))
            print(f"[{condition}] Recording video to {v_path}...")

        done = False
        total = 0.0
        t = 0

        while not done:
            obs_hwc, tag = to_hwc(obs)
            
            vis_rgb_84 = cv2.resize(raw_rgb, (obs_hwc.shape[1], obs_hwc.shape[0]), interpolation=cv2.INTER_AREA)

            obs_used = obs
            
            #Identify which patches to occlude
            patches_to_hide = []
            
            if condition == "random":
                patches_to_hide = random_patch_indices(obs_hwc.shape[0], obs_hwc.shape[1], occ.patch, occ.k, rng)
            
            elif condition == "sarfa":
                if sarfa_heatmap_fn is None:
                    raise ValueError("sarfa_heatmap_fn is required")
                
                #Get action & heatmap
                a_clean = policy_fn(obs)
                heat = sarfa_heatmap_fn(obs, a_clean)
                heat_hwc, _ = to_hwc(heat)
                heat_hw = heat_hwc[:, :, 0]
                
                patches_to_hide = topk_patch_indices_from_heat(heat_hw, occ.patch, occ.k)

            #Apply Occlusion to AGENT Observation
            if patches_to_hide:
                occ_obs_hwc = occlude_patches_hwc(obs_hwc, occ.patch, patches_to_hide, occ.mode)
                obs_used = from_hwc(occ_obs_hwc, tag)
            else:
                obs_used = obs

            #Visualizzation
            if video_writer is not None or (ep == 0 and t == 50 and condition in ("random", "sarfa")):
                
                patch_size = occ.patch if occ is not None else 8
                vis_occ_rgb = occlude_patches_hwc(vis_rgb_84, patch_size, patches_to_hide, mode="gray")
                
                frame_big = cv2.resize(vis_occ_rgb, (420, 420), interpolation=cv2.INTER_NEAREST)

                # A. Write to Video
                if video_writer is not None:
                    frame_bgr = cv2.cvtColor(frame_big, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

                if t == 50 and condition in ("random", "sarfa"):
                    fname = os.path.join(save_dir, f"vis_{condition}_ep{ep}_step{t}_color.png")
                    print(f"[{condition}] Saving snapshot to {fname}")
                    heatmap_overlay = None
                    if condition == "sarfa" and 'heat_hw' in locals():
                        heatmap_overlay = normalize_heatmap(heat_hw)
                    
                    save_high_res_vis(vis_rgb_84, vis_occ_rgb, fname, heatmap=heatmap_overlay, scale=5)

            a = policy_fn(obs_used)
            obs, r, terminated, truncated, _ = env.step(a)
            
            raw_rgb = _extract_ale_rgb(env)
            
            done = bool(terminated or truncated)
            total += float(r)
            t += 1

        if video_writer is not None:
            video_writer.release()
            print(f"[{condition}] Video saved.")

        results.append(EpisodeResult(ep, condition, total, t))

    return results


def build_policy(algo: str, env_id: str, ckpt: str, device: str, cfg: Dict):
    dev = torch.device(device)
    
    print(f"Building policy for {algo}...")
    # Create a temp env just to read action_space.n
    tmp_env = make_env_eval(env_id=env_id, seed=0, frame_skip=4, render_mode=None)
    obs_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    if algo == "ddqn":
        from ddqn import DDQN_Agent
        agent = DDQN_Agent(env=env_id, n_channels=cfg['ddqn']['n_channels'], obs_shape=obs_shape, n_actions=n_actions, device=dev) 
        agent.load(ckpt)
        return lambda obs: agent.act(obs, eps=0.0), "DDQN", agent

    if algo == "ppo":
        from ppo import PPO_Agent
        agent = PPO_Agent(obs_shape=obs_shape, n_actions=n_actions, device=dev)
        agent.load(ckpt)
        def policy_fn(obs):
            if obs.ndim == 3: obs = np.expand_dims(obs, 0)
            return int(agent.act(obs)[0][0])
        return policy_fn, "PPO", agent

    if algo == "sac":
        from sac import SACDiscrete_Agent
        agent = SACDiscrete_Agent(obs_shape=obs_shape, n_actions=n_actions, device=dev)
        agent.load(ckpt)
        return lambda obs: agent.act(obs, deterministic=True), "SAC", agent

    raise ValueError(f"Unknown algo: {algo}")



def build_sarfa_heatmap_fn(agent, algo_name):
    print(f"Building SARFA heatmap function for {algo_name}...")
    if agent is None:
        return None

    if algo_name == "DDQN":
        def wrapper(obs, action=None):
            return sarfa.sarfa_heatmap_DDQN(agent, obs)[0]
        return wrapper

    elif algo_name == "PPO":
        def wrapper(obs, action=None):
            return sarfa.sarfa_heatmap_PPO(agent, obs)[0]
        return wrapper

    elif algo_name == "SAC":
        def wrapper(obs):
            return sarfa.sarfa_heatmap_SAC(agent, obs)[0]
        return wrapper

    return None


def main():
    cfg = load_config("config.yaml")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="ALE/SpaceInvaders-v5")
    ap.add_argument("--algo", type=str, required=True, choices=["ddqn", "ppo", "sac"])
    ap.add_argument("--ckpt", type=str, default=None) 
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--video", action="store_true", help="Whether to save a video of the first episode under occlusion.")

    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", type=str, default="zero", choices=["zero", "gray", "mean", "random"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--frame_skip", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--run_sarfa", action="store_true", help="Also run SARFA-guided occlusion.")
    args = ap.parse_args()

    # env
    env = make_env_eval(env_id=args.env_id, seed=args.seed, frame_skip=args.frame_skip, render_mode=None)

    outdir = os.path.join(cfg["robustness"]["outdir"], args.algo)
    os.makedirs(outdir, exist_ok=True)  

    if args.ckpt is not None:
        ckpt = args.ckpt
    else:
        if args.algo == "ddqn":
            ckpt = cfg["ddqn"]["path_best_model"]
        elif args.algo == "ppo":
            ckpt = cfg["ppo"]["path_best_model"]
        elif args.algo == "sac":
            ckpt = cfg["sac"]["path_best_model"]
    
    print(f"Loading {args.algo} from {ckpt}...")
    policy_fn, algo_name, agent = build_policy(args.algo, args.env_id, ckpt, args.device, cfg=cfg)

    # occlusion config
    occ = OcclusionConfig(patch=args.patch, k=args.k, mode=args.mode, seed=args.seed)

    # run conditions
    results: List[EpisodeResult] = []
    results += run_eval(env, policy_fn, sarfa_heatmap_fn=None, episodes=args.episodes, occ=None, 
                        condition="clean", seed_offset=args.seed, save_dir=outdir, save_video=args.video)
    results += run_eval(env, policy_fn, sarfa_heatmap_fn=None, episodes=args.episodes, occ=occ, 
                        condition="random", seed_offset=args.seed + 10_000, save_dir=outdir, save_video=args.video)

    if args.run_sarfa:
        sarfa_heatmap_fn = build_sarfa_heatmap_fn(agent=agent, algo_name=algo_name)
        results += run_eval(env, policy_fn, sarfa_heatmap_fn=sarfa_heatmap_fn, episodes=args.episodes, occ=occ, 
                            condition="sarfa", seed_offset=args.seed + 20_000, save_dir=outdir, save_video=args.video)

    env.close()

    summ = add_deltas(summarize(results), base="clean")

    save_csv(results, os.path.join(outdir, "episodes.csv"))
    save_summary_csv(summ, os.path.join(outdir, "summary.csv"))
    plot_summary(summ, os.path.join(outdir, "summary.png"),
                 title=f"{algo_name} robustness ({args.env_id}) - patch={args.patch}, k={args.k}, mode={args.mode}")

    print("Saved to:", outdir)
    print("Summary:")
    for cond, s in summ.items():
        print(cond, s)


if __name__ == "__main__":
    main()