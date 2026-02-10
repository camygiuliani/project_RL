from __future__ import annotations
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm, trange
from utils import load_config
from wrappers import make_env, make_env_eval
import torch
import os
import csv
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

# matplotlib optional
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from wrappers import make_env_eval  # your wrappers.py :contentReference[oaicite:2]{index=2}


# =========================
# Obs / heatmap utilities
# =========================

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

    if mode == "zero":
        fill = 0
    elif mode == "gray":
        fill = 127 if out.dtype == np.uint8 else 0.5
    elif mode == "mean":
        fill = out.mean(axis=(0, 1), keepdims=True)
    elif mode == "random":
        fill = None
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
        elif mode == "mean":
            out[y0:y1, x0:x1, :] = fill
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


def run_eval(
    env,
    policy_fn: Callable[[np.ndarray], int],
    sarfa_heatmap_fn: Optional[Callable[[np.ndarray, int], np.ndarray]],
    episodes: int,
    occ: Optional[OcclusionConfig],
    condition: str,
    seed_offset: int = 0,
) -> List[EpisodeResult]:
    """
    condition:
      - "clean": no occlusion
      - "random": occlude K random patches
      - "sarfa": occlude top-K SARFA patches
    """
    results: List[EpisodeResult] = []
    rng = np.random.default_rng((occ.seed if occ else 0) + seed_offset)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        total = 0.0
        t = 0

        while not done:
            obs_hwc, tag = to_hwc(obs)

            if condition == "clean" or occ is None:
                obs_used = obs

            elif condition == "random":
                idx = random_patch_indices(obs_hwc.shape[0], obs_hwc.shape[1], occ.patch, occ.k, rng)
                occ_obs_hwc = occlude_patches_hwc(obs_hwc, occ.patch, idx, occ.mode)
                obs_used = from_hwc(occ_obs_hwc, tag)

            elif condition == "sarfa":
                if sarfa_heatmap_fn is None:
                    raise ValueError("sarfa_heatmap_fn is required for condition='sarfa'")

                # Fair protocol: choose action from CLEAN obs, then occlude, then act on occluded obs
                a_clean = policy_fn(obs)

                heat = sarfa_heatmap_fn(obs, a_clean)
                heat_hwc, _ = to_hwc(heat)
                heat_hw = heat_hwc[:, :, 0]

                idx = topk_patch_indices_from_heat(heat_hw, occ.patch, occ.k)
                occ_obs_hwc = occlude_patches_hwc(obs_hwc, occ.patch, idx, occ.mode)
                obs_used = from_hwc(occ_obs_hwc, tag)

            else:
                raise ValueError(f"Unknown condition: {condition}")

            a = policy_fn(obs_used)
            obs, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)

            total += float(r)
            t += 1

        results.append(EpisodeResult(ep, condition, total, t))

    return results


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


def build_policy(algo: str, env_id: str, ckpt: str, device: str):
    """
    Returns policy_fn(obs)->action and a short display name.
    Uses your agents:
      - ddqn.DDQN_Agent.act(obs, eps=0.0), load(path)
      - ppo.PPO_Agent.act(obs_batch)-> (actions, logps, values), load(path)
      - sac.SACDiscrete_Agent.act(obs, deterministic=True), load(path)
    """

    dev = torch.device(device)
    obs_shape = (84, 84, 4)  # your wrappers output :contentReference[oaicite:3]{index=3}

    # Create a temp env just to read action_space.n
    tmp_env = make_env_eval(env_id=env_id, seed=0, frame_skip=4, render_mode=None)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    if algo == "ddqn":
        from ddqn import DDQN_Agent  # :contentReference[oaicite:4]{index=4}
        agent = DDQN_Agent(env=env_id, n_channels=4, obs_shape=obs_shape, n_actions=n_actions, device=dev)
        agent.load(ckpt)

        def policy_fn(obs: np.ndarray) -> int:
            return agent.act(obs, eps=0.0)

        return policy_fn, "DDQN"

    if algo == "ppo":
        from ppo import PPO_Agent  # :contentReference[oaicite:5]{index=5}
        agent = PPO_Agent(obs_shape=obs_shape, n_actions=n_actions, env_id=env_id, device=dev)
        agent.load(ckpt)

        def policy_fn(obs: np.ndarray) -> int:
            obs_b = np.expand_dims(obs, axis=0)
            actions, _, _ = agent.act(obs_b)
            return int(actions[0])

        return policy_fn, "PPO"

    if algo == "sac":
        from sac import SACDiscrete_Agent  # :contentReference[oaicite:6]{index=6}
        agent = SACDiscrete_Agent(obs_shape=obs_shape, n_actions=n_actions, device=dev, env_id=env_id)
        agent.load(ckpt)

        def policy_fn(obs: np.ndarray) -> int:
            return agent.act(obs, deterministic=True)

        return policy_fn, "SAC"

    raise ValueError(f"Unknown algo: {algo}. Choose from: ddqn, ppo, sac")


# =========================
# SARFA hook (you connect it)
# =========================

def build_sarfa_heatmap_fn():
    """
    TODO: connect to your sarfa.py.

    Expected signature:
        heat = sarfa_heatmap_fn(obs_uint8, action_int)  -> np.ndarray (84,84) or (84,84,1)

    Minimal adapter example (YOU will replace this):
        from sarfa import your_function
        def sarfa_heatmap_fn(obs, action):
            heat, _ = your_function(..., obs=obs, action=action, ...)
            return heat
        return sarfa_heatmap_fn
    """
    # Placeholder to avoid breaking clean/random runs:
    return None


# =========================
# Main
# =========================

def main():
    # load config from config.yaml
    cfg = load_config("config.yaml")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="ALE/SpaceInvaders-v5")
    ap.add_argument("--algo", type=str, required=True, choices=["ddqn", "ppo", "sac"])
    ap.add_argument("--ckpt", type=str, default=cfg["ddqn"]["path_best_model"])
    ap.add_argument("--episodes", type=int, default=30)

    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", type=str, default="zero", choices=["zero", "gray", "mean", "random"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--frame_skip", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--run_sarfa", action="store_true", help="Also run SARFA-guided occlusion (requires sarfa hook).")
    args = ap.parse_args()

    # env
    env = make_env_eval(env_id=args.env_id, seed=args.seed, frame_skip=args.frame_skip, render_mode=None)  # :contentReference[oaicite:7]{index=7}

    # policy
    policy_fn, algo_name = build_policy(args.algo, args.env_id, args.ckpt, args.device)

    # occlusion config
    occ = OcclusionConfig(patch=args.patch, k=args.k, mode=args.mode, seed=args.seed)

    # sarfa hook
    sarfa_heatmap_fn = build_sarfa_heatmap_fn()

    # run conditions
    results: List[EpisodeResult] = []
    results += run_eval(env, policy_fn, sarfa_heatmap_fn, args.episodes, None, "clean", seed_offset=args.seed)
    results += run_eval(env, policy_fn, sarfa_heatmap_fn, args.episodes, occ, "random", seed_offset=args.seed + 10_000)

    if args.run_sarfa:
        if sarfa_heatmap_fn is None:
            raise RuntimeError("You set --run_sarfa but sarfa_heatmap_fn is None. Connect sarfa.py in build_sarfa_heatmap_fn().")
        results += run_eval(env, policy_fn, sarfa_heatmap_fn, args.episodes, occ, "sarfa", seed_offset=args.seed + 20_000)

    env.close()

    summ = add_deltas(summarize(results), base="clean")

    # save
    outdir = os.path.join(cfg["robustness"]["outdir"], args.algo)
    os.makedirs(outdir, exist_ok=True)

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