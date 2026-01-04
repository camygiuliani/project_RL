import numpy as np
import torch
import torch.nn.functional as F
import ale_py


def occlude(obs, x0, y0, w, h, fill=0):
    # obs: (84,84,4) uint8
    oc = obs.copy()
    oc[y0:y0+h, x0:x0+w, :] = fill
    return oc

@torch.no_grad()
def q_values(agent, obs_uint8):
    x = torch.from_numpy(obs_uint8).to(agent.device)
    x = x.permute(2,0,1).unsqueeze(0).float() / 255.0
    q = agent.q(x).squeeze(0)  # (A,)
    return q.detach().cpu().numpy()

def sarfa_heatmap(agent, obs_uint8, action=None, patch=8, stride=8):
    """
    Skeleton SARFA-like: measures how occluding patches changes Q(action).
    You can extend it to full SARFA (specificity+relevance).
    Returns: heatmap (H,W) float.
    """
    base_q = q_values(agent, obs_uint8)
    if action is None:
        action = int(np.argmax(base_q))

    H, W, _ = obs_uint8.shape
    heat = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            oc = occlude(obs_uint8, x, y, patch, patch, fill=0)
            q_oc = q_values(agent, oc)

            # simple attribution: drop in chosen-action value
            score = base_q[action] - q_oc[action]
            heat[y:y+patch, x:x+patch] += score

    # normalize for visualization
    heat = heat - heat.min()
    if heat.max() > 1e-8:
        heat = heat / heat.max()
    return heat, action

def main():
    import os
    import argparse
    import matplotlib.pyplot as plt

    from wrappers import make_env
    from networks import DQNCNN
    from dqn_agent import DQNAgent

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/dqn_step_200000.pt")
    parser.add_argument("--env", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(env_id=args.env, seed=args.seed)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (84,84,4)

    q = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    tgt = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    agent = DQNAgent(q, tgt, n_actions, device, double_dqn=True)
    agent.load(args.model)

    obs, _ = env.reset(seed=args.seed)

    heat, action = sarfa_heatmap(
        agent, obs, action=None, patch=args.patch, stride=args.stride
    )

    # Salva heatmap
    npy_path = os.path.join(args.outdir, "sarfa_heatmap.npy")
    np.save(npy_path, heat)

    # Salva immagine overlay (usa il primo canale del frame stack)
    gray = obs[..., 0]  # (84,84)
    plt.figure()
    plt.imshow(gray, cmap="gray")
    plt.imshow(heat, alpha=0.5)
    plt.title(f"SARFA-like heatmap (action={action})")
    plt.axis("off")
    png_path = os.path.join(args.outdir, "sarfa_overlay.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    env.close()
    print(f"[SARFA] Saved: {npy_path}")
    print(f"[SARFA] Saved: {png_path}")

if __name__ == "__main__":
    main()
