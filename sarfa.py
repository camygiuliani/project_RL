import numpy as np
import torch
import torch.nn.functional as F


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
def sarfa_heatmap(
    agent,
    obs_uint8,
    action=None,
    patch=8,
    stride=8,
    fill_mode="mean",
    batch_size=64,
    use_advantage=True,
    clamp_positive=True,
    use_action_flip=True,
    flip_weight=2.0
):
    """
    SARFA-like improved:
      - occlusion with fill_mode (mean instead of black)
      - score on Advantage (more action-specific)
      - batching for speed

    Returns: heatmap (H,W) in [0,1], action chosen
    """
    H, W, C = obs_uint8.shape
    assert C == 4, "(84,84,4)"

    # base Q
    # Q at the original observation
    base_q = q_values_batch(agent, obs_uint8[None, ...])[0]  # (A,)
    #best action at the original observation
    base_best = int(np.argmax(base_q))

    if action is None:
        # if action not provided, use the best action chosen by the agent
        action = int(np.argmax(base_q))

    if use_advantage:
        #how much better is the chosen action compared to others
        base_adv = base_q - base_q.max()  # (A,)
        # reference score for the chosen action
        base_score_ref = base_adv[action]
    else:
        #we use Q values directly
        base_score_ref = base_q[action]

    # building patch grid
    coords = [(x, y) for y in range(0, H - patch + 1, stride)
                    for x in range(0, W - patch + 1, stride)]

    #initialize heatmap and fill value
    heat = np.zeros((H, W), dtype=np.float32)
    fill = _get_fill_value(obs_uint8, mode=fill_mode)

    # in the loop we process chunks of  batches
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i+batch_size]
        n = len(chunk)

        # creating n copies of the observation
        batch = np.repeat(obs_uint8[None, ...], n, axis=0).copy()

        #in j-th copy, occlude the patch at chunk[j]
        for j, (x, y) in enumerate(chunk):
            batch[j, y:y+patch, x:x+patch, :] = fill

        #computes Q for all occluded observations in the batch
        q_oc = q_values_batch(agent, batch)  # (n,A)

        if use_action_flip:
            #best action for each occluded observation
            oc_best = np.argmax(q_oc, axis=1)              # (n,)
            #if best action changed (1), we have a flip. Otherwise (0)
            flips = (oc_best != base_best).astype(np.float32)  # (n,)
        else:
            #if not action flip, just zeros
            flips = np.zeros((q_oc.shape[0],), dtype=np.float32)


        if use_advantage:
            #compute advantage for evey occluded observation
            adv_oc = q_oc - q_oc.max(axis=1, keepdims=True)  # (n,A)
            #how much the advantage for the chosen action changed when occluding
            scores = base_score_ref - adv_oc[:, action]      # (n,)
        else:
            #using Q values directly
            scores = base_score_ref - q_oc[:, action]

        if use_action_flip:
            #increase score if we had an action flip
            scores = scores * (1.0 + flip_weight * flips)


        # if occlusion helps, ignore it
        if clamp_positive:
            scores = np.maximum(scores, 0.0)

        # add score on every pixel in the patch
        for (x, y), s in zip(chunk, scores):
            heat[y:y+patch, x:x+patch] += float(s)

    # normalization to [0,1] for visualization
    heat = heat - heat.min()
    mx = heat.max()
    if mx > 1e-8:
        heat = heat / mx

    return heat, action


def main():
    import os
    import argparse
    import matplotlib.pyplot as plt

    from wrappers import make_env
    from dqn_agent import DQN_Agent,DQNCNN

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

    # we create 2 networks with the same architecture
    q = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    tgt = DQNCNN(in_channels=obs_shape[2], n_actions=n_actions)
    #now we create the agent and load the model
    agent = DQN_Agent(q, tgt, n_actions, device, double_dqn=True)
    agent.load(args.model)

    #take an observation to analyze
    obs, _ = env.reset(seed=args.seed)

    heat, action = sarfa_heatmap(agent, obs, patch=args.patch, stride=args.stride,
                            fill_mode="mean", batch_size=64,
                            use_advantage=True, clamp_positive=True,
                            use_action_flip=True, flip_weight=2.0)


    # saving results in files
    npy_path = os.path.join(args.outdir, "sarfa_heatmap.npy")
    #np.save(npy_path, heat)

    # overlay heatmap on grayscale image
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
