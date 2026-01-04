import numpy as np
import torch
import torch.nn.functional as F

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
