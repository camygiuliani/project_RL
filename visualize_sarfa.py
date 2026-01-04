import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmap(gray_frame, heatmap, alpha=0.5):
    # gray_frame: (84,84) or (84,84,1)
    if gray_frame.ndim == 3:
        gray_frame = gray_frame[..., 0]
    plt.figure()
    plt.imshow(gray_frame, cmap="gray")
    plt.imshow(heatmap, alpha=alpha)
    plt.axis("off")
    plt.show()
