from PIL import Image
import numpy as np

def blur_noborder(image, radius, mode = "edge"):
    k = 2 * radius + 1
    pad = [(radius, radius), (radius, radius)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image.astype(np.float64), pad, mode = mode)

    S = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1) + padded.shape[2:])
    S[1:, 1:] = padded.cumsum(axis = 0).cumsum(axis = 1)

    sums = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    return sums / (k * k)


image = np.array(Image.open("AdventureWorksLogo.png").convert("L"), dtype = np.float64)
result = blur_noborder(image, radius = 5)

import matplotlib.pyplot as plt
plt.imshow(result, cmap = "grey", vmin = 0, vmax = 255)
plt.axis("off")
plt.show()