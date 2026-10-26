from PIL import Image
import time
import numpy as np

def blur(image, radius):
    h, w = image.shape
    k = 2 * radius + 1

    S = np.zeros((h + 1, w + 1), dtype = np.float64)
    S[1:, 1:] = image.cumsum(axis = 0).cumsum(axis = 1)

    window_sums = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]

    output = np.zeros_like(image)
    output[radius:h - radius, radius:w - radius] = window_sums / (k * k)
    return output

start = time.time()
image = np.array(Image.open("AdventureWorksLogo.png").convert("L"), dtype = np.float64)
result = blur(image, 5)
print(f"It took {time.time() - start} s to blur the image")

import matplotlib.pyplot as plt
plt.imshow(result, cmap = "gray", vmin = 0, vmax = 255)
plt.axis("off")
plt.show()