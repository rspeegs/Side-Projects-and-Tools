from PIL import Image
import numpy as np
import time
from numba import njit, prange

@njit(cache = True, parallel = True)
def blur(image, radius):
    h, w = image.shape
    output = np.zeros_like(image)
    normaliation = (2 * radius + 1) ** 2
    for i in prange(radius, h - radius):
        for j in range(radius, w - radius):
            total = 0.0
            for ki in range(-radius, radius + 1):
                for kj in range(-radius, radius + 1):
                    total += image[i + ki, j + kj]
            output[i, j] = total / normaliation
    return output

start = time.time()
image = np.array(Image.open("AdventureWorksLogo.png").convert("L"), dtype = np.float64)
result = blur(image, radius = 5)
print(f"It took {time.time() - start} s to blur the image")
 
import matplotlib.pyplot as plt
plt.imshow(result, cmap = "gray", vmin = 0, vmax = 255)
plt.axis("off")
plt.show()