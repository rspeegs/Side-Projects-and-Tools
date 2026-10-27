from PIL import Image
import numpy as np
from blur_noborder import blur_noborder

def blur_region(image, box, radius):
    """Box = (x0, y0, x1, y1) in pixel coords"""

    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)

    cx0, cy0 = max(x0 - radius, 0), max(y0 - radius, 0)
    cx1, cy1 = min(x1 + radius, w), min(y1 + radius, h)

    blurred = blur_noborder(image[cy0:cy1, cx0:cx1], radius)

    out = image.copy()
    out[y0:y1, x0:x1] = np.rint(blurred[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]).astype(image.dtype)
    return out

