import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from blur_region import blur_region

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

_landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options = mp_python.BaseOptions(
        model_asset_path = "face_landmarker.task"), 
        num_faces = 10))

def find_eye_boxes(image, margin = 0.5):
    """Return a list of (x0, y0, x1, y1) boxes, one per detected eye"""
    h, w = image.shape[:2]
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = np.ascontiguousarray(image))
    result = _landmarker.detect(mp_image)

    boxes = []
    for face in result.face_landmarks:
        for eye in (LEFT_EYE, RIGHT_EYE):
            xs = np.array([face[i].x for i in eye]) * w
            ys = np.array([face[i].y for i in eye]) * h
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

            mx, my = (x1 - x0) * margin, (y1 - y0) * margin * 2
            boxes.append((int(x0 - mx), int(y0 - my), int(x1 + mx), int(y1 + my)))
    return boxes

def blur_eyes(path_in, path_out, radius = None):
    image = np.array(ImageOps.exif_transpose(Image.open(path_in)).convert("RGB"))

    for box in find_eye_boxes(image):
        r = radius or max(5, (box[0]) // 3)
        image = blur_region(image, box, r)

    Image.fromarray(image).save(path_out)

blur_eyes("Ian_Hawke_(DavidCross).jpg", "blurred_Ian_Hawke.jpg")