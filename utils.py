import cv2
import numpy as np

def get_pixel_to_m(intrinsics, extrinsics, depth, pixel):
    point = np.linalg.inv(intrinsics) @ pixel
    point *= depth
    point = np.linalg.inv(extrinsics) @ point

    actual_location = point[:2]

    pixel = pixel - intrinsics[:, 2] # account for camera center
    pixel_location = pixel[:2]

    return actual_location / pixel_location