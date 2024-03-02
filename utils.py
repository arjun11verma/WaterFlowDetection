import cv2
import numpy as np

def get_pixel_to_m(intrinsics, extrinsics, depth, pixel):
    point = np.linalg.inv(intrinsics) @ pixel
    point *= depth
    point = np.append(point, np.ones(1)) # make it 3D homogenous
    point = np.linalg.inv(extrinsics) @ point

    actual_location = point[:2]

    pixel_location = pixel[:2]

    return actual_location / pixel_location

def equalize_img(img):
    return cv2.equalizeHist(img)