import cv2
import numpy as np

def calc_intrinsics():
    intrinsic_matrix = np.identity(3)
    return intrinsic_matrix