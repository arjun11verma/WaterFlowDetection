import cv2
import numpy as np
import utils
from collections import deque

from VideoFlow.configs.multiframes_sintel_submission import get_cfg
from VideoFlow.core.Networks.MOFNetStack import network

def calc_sma(input, ma, window_size=5):
    if not hasattr(calc_sma, "counter"):
        calc_sma.buffer = deque(maxlen=window_size)

    calc_sma.buffer.append(input)

    if (calc_sma.counter == window_size):
        calc_sma.counter = 0

    return (input / window_size) + ma - (calc_sma.buffer[0] / window_size)

def calc_ema(input, ma, factor=2, window_size=5):
    return input * (factor / (window_size + 1)) + ma * (1 - (factor / (window_size + 1)))

def calc_flow_classical(prev_image, next_image):
    """Accepts two grayscale images of water and calculates the avg flow in the specified direction"""
    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    return flow

class AI_Flow():
    def __init__(self):
        self.model = None
    
    def calc_flow_ai(self, prev_image, next_image):
        flow = 

def process_flow_angle(flow, direction=0, tolerance=(np.pi / 24)):
    """Take in flow and filter it such that we can calculate an accurate flow measure for the given direction (0 = x, 1 = y)"""
    flow = flow.reshape((flow.shape[0] * flow.shape[1], flow.shape[2]))
    ang = np.arctan2(flow[:, 1], flow[:, 0])
    flow = flow[:, direction]

    valid_ranges = None

    if direction == 0:
        valid_ranges = [(-tolerance, tolerance), (-np.pi + tolerance, np.pi - tolerance)]
    else:
        valid_ranges = [((np.pi / 2) - tolerance, (np.pi / 2) + tolerance), ((-np.pi / 2) - tolerance, (-np.pi / 2) + tolerance)]

    net_flow, size = 0, 0
    for range in valid_ranges:
        low, high = range
        valid_flow = flow[np.logical_and(ang > low, ang < high)]
        net_flow += np.sum(valid_flow)
        size += valid_flow.shape[0]

    return net_flow / size

def process_flow_mode(flow, direction=0, num_bins=19):
    flow = flow.reshape((flow.shape[0] * flow.shape[1], flow.shape[2]))
    ang = np.arctan2(flow[:, 1], flow[:, 0]) * (180 / np.pi)
    flow = flow[:, direction]

    ang = ang.astype(np.int16) # round to nearest degree for all angles
    bins = np.linspace(-180, 180, num_bins)
    hist, bins = np.histogram(ang, bins=bins) # 10 degree intervals

    motion_direction = np.argmax(hist)
    low, high = bins[motion_direction], bins[motion_direction + 1]

    return np.mean(flow[np.logical_and(ang >= low, ang <= high)])



