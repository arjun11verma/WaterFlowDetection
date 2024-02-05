import cv2
import numpy as np
from collections import deque
import utils
import os

def calc_sma(input, ma, window_size=5):
    if not hasattr(calc_sma, "counter"):
        calc_sma.buffer = deque(maxlen=window_size)

    calc_sma.buffer.append(input)

    if (calc_sma.counter == window_size):
        calc_sma.counter = 0

    return (input / window_size) + ma - (calc_sma.buffer[0] / window_size)

def calc_ema(input, ma, factor=2, window_size=5):
    return input * (factor / (window_size + 1)) + ma * (1 - (factor / (window_size + 1)))

def calc_flow(prev_image, next_image):
    """Accepts two BGR images of water and calculates the net flow on the x-axis"""
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    x_axis_flow = flow[:, 0]
    net_x_flow = np.sum(x_axis_flow) / x_axis_flow.shape[0]

    return net_x_flow

def analyze_recording(recording_name, camera_intrinsics, depth, filter=lambda x, y: x):
    video = cv2.VideoCapture(os.path.join('C:\GeorgiaTech\Research\GraduateProject\WaterFlowDetection\recordings', recording_name))
    
    ret, prev_image = video.read()

    pixel_conversion = utils.get_pixel_to_m(
        camera_intrinsics, 
        np.identity(4), 
        depth, 
        np.array([prev_image.shape[0], prev_image.shape[1]])
    )
    
    flows = []
    smoothed_flows = []
    smoothed_flow = 0

    while(video.isOpened()):
        ret, image = video.read()

        flow = calc_flow(prev_image, image) * pixel_conversion[0]
        smoothed_flow = filter(flow, smoothed_flow)

        flows.append(flow)
        smoothed_flows.append(smoothed_flow)

        prev_image = image

    flows = np.array(flows)
    smoothed_flows = np.array(smoothed_flows)

def analyze_input():
    pass