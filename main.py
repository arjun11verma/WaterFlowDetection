import flow_calculation, calibration, utils
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def analyze_recording(recording_name, camera_intrinsics, depth, filter=lambda x, y: x, equalize=False, num_frames=100):
    vidpath = os.path.join('recordings', recording_name)
    video = cv2.VideoCapture(vidpath)
    
    ret, prev_image = video.read()
    hsv = np.zeros_like(prev_image)
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    if equalize: prev_image = utils.equalize_img(prev_image)

    pixel_conversion = utils.get_pixel_to_m(
        camera_intrinsics, 
        np.identity(4), 
        depth,
        np.array([prev_image.shape[0], prev_image.shape[1], 1])
    )
    
    flows = []
    smoothed_flows = []
    smoothed_flow = 0

    for i in range(num_frames):
        ret, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if equalize: image = utils.equalize_img(image)

        # 0 = x, 1 = y
        flow = flow_calculation.calc_flow_classical(prev_image, image)

        # visualize flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite('./flow.png', bgr)
        cv2.imwrite('./img.png', image)
        # done

        y_flow_angle = flow_calculation.process_flow_angle(flow, direction=1)
        x_flow_angle = flow_calculation.process_flow_angle(flow, direction=0)
        y_flow_mode = flow_calculation.process_flow_mode(flow, direction=1)
        x_flow_mode = flow_calculation.process_flow_mode(flow, direction=0)

        if (i == 50):
            print(1)

        flows.append(flow)

        prev_image = image

    flows = np.array(flows)
    smoothed_flows = np.array(smoothed_flows)

    plt.plot(smoothed_flows)
    plt.savefig('./flows.png')

if __name__ == '__main__':
    intrinsics = np.array([
        [500, 0, 960],
        [0, 500, 540],
        [0, 0, 1]
    ])
    analyze_recording('WIN_20240301_16_57_32_Pro.mp4', intrinsics, 3, equalize=False, num_frames=1000)