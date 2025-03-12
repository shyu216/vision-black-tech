import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.insert(0, image)
    return gaussian_pyramid

def build_laplacian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    laplacian_pyramid = []
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    
    return laplacian_pyramid

def process_pyramid(pyr, lowpass1, lowpass2, r1, r2, alpha, lambda_c, nlevels, image):
    for i in range(nlevels):
        cv2.addWeighted(lowpass1[i], 1 - r1, pyr[i], r1, 0, dst=lowpass1[i])
        cv2.addWeighted(lowpass2[i], 1 - r2, pyr[i], r2, 0, dst=lowpass2[i])
    
    filtered = [lp1 - lp2 for lp1, lp2 in zip(lowpass1, lowpass2)]

    delta = lambda_c / 8 / (1 + alpha)
    exaggeration_factor = 2
    lambda_ = (image.shape[0]**2 + image.shape[1]**2)**0.5 / 3

    for l in range(nlevels):
        if l == 0 or l == nlevels - 1:
            filtered[l].fill(0)
            continue
        
        currAlpha = (lambda_ / delta / 8 - 1) * exaggeration_factor
        if currAlpha > alpha:
            filtered[l] *= alpha
        else:
            filtered[l] *= currAlpha
        
        lambda_ /= 2

    upsampled = filtered[0].copy()
    for l in range(1, nlevels):
        upsampled = cv2.pyrUp(upsampled, dstsize=(filtered[l].shape[1], filtered[l].shape[0]))
        upsampled += filtered[l]
    
    amplified_image = image + upsampled
    return amplified_image

alpha = 10
lambda_c = 16
r1 = 0.5
r2 = 0.05
chromAttenuation = 0.1
nlevels = 4

pipeline = rs.pipeline()
config = rs.config()
# bug: only enable color stream will fail to get frames
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)


frames = pipeline.wait_for_frames(timeout_ms=5000)
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()
color_image = np.asanyarray(color_frame.get_data())
if color_image is None:
    print("无法接收帧 (stream end?). Exiting ...")
    pipeline.stop()
    exit()

roi = cv2.selectROI('Select ROI', color_image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select ROI')

frame_ycrcb = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
frame_y = frame_ycrcb[:, :, 0]
pyr = build_laplacian_pyramid(frame_y, levels=nlevels)
laplowpass1 = copy.deepcopy(pyr)
laplowpass2 = copy.deepcopy(pyr)
pyr = build_gaussian_pyramid(frame_y, levels=nlevels)
gaulowpass1 = copy.deepcopy(pyr)
gaulowpass2 = copy.deepcopy(pyr)

fig, ax = plt.subplots()
intensity_queues = [deque(maxlen=100) for _ in range(3)]
labels = ['Origin', 'Lap Pyr Amplified', 'Gau Pyr Amplified']
lines = [ax.plot([], [], lw=2, label=f'{label}')[0] for label in labels]
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.set_xlabel('Frame')
ax.set_ylabel('Average intensity')
ax.legend()

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame:
        return lines

    color_image = np.asanyarray(color_frame.get_data())

    frame_ycrcb = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    frame_y = frame_ycrcb[:, :, 0]

    roi_intensity = np.mean(frame_y[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    intensity_queues[0].append(roi_intensity)

    lines[0].set_data(range(len(intensity_queues[0])), intensity_queues[0])

    cv2.putText(color_image, f"ROI Average Intensity: {roi_intensity:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(color_image, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('RGB Image', color_image)

    start_time = time.time()

    pyr = build_laplacian_pyramid(frame_y, levels=nlevels)
    amplified_y = process_pyramid(pyr, laplowpass1, laplowpass2, r1, r2, alpha, lambda_c, nlevels, frame_y)
    frame_ycrcb[:, :, 0] = amplified_y
    amplified_color_image = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
    roi_intensity_lap = np.mean(amplified_y[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    intensity_queues[1].append(roi_intensity_lap)

    cv2.putText(amplified_color_image, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(amplified_color_image, f"ROI Average Intensity: {roi_intensity_lap:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(amplified_color_image, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('Laplacian Pyramid Amplified Image', amplified_color_image)

    start_time = time.time()

    pyr = build_gaussian_pyramid(frame_y, levels=nlevels)
    amplified_y = process_pyramid(pyr, gaulowpass1, gaulowpass2, r1, r2, alpha, lambda_c, nlevels, frame_y)
    frame_ycrcb[:, :, 0] = amplified_y
    amplified_color_image = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
    roi_intensity_gau = np.mean(amplified_y[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    intensity_queues[2].append(roi_intensity_gau)

    cv2.putText(amplified_color_image, f"Delay: {(time.time() - start_time)*1000:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(amplified_color_image, f"ROI Average Intensity: {roi_intensity_gau:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(amplified_color_image, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('Gaussian Pyramid Amplified Image', amplified_color_image)

    lines[0].set_data(range(len(intensity_queues[0])), intensity_queues[0])
    lines[1].set_data(range(len(intensity_queues[1])), intensity_queues[1])
    lines[2].set_data(range(len(intensity_queues[2])), intensity_queues[2])
    ax.set_xlim(0, max(len(intensity_queues[0]), len(intensity_queues[1]), len(intensity_queues[2])))
    ax.set_ylim(0, max(max(intensity_queues[0]), max(intensity_queues[1]), max(intensity_queues[2])) + 1)

    if cv2.waitKey(1) == ord('q'):
        plt.close(fig)
        pipeline.stop()
        cv2.destroyAllWindows()
        return lines

    return lines

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)
plt.show()
