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

def process_pyramid(pyr, lowpass1, lowpass2, r1, r2, alpha, lambda_c, nlevels, depth_image):
    for i in range(nlevels):
        cv2.addWeighted(lowpass1[i], 1 - r1, pyr[i], r1, 0, dst=lowpass1[i])
        cv2.addWeighted(lowpass2[i], 1 - r2, pyr[i], r2, 0, dst=lowpass2[i])
    
    filtered = [lp1 - lp2 for lp1, lp2 in zip(lowpass1, lowpass2)]

    delta = lambda_c / 8 / (1 + alpha)
    exaggeration_factor = 2
    lambda_ = (depth_image.shape[0]**2 + depth_image.shape[1]**2)**0.5 / 3

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
    
    amplified_depth = depth_image + upsampled
    return amplified_depth

alpha = 50
lambda_c = 16
r1 = 0.5
r2 = 0.05
chromAttenuation = 0.1
nlevels = 4

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
depth_image = np.asanyarray(depth_frame.get_data())
if depth_image is None:
    print("无法接收帧 (stream end?). Exiting ...")
    pipeline.stop()
    exit()

depth_image = depth_image.astype(np.float32)
depth_image /= 1000.0

color_scale_factor = 255 / 10
print(f"Color Scale Factor: {color_scale_factor}")
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=color_scale_factor), cv2.COLORMAP_JET)

roi = cv2.selectROI('Select ROI', depth_colormap, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select ROI')

pyr = build_laplacian_pyramid(depth_image, levels=nlevels)
laplowpass1 = copy.deepcopy(pyr)
laplowpass2 = copy.deepcopy(pyr)
pyr = build_gaussian_pyramid(depth_image, levels=nlevels)
gaulowpass1 = copy.deepcopy(pyr)
gaulowpass2 = copy.deepcopy(pyr)

fig, ax = plt.subplots()
depth_queues = [deque(maxlen=100) for _ in range(3)]
labels = ['Origin', 'Lap Pyr Amplified', 'Gau Pyr Amplified']
lines = [ax.plot([], [], lw=2, label=f'{label}')[0] for label in labels]
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.set_xlabel('Frame')
ax.set_ylabel('Average depth')
ax.legend()

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return lines

    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_image /= 1000.0

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=color_scale_factor), cv2.COLORMAP_JET)

    roi_frame = copy.deepcopy(depth_image)
    roi_depth = np.mean(roi_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    depth_queues[0].append(roi_depth)

    lines[0].set_data(range(len(depth_queues[0])), depth_queues[0])

    cv2.putText(depth_colormap, f"ROI Average Depth: {roi_depth:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(depth_colormap, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('Depth Image', depth_colormap)

    start_time = time.time()

    pyr = build_laplacian_pyramid(depth_image, levels=nlevels)
    amplified_depth = process_pyramid(pyr, laplowpass1, laplowpass2, r1, r2, alpha, lambda_c, nlevels, depth_image)
    roi_depth_lap = np.mean(amplified_depth[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    depth_queues[1].append(roi_depth_lap)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(amplified_depth, alpha=color_scale_factor), cv2.COLORMAP_JET)
    print(f"延迟: {(time.time() - start_time)*1000:.4f}ms")
    cv2.putText(depth_colormap, f"Delay: {(time.time() - start_time)*1000:.4f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(depth_colormap, f"ROI Average Depth: {roi_depth_lap:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(depth_colormap, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('Laplacian Pyramid Amplified Depth Image', depth_colormap)

    start_time = time.time()

    pyr = build_gaussian_pyramid(depth_image, levels=nlevels)
    amplified_depth = process_pyramid(pyr, gaulowpass1, gaulowpass2, r1, r2, alpha, lambda_c, nlevels, depth_image)
    roi_depth_gau = np.mean(amplified_depth[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
    depth_queues[2].append(roi_depth_gau)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(amplified_depth, alpha=color_scale_factor), cv2.COLORMAP_JET)
    print(f"延迟: {(time.time() - start_time)*1000:.4f}ms")
    cv2.putText(depth_colormap, f"Delay: {(time.time() - start_time)*1000:.4f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(depth_colormap, f"ROI Average Depth: {roi_depth_gau:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(depth_colormap, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('Gaussian Pyramid Amplified Depth Image', depth_colormap)

    lines[0].set_data(range(len(depth_queues[0])), depth_queues[0])
    lines[1].set_data(range(len(depth_queues[1])), depth_queues[1])
    lines[2].set_data(range(len(depth_queues[2])), depth_queues[2])
    ax.set_xlim(0, max(len(depth_queues[0]), 100))
    ax.set_ylim(0, max(max(depth_queues[0]), max(depth_queues[1]), max(depth_queues[2])) * 2)

    if cv2.waitKey(1) == ord('q'):
        plt.close(fig)
        pipeline.stop()
        cv2.destroyAllWindows()
        return lines

    return lines

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)
plt.show()