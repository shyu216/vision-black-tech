import pyrealsense2 as rs
import numpy as np
import cv2

# 配置 RealSense 流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动流
pipeline.start(config)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 将深度图像转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data()) # 毫米

        # 将深度图像归一化到 0-255 范围内以便显示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示深度图像
        cv2.imshow('Depth Image', depth_colormap)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止流
    pipeline.stop()

cv2.destroyAllWindows()