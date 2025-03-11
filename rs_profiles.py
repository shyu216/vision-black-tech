import pyrealsense2 as rs

# 配置 RealSense 流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 查询相机支持的最大分辨率和帧率
pipeline_profile = pipeline.start(config)
device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

# 获取深度传感器支持的所有模式
depth_sensor_profiles = depth_sensor.get_stream_profiles()
for profile in depth_sensor_profiles:
    print(profile)

# 停止流以重新配置
pipeline.stop()