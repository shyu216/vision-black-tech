import cv2
import numpy as np
import time

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 加载 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 初始化追踪器
tracker = cv2.TrackerKCF_create()
initialized = False

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (stream end?). Exiting ...")
        break

    start_time = time.time()

    if initialized:
        # 更新追踪器
        success, bbox = tracker.update(frame)
        if not success:
            initialized = False
    if not initialized:
        # 如果追踪失败或未初始化，重新检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            bbox = (x, y, w, h)
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bbox)
            initialized = True

    if initialized:
        # 绘制追踪结果
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    end_time = time.time()
    delay_ms = (end_time - start_time) * 1000
    print("延迟: %.2fms" % delay_ms)
    cv2.putText(frame, "Delay: %.2fms" % delay_ms, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示原始帧
    cv2.imshow('Original Camera', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()