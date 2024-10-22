"""
使用模型:YOLOv8m(可于第21行修改)
注意修改视频文件地址(第66行) ————视频文件不建议太长
使用时请于Terminal中输入"streamlit run TeamC_data_visualization.py"
请确保所需依赖、库已安装
结果在网页中显示，所以建议不要拒绝你所使用的软件访问浏览器
视频流检测图片仅显示3张，但实际检测为 3x-1帧(如1 2 3帧检测第2帧)
"""


import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# 加载 YOLOv8m 模型
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()


# 假设检测结果来自YOLO模型
@st.cache_data
def process_detection_data(detection_frames):
    """
    处理YOLO检测结果，将每帧中的目标检测数量进行汇总
    """
    frame_data = []
    for frame_id, detections in detection_frames:
        count = {"frame_id": frame_id, "car": 0, "pedestrian": 0, "bike": 0}
        for detection in detections:
            class_id = int(detection[5])
            class_type = model.names[class_id]
            if class_type in count:
                count[class_type] += 1
        frame_data.append(count)
    return frame_data


# 视频帧目标检测可视化
def draw_detections_on_frame(frame, detections):
    """
    在每一帧上绘制检测到的目标框
    """
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        confidence = detection[4]
        class_id = int(detection[5])
        label = model.names[class_id]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


# Streamlit 展示Web界面
st.title("Traffic Monitoring Dashboard\n交通监控仪表板")
st.write("Processed Video Frames with Detections:")

# 视频展示部分
video_path = "D:/Games/traffic_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_id = 1
detection_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每三帧为一组，处理组中的第 2 帧（即 frame_id % 3 == 2 的帧）
    if frame_id % 3 == 2:
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()
        detection_frames.append((frame_id, detections))
        frame = draw_detections_on_frame(frame, detections)

        # 控制帧展示的频率，减少 Streamlit 的负担
        if frame_id % 30 == 2:  # 每 30 帧展示一次
            st.image(frame, channels="BGR", caption=f"Frame {frame_id}")

    frame_id += 1

# 处理检测数据并生成可视化
if detection_frames:
    detection_results = process_detection_data(detection_frames)
    df = pd.DataFrame(detection_results)

    # 使用 Plotly 生成折线图
    fig = px.line(df, x="frame_id", y=["car", "pedestrian", "bike"],
                  labels={"frame_id": "Frame ID", "value": "Count"},
                  title="Traffic Participants Count Over Time")

    st.write("Traffic Count Over Time:")
    st.plotly_chart(fig)
else:
    st.write("No detection data available.")

cap.release()
