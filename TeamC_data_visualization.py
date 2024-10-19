# 使用模型:YOLOv8m(可于第16行修改)
# 注意修改视频文件地址(第66行)
# 使用时请于Terminal中输入"streamlit run TeamC_data_visualization.py"
# 请确保所需依赖、库已安装
# 结果在网页中显示，所以建议不要拒绝你所使用的软件访问浏览器


import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# 加载 YOLOv8m 模型
model = YOLO('yolov8m.pt')


# 假设检测结果来自YOLO模型
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
    Args:
        frame: 原始视频帧
        detections: YOLO检测结果，包含目标的类别和位置
    """
    for detection in detections:
        # 提取坐标
        x1, y1, x2, y2 = map(int, detection[:4])  # 边界框坐标
        confidence = detection[4]  # 置信度
        class_id = int(detection[5])  # 类别ID
        label = model.names[class_id]  # 类别名称
        color = (0, 255, 0)  # 绿色边框
        # 绘制边框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 添加标签
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


# Streamlit 展示Web界面
st.title("Traffic Monitoring Dashboard\n交通监控仪表板")

# 视频展示部分
st.write("Processed Video Frames with Detections:")

# 打开视频文件
cap = cv2.VideoCapture("D:/Games/traffic_video.mp4")  # 替换为你的视频路径

frame_id = 1
detection_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每三帧为一组，处理组中的第 2 帧（即 frame_id % 3 == 2 的帧）
    if frame_id % 3 == 2:
        # 使用 YOLO 模型进行检测
        results = model(frame)  # 使用 YOLOv8m 模型检测目标
        detections = results[0].boxes.data.cpu().numpy()  # 提取检测结果
        detection_frames.append((frame_id, detections))  # 保存 frame_id 和检测数据
        frame = draw_detections_on_frame(frame, detections)  # 绘制检测结果到帧上

        # 将处理后的帧展示在Streamlit中
        st.image(frame, channels="BGR", caption=f"Frame {frame_id}")

    frame_id += 1

# 处理检测数据并生成可视化
detection_results = process_detection_data(detection_frames)

# 检查是否有数据
if detection_results:
    df = pd.DataFrame(detection_results)  # 转换为 DataFrame

    # 使用 Plotly 生成折线图
    fig = px.line(df, x="frame_id", y=["car", "pedestrian", "bike"],
                  labels={"frame_id": "Frame ID", "value": "Count"},
                  title="Traffic Participants Count Over Time")

    # 展示折线图
    st.write("Traffic Count Over Time:")
    st.plotly_chart(fig)
else:
    st.write("No detection data available.")

cap.release()
