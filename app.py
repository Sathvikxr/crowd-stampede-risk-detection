import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ------------------ Page setup ------------------
st.set_page_config(page_title="Crowd Stampede Risk Dashboard", layout="wide")

st.markdown("""
# 🚨 Crowd Behavior & Stampede Risk Monitoring System  
**Early Warning Analytics Dashboard (Prototype)**
""")

# ------------------ Load YOLO model ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------ Video input ------------------
video_path = "videos/sample.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("Could not open video file.")
    st.stop()

ret, prev_frame = cap.read()
if not ret:
    st.error("Could not read first frame.")
    st.stop()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_width * frame_height
zone_width = frame_width // 3

# ------------------ Buffers ------------------
people_history = []
risk_history = []
risk_memory = deque(maxlen=12)

# ------------------ Layout ------------------
status_bar = st.empty()
col1, col2 = st.columns([2.2, 1.3])

video_placeholder = col1.empty()
info_placeholder = col2.empty()
graph_placeholder = st.empty()

start_button = st.button("▶ Start Monitoring")

# ------------------ Helper: zone risk ----------
def get_zone_risk(zone_count, total):
    if total == 0:
        return "LOW", "🟢"
    ratio = zone_count / total
    if ratio > 0.5:
        return "HIGH", "🔴"
    elif ratio > 0.25:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🟢"

# ------------------ Main loop ------------------
if start_button:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people_count = 0
        person_centers = []

        left_count = 0
        center_count = 0
        right_count = 0

        # -------- YOLO detection --------
        results = model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf > 0.4:
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    person_centers.append((cx, cy))

                    if cx < zone_width:
                        left_count += 1
                    elif cx < 2 * zone_width:
                        center_count += 1
                    else:
                        right_count += 1

        # -------- Density --------
        density = people_count / frame_area
        if density < 0.00005:
            density_level = "LOW"
        elif density < 0.00015:
            density_level = "MEDIUM"
        else:
            density_level = "HIGH"

        # -------- Motion --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)

        if motion_score < 1.0:
            motion_state = "CALM"
        elif motion_score < 3.0:
            motion_state = "MOVING"
        else:
            motion_state = "CHAOTIC"

        prev_gray = gray

        # -------- Global Risk (unchanged) --------
        if density_level == "HIGH" and motion_state == "CHAOTIC":
            instant_risk = 3
        elif density_level in ["MEDIUM", "HIGH"] or motion_state == "MOVING":
            instant_risk = 2
        else:
            instant_risk = 1

        risk_memory.append(instant_risk)
        stable_risk = round(np.mean(risk_memory))

        if stable_risk == 3:
            final_risk = "HIGH RISK"
            icon = "🔴"
        elif stable_risk == 2:
            final_risk = "WARNING"
            icon = "🟡"
        else:
            final_risk = "NORMAL"
            icon = "🟢"

        people_history.append(people_count)
        risk_history.append(stable_risk)

        # -------- Zone Risks --------
        left_risk, left_icon = get_zone_risk(left_count, people_count)
        center_risk, center_icon = get_zone_risk(center_count, people_count)
        right_risk, right_icon = get_zone_risk(right_count, people_count)

        # -------- Zone visuals --------
        cv2.line(frame, (zone_width, 0), (zone_width, frame_height), (255, 255, 255), 2)
        cv2.line(frame, (2 * zone_width, 0), (2 * zone_width, frame_height), (255, 255, 255), 2)

        cv2.putText(frame, f"LEFT ({left_icon})",
                    (zone_width // 2 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"CENTER ({center_icon})",
                    (zone_width + zone_width // 2 - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"RIGHT ({right_icon})",
                    (2 * zone_width + zone_width // 2 - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -------- Heatmap --------
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        for (x, y) in person_centers:
            heatmap[y, x] += 1

        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        frame = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # -------- Display --------
        status_bar.markdown(f"### CURRENT STATUS: {icon} **{final_risk}**")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        info_placeholder.markdown(f"""
        ### 📊 Global Metrics
        **Total People:** {people_count}  
        **Density Level:** {density_level}  
        **Motion State:** {motion_state}  

        ---
        ### 🧭 Zone-wise Risk
        - **Left Zone:** {left_icon} {left_risk}  
        - **Center Zone:** {center_icon} {center_risk}  
        - **Right Zone:** {right_icon} {right_risk}  

        ---
        **Decision Model:** Global + Zone-based analysis
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(2, 1, figsize=(7, 4))
        ax[0].plot(people_history, color="blue")
        ax[0].set_title("People Count Over Time")

        ax[1].plot(risk_history, color="red")
        ax[1].set_title("Stable Global Risk Level")

        plt.tight_layout()
        graph_placeholder.pyplot(fig)

        time.sleep(0.05)

    cap.release()
    st.success("Monitoring completed.")
