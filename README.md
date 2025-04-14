# Road_Traffic_Monitoring
# 🚦 Intelligent Road Traffic Monitoring System

This project provides a computer vision-based traffic monitoring and vehicle counting system using YOLOv8. It detects and tracks vehicles from video feeds or RTSP streams, counts lane-wise traffic, and provides visual analytics including pie charts for vehicle distribution.

---

## 📁 Project Structure

- `vehicle_monitoring.py`: Monitors traffic in real-time using an RTSP camera feed.
- `app.py`: Processes and analyzes traffic footage from a pre-recorded video file.
- `road.ipynb`: Notebook version for demo, experimentation, and visualization.

---

## 🎯 Features

- 🧠 Vehicle detection and classification using **YOLOv8**
- 🛣️ Lane-based vehicle counting (left, middle, right)
- 📊 Real-time pie chart generation to visualize vehicle distribution
- 🔄 Object tracking with a custom **SimpleTracker**
- 📹 Works with both recorded videos and **RTSP live streams**
- 💾 Saves output video with annotations and charts

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install ultralytics opencv-python cvzone matplotlib numpy
```

Additional dependencies:
- `ffmpeg` (for video codec compatibility)
- Python 3.8+

---

## 🚀 How to Run

### 📡 RTSP Stream (Real-Time Monitoring)

Edit `vehicle_monitoring.py`:

```python
rtsp_url = "rtsp://username:password@ip_address:port/stream"
model_path = "path/to/your/yolov8x.pt"
```

Run the script:

```bash
python vehicle_monitoring.py
```

### 🎥 Video File (Offline Monitoring)

Edit `app.py`:

```python
video = cv2.VideoCapture("path/to/traffic.mp4")
model = YOLO("path/to/yolov8n.pt")
```

Run:

```bash
python app.py
```

Output will be saved as `traffic_output.mp4`.

---

## 📊 Output

- Annotated video with:
  - Bounding boxes and object IDs
  - Lane-based vehicle counts
  - Overlaid pie charts
- Live visualization through OpenCV window

---

## 📌 Notes

- Only vehicles of interest (e.g., **car**, **motorcycle**, **bus**, **truck**) are counted.
- Pie charts are dynamically created using `matplotlib` and overlaid with OpenCV.
- Detection accuracy and performance depend on lighting and video quality.

---

## 🧑‍💻 Author

Sharath Yanamandra  
For queries or collaboration, reach out via GitHub: [sharath-yanamandra](https://github.com/sharath-yanamandra)

---

Let me know if you'd like to include visuals (like sample outputs) or instructions for deploying this as a web app!
