import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import matplotlib.pyplot as plt
import io
import time
from datetime import datetime

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class_names_goal = ['car', 'motorcycle', 'bus', 'truck']

class RTSPTrafficMonitor:
    def __init__(self, rtsp_url, model_path):
        # Initialize RTSP capture
        self.rtsp_url = rtsp_url
        self.video = cv2.VideoCapture(rtsp_url)
        
        # Configure RTSP stream settings
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if not self.video.isOpened():
            raise ValueError("Error: Could not open RTSP stream. Please check the URL and connection.")
        
        # Original video dimensions
        self.orig_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        
        # Calculate dimensions for grid layout
        self.grid_width = int(self.orig_width * 1.3)
        self.grid_height = self.orig_height
        self.video_width = int(self.grid_width * 0.7)
        self.chart_width = self.grid_width - self.video_width
        
        # Initialize YOLOv8x model
        self.model = YOLO(model_path)
        
        # Initialize video writer with timestamp in filename
        output_filename = f"traffic_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.grid_width, self.grid_height))
        
        # Initialize counters and tracking
        self.vehicle_count = {'left': 0, 'right': 0}
        self.counted_vehicle_ids = {'left': set(), 'right': set()}
        self.line_y = int(self.orig_height * 0.65)
        self.mask = self.create_mask()
        self.tracker = SimpleTracker()
        
        # Frame processing settings
        self.frame_skip = 2
        self.max_reconnect_attempts = 5

    def create_pie_chart(self):
        plt.figure(figsize=(8, 8))
        plt.clf()
        
        # Enhanced color scheme for better visibility
        colors = ['#FF6B6B', '#4ECDC4']  # Coral Red and Turquoise
        lanes = list(self.vehicle_count.keys())
        total_counts = [self.vehicle_count[lane] for lane in lanes]
        
        total_vehicles = sum(total_counts)
        labels = [f'{lane.upper()}\n{count} vehicles' for lane, count in zip(lanes, total_counts)]
        sizes = total_counts if sum(total_counts) > 0 else [1, 1]
        
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
            textprops={"fontsize": 12, "fontweight": "bold", "color": "white"}
        )
        
        plt.title(f'Vehicle Distribution\nTotal: {total_vehicles} vehicles',
                 color='white',
                 pad=20,
                 fontsize=14,
                 fontweight='bold')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1a1a', transparent=False, bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close()
        
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img

    def create_grid_layout(self, frame, pie_chart):
        # Create black canvas for grid
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        
        # Resize video frame to fit 70% of grid
        video_frame = cv2.resize(frame, (self.video_width, self.grid_height))
        
        # Resize pie chart to fit 30% of grid
        pie_chart = cv2.resize(pie_chart, (self.chart_width, self.grid_height))
        
        # Place video and pie chart in grid
        grid[:, :self.video_width] = video_frame
        grid[:, self.video_width:] = pie_chart
        
        # Add separator line
        cv2.line(grid, 
                 (self.video_width, 0),
                 (self.video_width, self.grid_height),
                 (255, 255, 255),
                 2)
        
        return grid

    def create_mask(self):
        mask = np.zeros((self.orig_height, self.orig_width), dtype=np.uint8)
        pts = np.array([
            [int(self.orig_width * 0.15), int(self.orig_height * 0.55)],
            [int(self.orig_width * 0.85), int(self.orig_height * 0.55)],
            [self.orig_width, self.orig_height],
            [0, self.orig_height]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def detect_vehicles(self, image_region):
        detections = []
        results = self.model(image_region, stream=True)
        
        for r in results:
            for box in r.boxes:
                class_name = class_names[int(box.cls[0])]
                if class_name not in class_names_goal:
                    continue
                
                confidence = round(float(box.conf[0]) * 100, 2)
                if confidence < 60:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, float(box.conf[0]), class_name])
        
        return detections

    def process_frame(self, frame):
        # Resize frame to original dimensions
        frame = cv2.resize(frame, (self.orig_width, self.orig_height))
        image_region = cv2.bitwise_and(frame, frame, mask=self.mask)
        
        # Detect and track vehicles
        detections = self.detect_vehicles(image_region)
        tracked_objects = self.tracker.update(detections)
        
        # Count vehicles and annotate frame
        self.count_vehicles(frame, tracked_objects)
        self.annotate_frame(frame, tracked_objects)
        
        # Create pie chart
        pie_chart = self.create_pie_chart()
        
        # Create grid layout
        grid_frame = self.create_grid_layout(frame, pie_chart)
        
        return grid_frame

    def count_vehicles(self, frame, tracked_objects):
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, cls = obj
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            if self.line_y - 10 < center_y < self.line_y + 10:
                if center_x < self.orig_width // 2:  # Left side
                    if obj_id not in self.counted_vehicle_ids['left']:
                        self.counted_vehicle_ids['left'].add(obj_id)
                        self.vehicle_count['left'] += 1
                        cv2.line(frame, 
                                (0, self.line_y), 
                                (self.orig_width // 2, self.line_y), 
                                (46, 204, 113), 3)
                else:  # Right side
                    if obj_id not in self.counted_vehicle_ids['right']:
                        self.counted_vehicle_ids['right'].add(obj_id)
                        self.vehicle_count['right'] += 1
                        cv2.line(frame, 
                                (self.orig_width // 2, self.line_y), 
                                (self.orig_width, self.line_y), 
                                (46, 204, 113), 3)

    def annotate_frame(self, frame, tracked_objects):
        # Draw detection line
        cv2.line(frame, (0, self.line_y), (self.orig_width, self.line_y),
                 (255, 255, 255), 1, cv2.LINE_AA)
        
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, cls = obj
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), l=9, rt=2, colorR=(255, 255, 255))
            
            text = f'ID: {obj_id} {cls}'
            font_scale = 0.6
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(frame,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         (46, 204, 113),
                         -1)
            
            cv2.putText(frame,
                       text,
                       (x1 + 5, y1 - 5),
                       font,
                       font_scale,
                       (255, 255, 255),
                       thickness)
        
        cvzone.putTextRect(frame,
                          f'Left Lane: {self.vehicle_count["left"]}',
                          (20, 40),
                          scale=2,
                          thickness=3,
                          offset=10,
                          colorR=(52, 73, 94),
                          colorT=(255, 255, 255))
        
        cvzone.putTextRect(frame,
                          f'Right Lane: {self.vehicle_count["right"]}',
                          (self.orig_width - 280, 40),
                          scale=2,
                          thickness=3,
                          offset=10,
                          colorR=(52, 73, 94),
                          colorT=(255, 255, 255))

    def process_rtsp_stream(self):
        frame_count = 0
        reconnect_attempts = 0
        
        try:
            while True:
                try:
                    ret, frame = self.video.read()
                    if not ret:
                        print("Failed to grab frame from stream")
                        reconnect_attempts += 1
                        if reconnect_attempts > self.max_reconnect_attempts:
                            print("Max reconnection attempts reached. Exiting...")
                            break
                        
                        print(f"Attempting to reconnect... ({reconnect_attempts}/{self.max_reconnect_attempts})")
                        self.video.release()
                        time.sleep(2)
                        self.video = cv2.VideoCapture(self.rtsp_url)
                        continue
                    
                    reconnect_attempts = 0
                    
                    frame_count += 1
                    if frame_count % self.frame_skip != 0:
                        continue
                    
                    processed_frame = self.process_frame(frame)
                    
                    cv2.imshow('Traffic Monitoring', processed_frame)
                    self.out.write(processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue
                    
        finally:
            print("Cleaning up resources...")
            self.video.release()
            self.out.release()
            cv2.destroyAllWindows()

class SimpleTracker:
    def __init__(self, max_age=20):
        self.next_id = 1
        self.tracked_objects = {}
        self.max_age = max_age

    def update(self, detections):
        new_tracked_objects = {}
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            best_match = None
            min_distance = float('inf')
            
            for obj_id, obj in self.tracked_objects.items():
                dist = np.linalg.norm(np.array(center) - np.array(obj['center']))
                if dist < min_distance:
                    min_distance = dist
                    best_match = obj_id
            
            if best_match is not None and min_distance < 50:
                new_tracked_objects[best_match] = {
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'class': cls,
                    'age': 0
                }
            else:
                new_tracked_objects[self.next_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'class': cls,
                    'age': 0
                }
                self.next_id += 1
        
        for obj_id in self.tracked_objects:
            if obj_id not in new_tracked_objects:
                self.tracked_objects[obj_id]['age'] += 1
                if self.tracked_objects[obj_id]['age'] < self.max_age:
                    new_tracked_objects[obj_id] = self.tracked_objects[obj_id]
        
        self.tracked_objects = new_tracked_objects
        return [(obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3], obj_id, obj['class']) 
                for obj_id, obj in self.tracked_objects.items()]

def main():
    # RTSP stream URL - replace with your RTSP stream URL
    rtsp_url = "rtsp://username:password@ip_address:port/stream"
    
    # Path to YOLOv8 model
    model_path = "path/to/your/yolov8x.pt"
    
    try:
        # Create and run monitor
        monitor = RTSPTrafficMonitor(rtsp_url, model_path)
        monitor.process_rtsp_stream()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()