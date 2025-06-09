import threading
import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")
model.to('cuda')

# RTSP URL
RTSP_URL = "rtsp://admin:adm12345@192.168.24.100/ISAPI/Streaming/channels/102/preview"

