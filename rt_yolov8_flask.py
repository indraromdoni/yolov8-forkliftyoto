from flask import Flask, render_template, Response
import io
import cv2
import numpy as np
import time
import ssl
import threading
import random
from collections import deque
import torch

app = Flask(__name__)

frame = deque(maxlen=3)
buffer = [[], "", []]
frame_lock = threading.Lock()
buffer_lock = threading.Lock()

_exit = False
isRun1 = False
isRun2 = False

# Load YOLOv5 model (pastikan best.pt ada di folder yang sama)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestYolov5.pt')
model.to('cuda')  # Hapus baris ini jika tidak pakai GPU

# Ambil nama kelas dari model
names = model.names
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names.values()}

# RTSP URL
RTSP_URL = "rtsp://admin:adm12345@192.168.24.100/ISAPI/Streaming/channels/102/preview"

def getImg():
    global _exit, isRun1
    if isRun1:
        return
    isRun1 = True

    cap = cv2.VideoCapture(RTSP_URL)
    fail_count = 0

    while not _exit:
        ok, img = cap.read()
        if not ok:
            fail_count += 1
            if fail_count > 5:
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                fail_count = 0
            time.sleep(0.1)
            continue

        with frame_lock:
            frame.append(img)
        time.sleep(0.05)  # ~20 FPS

    cap.release()
    print("Get live image function finished")


def imgProc():
    global _exit, isRun2
    if isRun2:
        return
    isRun2 = True

    while not _exit:
        with frame_lock:
            if not frame:
                time.sleep(0.05)
                continue
            img = frame[-1].copy()

        # Resize ke 640x480 (atau sesuai input model)
        img_resized = cv2.resize(img, (640, 480))
        results = model(img_resized, verbose=False)
        boxes = results[0].boxes

        # Simpan semua box, nama, dan warna
        if boxes and len(boxes) > 0:
            all_boxes = []
            all_names = []
            all_colors = []
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                cls_id = int(boxes.cls[i].cpu().numpy())
                score = float(boxes.conf[i].cpu().numpy())
                name = f"{names[cls_id]} {score:.2f}"
                color = colors[names[cls_id]]
                all_boxes.append(box.tolist())
                all_names.append(name)
                all_colors.append(color)
            with buffer_lock:
                buffer[0], buffer[1], buffer[2] = all_boxes, all_names, all_colors
        else:
            with buffer_lock:
                buffer[0], buffer[1], buffer[2] = [], [], []

        time.sleep(0.05)

    print("Image processing finished")


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        with frame_lock:
            if not frame:
                continue
            img = frame[-1].copy()

        with buffer_lock:
            boxes, names_, colors_ = buffer

        if boxes:
            for box, name, color in zip(boxes, names_, colors_):
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img, name, (box[0], box[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)

        encode_return_code, image_buffer = cv2.imencode('.jpg', img)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    t1 = threading.Thread(target=getImg, daemon=True)
    t2 = threading.Thread(target=imgProc, daemon=True)
    t1.start()
    t2.start()

    '''context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    context.load_cert_chain(
        certfile='C:\\Users\\enggineadmin\\Documents\\From E\\Aplikasi python\\mqtt image\\cert.crt',
        keyfile='C:\\Users\\enggineadmin\\Documents\\From E\\Aplikasi python\\mqtt image\\cert.key',
        password='P4ssword'
    )

    app.run(host='0.0.0.0', debug=False, threaded=True, ssl_context=context, port=5000)'''
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5000)

    _exit = True
    t1.join()
    t2.join()
