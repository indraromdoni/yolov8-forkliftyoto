from flask import Flask, render_template, Response
import io
import cv2
import numpy as np
import time
import ssl
import threading
import onnxruntime as ort
import random
from collections import deque

app = Flask(__name__)

# Thread-safe frame and buffer
frame = deque(maxlen=3)
buffer = [[], "", []]
frame_lock = threading.Lock()
buffer_lock = threading.Lock()

_exit = False
isRun1 = False
isRun2 = False

# ONNX model setup
session = ort.InferenceSession(
    "best.onnx",
    providers=['CPUExecutionProvider']
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

names = ["Forklift", "Yoto"]
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

def getImg():
    global _exit, isRun1
    if isRun1:
        return
    isRun1 = True

    cap = cv2.VideoCapture("rtsp://admin:adm12345@192.168.24.80/ISAPI/Streaming/channels/102/preview")
    fail_count = 0

    while not _exit:
        ok, img = cap.read()
        if not ok:
            fail_count += 1
            if fail_count > 5:
                cap.release()
                cap = cv2.VideoCapture("rtsp://admin:adm12345@192.168.24.80/ISAPI/Streaming/channels/102/preview")
                fail_count = 0
            time.sleep(0.1)
            continue

        with frame_lock:
            frame.append(img)
        time.sleep(0.05)  # ~20 FPS

    cap.release()
    print("Get live image function finished")

def preprocess(image):
    # Resize ke 640x640 (atau sesuai input model)
    img_resized = cv2.resize(image, (640, 640))
    # Transpose ke channel-first (3, 640, 640)
    img_transposed = img_resized.transpose((2, 0, 1))
    # Normalize ke 0–1
    img_norm = img_transposed.astype(np.float32) / 255.0
    # Add batch dimension: (1, 3, 640, 640)
    return np.expand_dims(img_norm, axis=0)

def imgProc():
    global _exit, isRun2
    if isRun2:
        return
    isRun2 = True

    cls_old = 0

    while not _exit:
        with frame_lock:
            if not frame:
                time.sleep(0.05)
                continue
            img = frame[-1].copy()

        im = preprocess(img)
        # Inference ONNX
        output_data = session.run([output_name], {input_name: im})[0]

        cls_now = int(output_data[0][5])
        if cls_now != cls_old:
            print("DC High" if cls_now else "DC Low")
            cls_old = cls_now

        x0, y0, x1, y1 = output_data[0][1:5]
        box = [int(val) for val in [x0, y0, x1, y1]]
        score = round(float(output_data[0][6]), 3)
        name = f"{names[cls_now]} {score}"
        color = colors[names[cls_now]]

        with buffer_lock:
            buffer[0], buffer[1], buffer[2] = box, name, color

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
            box, name, color = buffer

        if box:
            cv2.rectangle(img, box[:2], box[2:], color, 2)
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

    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    context.load_cert_chain(
        certfile='C:\\Users\\enggineadmin\\Documents\\From E\\Aplikasi python\\mqtt image\\cert.crt',
        keyfile='C:\\Users\\enggineadmin\\Documents\\From E\\Aplikasi python\\mqtt image\\cert.key',
        password='P4ssword'
    )

    app.run(host='0.0.0.0', debug=False, threaded=True, ssl_context=context, port=5000)

    _exit = True
    t1.join()
    t2.join()
