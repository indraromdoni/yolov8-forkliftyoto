from flask import Flask, render_template, Response
import io
import cv2
import numpy as np
import time
import ssl
import threading
import tensorflow as tf
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

# TFLite model setup
interpreter = tf.lite.Interpreter(
    model_path="best-fp16.tflite",
    num_threads=1
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

names = ["Forklift", "Yoto"]
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

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


def preprocess(image):
    # Resize ke ukuran model (asumsinya 640x640)
    img_resized = cv2.resize(image, (640, 640))
    # Konversi BGR ke RGB jika model mengharapkan RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Normalisasi ke [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    # Tambahkan dimensi batch: (1, 640, 640, 3)
    return np.expand_dims(img_norm, axis=0)


def imgProc():
    global _exit, isRun2
    if isRun2:
        return
    isRun2 = True

    cls_old = -1  # inisialisasi berbeda dari 0 untuk deteksi awal

    while not _exit:
        with frame_lock:
            if not frame:
                time.sleep(0.05)
                continue
            img = frame[-1].copy()

        # Preprocess image
        im = preprocess(img)
        interpreter.set_tensor(input_details[0]["index"], im)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])  # shape (1, N, 7)

        detections = output_data[0]  # remove batch dim -> shape (N, 7)
        conf_thresh = 0.3

        # Filter valid detections
        valid_detections = [
            det for det in detections if det[4] > conf_thresh
        ]

        if not valid_detections:
            with buffer_lock:
                buffer[0], buffer[1], buffer[2] = [], "", []
            time.sleep(0.05)
            continue

        # Choose best detection (highest object_conf * class_conf)
        best = max(valid_detections, key=lambda det: det[4] * det[6])

        try:
            x0, y0, x1, y1 = map(int, best[0:4])
            cls_id = int(best[5])
            score = float(best[6])
        except Exception as e:
            print("Parsing error:", e)
            time.sleep(0.05)
            continue

        if cls_id != cls_old:
            print("DC High" if cls_id else "DC Low")
            cls_old = cls_id

        name = f"{names[cls_id]} {round(score, 3)}"
        color = colors[names[cls_id]]

        with buffer_lock:
            buffer[0], buffer[1], buffer[2] = [x0, y0, x1, y1], name, color

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
    '''context.load_cert_chain(
        certfile='cert.crt',
        keyfile='cert.key',
        password='P4ssword'
    )

    app.run(host='0.0.0.0', debug=False, threaded=True, ssl_context=context, port=5000)'''
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5000)

    _exit = True
    t1.join()
    t2.join()
