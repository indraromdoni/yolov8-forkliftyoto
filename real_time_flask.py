import threading
import cv2
import time
import av  # PyAV
import numpy as np
import json
import os
from collections import deque
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, request, redirect, url_for
from pyModbusTCP.client import ModbusClient

client = ModbusClient(host="192.168.23.73", port=502, auto_open=True)

# Load YOLOv8 model
model = YOLO("bestV4.pt")
model.to('cuda')  # Hapus atau komentari jika tidak pakai GPU

# RTSP URL
RTSP_URL = "rtsp://admin:adm12345@192.168.24.100/ISAPI/Streaming/channels/102/preview"

# Shared resources
frame_lock = threading.Lock()
shared_frame = deque(maxlen=3)
annotated_frame = None
_exit = False

forklift_buffer = deque(maxlen=5)
yoto_buffer = deque(maxlen=5)

# Garis diagonal (bisa diubah dari web)
LINE1_P1 = [390, 480]
LINE1_P2 = [550, 120]
LINE2_P1 = [40, 144]
LINE2_P2 = [205, 60]

app = Flask(__name__)

getimg_thread_started = False
inference_thread_started = False

LINES_FILE = "lines.json"

def save_lines():
    global LINE1_P1, LINE1_P2, LINE2_P1, LINE2_P2
    data = {
        "LINE1_P1": LINE1_P1,
        "LINE1_P2": LINE1_P2,
        "LINE2_P1": LINE2_P1,
        "LINE2_P2": LINE2_P2
    }
    with open(LINES_FILE, "w") as f:
        json.dump(data, f)

def load_lines():
    global LINE1_P1, LINE1_P2, LINE2_P1, LINE2_P2
    if os.path.exists(LINES_FILE):
        with open(LINES_FILE, "r") as f:
            data = json.load(f)
            LINE1_P1 = data.get("LINE1_P1", LINE1_P1)
            LINE1_P2 = data.get("LINE1_P2", LINE1_P2)
            LINE2_P1 = data.get("LINE2_P1", LINE2_P1)
            LINE2_P2 = data.get("LINE2_P2", LINE2_P2)

def getImg():
    global _exit, getimg_thread_started
    getimg_thread_started = True
    container = av.open(RTSP_URL)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    while not _exit:
        try:
            for frame in container.decode(stream):
                img = frame.to_ndarray(format='bgr24')
                with frame_lock:
                    shared_frame.append(img)
                if _exit:
                    break
                time.sleep(0.01)
            container = av.open(RTSP_URL)
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
        except Exception as e:
            print(f"RTSP/PyAV error: {e}")
            time.sleep(1)
            continue
    print("Get live image function finished")

def inference_thread():
    global _exit, annotated_frame, LINE1_P1, LINE1_P2, LINE2_P1, LINE2_P2, inference_thread_started
    inference_thread_started = True
    while not _exit:
        with frame_lock:
            if len(shared_frame) == 0:
                frame = None
            else:
                frame = shared_frame[-1].copy()
                shared_frame.clear()
        if frame is None:
            time.sleep(0.1)
            continue

        frame_resized = cv2.resize(frame, (640, 480))
        start_time = time.time()
        results = model(frame_resized, verbose=False, conf=0.25)
        annotated = results[0].plot()

        # --- Dua garis diagonal dari variabel ---
        cv2.line(annotated, tuple(LINE1_P1), tuple(LINE1_P2), (0, 255, 255), 2)
        cv2.line(annotated, tuple(LINE2_P1), tuple(LINE2_P2), (255, 0, 255), 2)

        # --- Visualisasi area deteksi in-between ---
        # Buat poligon dari ujung-ujung garis
        pts = np.array([
            LINE1_P1,
            LINE1_P2,
            LINE2_P2,
            LINE2_P1
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))

        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        alpha = 0.2  # transparansi area
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Hitung persamaan garis dari variabel
        x1a, y1a = LINE1_P1
        x1b, y1b = LINE1_P2
        m1 = (y1b - y1a) / (x1b - x1a) if (x1b - x1a) != 0 else 0.0001
        c1 = y1a - m1 * x1a

        x2a, y2a = LINE2_P1
        x2b, y2b = LINE2_P2
        m2 = (y2b - y2a) / (x2b - x2a) if (x2b - x2a) != 0 else 0.0001
        c2 = y2a - m2 * x2a

        boxes = results[0].boxes
        forklift_detected = False
        yoto_detected = False
        in_between = False

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            label = model.names[cls_id].lower()
            if "forklift" in label:
                forklift_detected = True
                center_x = int((x1 + x2) / 2)
                bottom_y = y2

                garis1_y = int(m1 * center_x + c1)
                garis2_y = int(m2 * center_x + c2)
                y_min = min(garis1_y, garis2_y)
                y_max = max(garis1_y, garis2_y)
                if y_min < bottom_y < y_max:
                    in_between = True

            if "yoto" in label:
                yoto_detected = True

        forklift_buffer.append(forklift_detected)
        yoto_buffer.append(yoto_detected)

        stable_forklift = sum(forklift_buffer) >= 3
        stable_yoto = sum(yoto_buffer) >= 3

        if stable_forklift and stable_yoto and in_between:
            outside_elapse = time.time() - start_outside
            start_inbetween = time.time()
        else:
            inbetween_elapse = time.time() - start_inbetween
            start_outside = time.time()
        
        if inbetween_elapse>2 and outside_elapse<=2:
            flag_inbetween = True
        elif outside_elapse>2:
            flag_inbetween = False

        if flag_inbetween:
            #print("Forklift and Yoto detected in between lines")
            res = client.write_single_coil(17, True)
            #print(res)
            cv2.putText(annotated, "Forklift+Yoto DI ANTARA GARIS! (STABIL)", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            res = client.write_single_coil(17, False)
            
# Simpan annotated frame untuk streaming
        annotated_frame = annotated.copy()
        finish_time = time.time()
        #print(f"Inference time: {finish_time - start_time:.2f} seconds")
        time.sleep(0.05)
        #time.sleep(finish_time - start_time)
    client.close()
    print("Inference thread finished")

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Traffic Light System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f8f8f8; }
        .container { max-width: 1000px; margin: auto; padding: 10px; }
        h2 { text-align: center; }
        img { width: 100%; max-width: 800px; display: block; margin: auto; border-radius: 8px; }
        a.button { display: inline-block; margin: 10px auto; padding: 10px 20px; background: #007bff; color: #fff; border-radius: 5px; text-decoration: none; text-align: center;}
        a.button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Traffic Light System</h2>
            <a href="{{ url_for('adjust') }}" class="button">Atur Garis Diagonal</a>
            <img src="{{ url_for('video_feed') }}">
        </div>
    </body>
    </html>
    """)

@app.route('/adjust', methods=['GET', 'POST'])
def adjust():
    global LINE1_P1, LINE1_P2, LINE2_P1, LINE2_P2
    if request.method == 'POST':
        try:
            LINE1_P1 = [int(request.form['l1x1']), int(request.form['l1y1'])]
            LINE1_P2 = [int(request.form['l1x2']), int(request.form['l1y2'])]
            LINE2_P1 = [int(request.form['l2x1']), int(request.form['l2y1'])]
            LINE2_P2 = [int(request.form['l2x2']), int(request.form['l2y2'])]
            save_lines()  # Simpan garis ke file
        except Exception as e:
            print("Invalid input:", e)
        return redirect(url_for('adjust'))  # Tetap di halaman adjust agar bisa lihat hasilnya

    # Form HTML + live stream
    return render_template_string("""
    <html>
    <head>
        <title>Adjust Garis Diagonal</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f8f8f8; }
        .container { max-width: 1000px; margin: auto; padding: 10px; }
        h2 { text-align: center; }
        form { background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
        label { display: inline-block; width: 60px; }
        input[type="number"] { width: 60px; padding: 5px; margin: 2px 0; border-radius: 4px; border: 1px solid #ccc;}
        input[type="submit"] { padding: 10px 20px; background: #007bff; color: #fff; border: none; border-radius: 5px; margin-top: 10px;}
        input[type="submit"]:hover { background: #0056b3; }
        img { width: 100%; max-width: 800px; display: block; margin: auto; border-radius: 8px; }
        a.button { display: inline-block; margin: 10px auto; padding: 10px 20px; background: #007bff; color: #fff; border-radius: 5px; text-decoration: none; text-align: center;}
        a.button:hover { background: #0056b3; }
        @media (max-width: 600px) {
            input[type="number"] { width: 40px; }
            .container { padding: 2px; }
        }
        </style>
    </head>
    <body>
    <div class="container">
    <h2>Adjust Garis Diagonal</h2>
    <form method="post">
      <b>Garis 1:</b><br>
      <label>x1:</label> <input type="number" name="l1x1" value="{{l1x1}}" required>
      <label>y1:</label> <input type="number" name="l1y1" value="{{l1y1}}" required>
      <label>x2:</label> <input type="number" name="l1x2" value="{{l1x2}}" required>
      <label>y2:</label> <input type="number" name="l1y2" value="{{l1y2}}" required><br>
      <b>Garis 2:</b><br>
      <label>x1:</label> <input type="number" name="l2x1" value="{{l2x1}}" required>
      <label>y1:</label> <input type="number" name="l2y1" value="{{l2y1}}" required>
      <label>x2:</label> <input type="number" name="l2x2" value="{{l2x2}}" required>
      <label>y2:</label> <input type="number" name="l2y2" value="{{l2y2}}" required><br>
      <input type="submit" value="Update">
    </form>
    <img src="{{ url_for('video_feed') }}">
    <a href="{{ url_for('index') }}" class="button">Kembali ke Stream Utama</a>
    </div>
    </body>
    </html>
    """, l1x1=LINE1_P1[0], l1y1=LINE1_P1[1], l1x2=LINE1_P2[0], l1y2=LINE1_P2[1],
         l2x1=LINE2_P1[0], l2y1=LINE2_P1[1], l2x2=LINE2_P2[0], l2y2=LINE2_P2[1])

def gen():
    global annotated_frame
    while True:
        if annotated_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_lines()  # Muat garis dari file saat aplikasi dimulai
    if not getimg_thread_started:
        t1 = threading.Thread(target=getImg, daemon=True)
        t1.start()
    if not inference_thread_started:
        t2 = threading.Thread(target=inference_thread, daemon=True)
        t2.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    _exit = True
    t1.join()
    t2.join()
