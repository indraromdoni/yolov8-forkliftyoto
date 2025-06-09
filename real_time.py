import threading
import cv2
import time
from collections import deque
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")
model.to('cuda')  # Hapus atau komentari jika tidak pakai GPU

# RTSP URL
RTSP_URL = "rtsp://admin:adm12345@192.168.24.100/ISAPI/Streaming/channels/102/preview"

# Shared resources
frame_lock = threading.Lock()
shared_frame = deque(maxlen=3)
_exit = False
isRun1 = False
isRun2 = False

# Koordinat garis khayal (misal, horizontal di tengah frame)
LINE_Y = 240  # Untuk frame 480px tinggi, garis di tengah

# Thread 1: Mengambil frame dari RTSP
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
            time.sleep(0.1)  # Tunggu sebelum mencoba lagi
            continue

        with frame_lock:
            shared_frame.append(img)
        time.sleep(0.05)
    cap.release()
    print("Get live image function finished")

# Thread 2: Inference dan tampilkan hasil
def inference_thread():
    global _exit, isRun2
    if isRun2:
        return
    isRun2 = True

    while not _exit:
        with frame_lock:
            if len(shared_frame) == 0:
                frame = None
            else:
                frame = shared_frame[-1].copy()
        if frame is None:
            time.sleep(0.1)
            continue

        # Resize opsional untuk performa
        frame_resized = cv2.resize(frame, (640, 480))

        # Jalankan inference
        results = model(frame_resized, verbose=False, conf=0.52)

        # Gambar bounding box di frame_resized
        annotated = results[0].plot()

        # --- Tambahan: Dua garis diagonal ---
        # Garis 1: dari (180, 480) ke (500, 120)
        cv2.line(annotated, (180, 480), (500, 120), (0, 255, 255), 2)
        # Garis 2: dari (50, 180) ke (240, 70)
        cv2.line(annotated, (50, 180), (240, 70), (255, 0, 255), 2)

        boxes = results[0].boxes
        forklift_detected = False
        yoto_detected = False
        crossed1 = False
        crossed2 = False

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            label = model.names[cls_id].lower()
            if "forklift" in label:
                forklift_detected = True
                # Titik tengah bawah bounding box
                center_x = int((x1 + x2) / 2)
                bottom_y = y2

                # --- Cek crossing garis 1 (180,480)-(500,120) ---
                # Persamaan garis: y = m1*x + c1
                m1 = (120 - 480) / (500 - 180)
                c1 = 480 - m1 * 180
                garis1_y = int(m1 * center_x + c1)
                if bottom_y < garis1_y:
                    crossed1 = True

                # --- Cek crossing garis 2 (50,180)-(240,70) ---
                m2 = (70 - 180) / (240 - 50)
                c2 = 180 - m2 * 50
                garis2_y = int(m2 * center_x + c2)
                if bottom_y < garis2_y:
                    crossed2 = True

            if "yoto" in label:
                yoto_detected = True

        # Tandai crossing jika forklift & yoto terdeteksi dan forklift melewati salah satu garis
        if forklift_detected and yoto_detected and crossed1:
            cv2.putText(annotated, "Forklift+Yoto LEWAT GARIS 1!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if forklift_detected and yoto_detected and crossed2:
            cv2.putText(annotated, "Forklift+Yoto LEWAT GARIS 2!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow("YOLOv8 Inference", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            _exit = True
            break
        time.sleep(0.05)  # Hindari beban CPU berlebihan

    cv2.destroyAllWindows()
    print("Inference thread finished")

# Mulai thread
t1 = threading.Thread(target=getImg)
t2 = threading.Thread(target=inference_thread)

t1.start()
t2.start()

t1.join()
t2.join()
