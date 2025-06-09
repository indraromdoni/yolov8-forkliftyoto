from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  # Path ke model hasil training

# Inference pada gambar
results = model("C:\\Users\\indra.romdoni\\OneDrive - mapi.co.id\\Pictures\\Forklift\\IMG_0151.jpeg")  # Bisa juga path ke video atau folder

# Menampilkan hasil
results[0].show()         # Menampilkan hasil secara visual
results[0].save(filename="output.jpg")  # Simpan hasil anotasi

# Menampilkan prediksi dalam bentuk array
print(results[0].boxes.xyxy)  # Bounding box
print(results[0].boxes.conf)  # Confidence
print(results[0].boxes.cls)   # Class