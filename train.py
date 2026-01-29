from ultralytics import YOLO

# 1. โหลดโมเดลสำหรับ Classification โดยเฉพาะ
model = YOLO('yolo26s-cls.pt') # หรือ 'yolov8n-cls.pt'

# 2. สั่ง Train
results = model.train(
    data='my_dataset2',  # ใส่ Path ของโฟลเดอร์หลักที่มี train/val
    epochs=50,                  # จำนวนรอบในการเทรน
    imgsz=224,                  # ขนาดรูปภาพ (มาตรฐาน Classification คือ 224)
    batch=2,                  # ขนาดของ Batch
    project='my_classification', # ชื่อโปรเจกต์
    name='experiment_2'         # ชื่อการทดลอง
)

# 3. ตรวจสอบความแม่นยำ
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")

