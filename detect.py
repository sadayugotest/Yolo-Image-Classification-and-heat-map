import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. โหลดโมเดล
model_path = 'best2.pt'
model = YOLO(model_path)

# ทำ Prediction เพื่อหาว่ารูปนี้คือ Class อะไร
img_path = 'NG.jpg'
results = model.predict(img_path)[0]
class_id = results.probs.top1  # Index ของคลาสที่คะแนนสูงสุด
class_name = results.names[class_id]  # ชื่อคลาส (เช่น 'OK' หรือ 'NG')
confidence = results.probs.top1conf.item() * 100 # ค่าความมั่นใจ (%)

# เตรียมโมเดลสำหรับ Grad-CAM
model.model.train()
for param in model.model.parameters():
    param.requires_grad = True

# 2. ระบุ Layer
target_layers = [model.model.model[-2]]

# 3. เตรียมรูปภาพ
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (224, 224))
img_float = np.float32(img_resized) / 255

input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
input_tensor.requires_grad = True

# 4. สร้าง Grad-CAM (ใช้ class_id ที่โมเดลทายได้มาทำ Heatmap)
cam = GradCAM(model=model.model, target_layers=target_layers)
targets = [ClassifierOutputTarget(class_id)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

# 5. วาด Heatmap
visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR) # แปลงกลับเป็น BGR สำหรับ OpenCV

# 6. เขียนข้อความ Class Name และ Confidence ลงบนภาพ
label = f"Result: {class_name} ({confidence:.2f}%)"
# ใส่พื้นหลังสีดำให้ตัวอักษรอ่านง่ายขึ้น
cv2.rectangle(visualization, (5, 5), (210, 35), (0, 0, 0), -1)
cv2.putText(visualization, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1, cv2.LINE_AA)

# 7. แสดงผลและบันทึกภาพ
cv2.imshow('Anomaly Heatmap with Label', visualization)
cv2.imwrite('result_with_label.jpg', visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()