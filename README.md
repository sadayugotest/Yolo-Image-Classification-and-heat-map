# 🔍 YOLO Classification + Grad-CAM Heatmap

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ultralytics-YOLO-purple?logo=yolo&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-GradCAM-orange?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey"/>
</p>

โปรแกรมสำหรับ **ตรวจสอบความผิดปกติ (Anomaly Detection)** บนชิ้นงานหรือผลิตภัณฑ์ โดยใช้ **YOLO Classification** ร่วมกับ **Grad-CAM Heatmap** เพื่อแสดงบริเวณที่โมเดลให้ความสนใจในการตัดสินใจจำแนกประเภท

---

## 📋 สารบัญ

- [ภาพตัวอย่างผลลัพธ์](#-ภาพตัวอย่างผลลัพธ์)
- [โครงสร้างโปรเจกต์](#-โครงสร้างโปรเจกต์)
- [การทำงานของระบบ](#-การทำงานของระบบ)
- [ความต้องการของระบบ](#-ความต้องการของระบบ)
- [การติดตั้ง](#-การติดตั้ง)
- [การเตรียม Dataset](#-การเตรียม-dataset)
- [การ Train โมเดล](#-การ-train-โมเดล)
- [การ Detect และแสดง Heatmap](#-การ-detect-และแสดง-heatmap)
- [การอ่านผลลัพธ์](#-การอ่านผลลัพธ์)
- [Parameter ที่ปรับได้](#-parameter-ที่ปรับได้)

---

## 🖼️ ภาพตัวอย่างผลลัพธ์

### ตัวอย่างการทำงานของ Grad-CAM Heatmap

```
┌───────────────────────────────────────────────────────────────┐
│          ภาพ Input              →         ผลลัพธ์              │
├───────────────────────────────┬───────────────────────────────┤
│                               │  ┌──────────────────────────┐ │
│   ┌─────────────────────┐     │  │ Result: NG (97.45%)      │ │
│   │                     │     │  └──────────────────────────┘ │
│   │   ชิ้นงาน / รูปภาพ     │     │                               │
│   │    (ภาพต้นฉบับ)      │ ──▶ │ 🟥🟥🟧🟨🟩🟩🟩🟩🟩🟩🟩  │
│   │                     │     │  🟥🟥🟧🟨🟩🟩🟩🟩🟩🟩🟩  │
│   └─────────────────────┘     │  🟧🟧🟨🟨🟩🟩🟩🟩🟩🟩🟩  │
│                               │  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩  │
│                               │                               │
│                               │  🔴 = บริเวณที่มีความผิดปกติ       │
│                               │  🟢 = บริเวณปกติ                │
└───────────────────────────────┴───────────────────────────────┘
```

### ตัวอย่างผลลัพธ์จริงที่ได้

| กรณี | ภาพต้นฉบับ | ผลลัพธ์ Heatmap | การตัดสินใจ |
|------|-----------|----------------|------------|
| ชิ้นงาน **NG** (มีตำหนิ) | ภาพชิ้นงานที่มีรอยขีดข่วน | พื้นที่สีแดง-ส้มบริเวณรอยตำหนิ | `Result: NG (97.45%)` |
| ชิ้นงาน **OK** (ปกติ) | ภาพชิ้นงานปกติ | พื้นที่สีเขียวกระจายทั่วภาพ | `Result: OK (99.12%)` |

### ตัวอย่าง Output บนหน้าจอ

```
┌──────────────────────────────────────────────────┐
│  Result: NG (97.45%)                             │
│                                                  │
│  ░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│  ░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│  ░░░░░▓▓▓▓████████▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░    │
│  ░░░░░▓▓████████████▓░░░░░░░░░░░░░░░░░░░░░░░░    │
│  ░░░░░░░▓▓▓▓████▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│  ░░░░░░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│                                                  │
│  ████ = จุดที่โมเดลโฟกัส (สีแดง)                     │
│  ▓▓▓▓ = บริเวณรอบข้าง (สีส้ม/เหลือง)                  │
│  ░░░░ = บริเวณปกติ (สีเขียว/น้ำเงิน)                    │
└──────────────────────────────────────────────────┘
```

---

## 📁 โครงสร้างโปรเจกต์

```
Github/
├── 📄 train.py                  # สคริปต์สำหรับ Train โมเดล
├── 📄 detect.py                 # สคริปต์สำหรับ Detect + แสดง Heatmap
├── 🤖 yolo26s-cls.pt            # Pretrained YOLO Classification model
├── 🤖 best2.pt                  # โมเดลที่ Train แล้ว (สร้างหลัง Train)
│
└── 📂 my_dataset/               # Dataset สำหรับ Train
    ├── 📂 train/                # รูปสำหรับ Training (80%)
    │   ├── 📂 OK/               # รูปชิ้นงานปกติ
    │   │   ├── img_001.jpg
    │   │   ├── img_002.jpg
    │   │   └── ...
    │   └── 📂 NG/               # รูปชิ้นงานผิดปกติ
    │       ├── img_001.jpg
    │       ├── img_002.jpg
    │       └── ...
    │
    └── 📂 val/                  # รูปสำหรับ Validation (20%)
        ├── 📂 OK/
        └── 📂 NG/
```

---

## 🔄 การทำงานของระบบ

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ของระบบ                              │
└─────────────────────────────────────────────────────────────────────┘

  [Dataset]          [Training]           [Model]          [Detection]
     │                   │                   │                  │
  ┌──▼──┐           ┌────▼────┐          ┌───▼───┐         ┌───▼───┐
  │ OK/ │──────────▶│         │          │       │         │Input  │
  │ NG/ │  train.py │  YOLO   │─────────▶│best2  │────────▶│Image  │
  └─────┘           │Training │          │  .pt  │         └───────┘
                    └─────────┘          └───────┘             │
                                                               │
                    ┌──────────────────────────────────────────┘
                    │
                    ▼
              ┌──────────┐      ┌──────────────┐      ┌────────────────┐
              │Prediction│─────▶│  Grad-CAM    │─────▶│  Heatmap +    │
              │Class: NG │      │  Heatmap     │      │  Label Output  │
              │Conf: 97% │      │  Generation  │      │  result.jpg    │
              └──────────┘      └──────────────┘      └────────────────┘
```

**ขั้นตอนการทำงานของ `detect.py`:**

```
ขั้นตอนที่ 1: โหลดโมเดล best2.pt
     ↓
ขั้นตอนที่ 2: YOLO Predict → ได้ Class (OK/NG) + Confidence (%)
     ↓
ขั้นตอนที่ 3: เตรียมโมเดลสำหรับ Grad-CAM (เปิด gradient)
     ↓
ขั้นตอนที่ 4: ระบุ Target Layer (layer ก่อนสุดท้าย)
     ↓
ขั้นตอนที่ 5: Forward Pass + Backward Pass → คำนวณ Gradient
     ↓
ขั้นตอนที่ 6: สร้าง Heatmap (Grayscale CAM → Overlay บนรูปต้นฉบับ)
     ↓
ขั้นตอนที่ 7: เขียน Label + Confidence ลงบนภาพ
     ↓
ขั้นตอนที่ 8: แสดงผลและบันทึกเป็น result_with_label.jpg
```

---

## ⚙️ ความต้องการของระบบ

| รายการ | ความต้องการขั้นต่ำ | แนะนำ |
|--------|-----------------|-------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| GPU | ไม่จำเป็น (ใช้ CPU ได้) | NVIDIA GPU (CUDA) |
| Storage | 1 GB | 5 GB+ |
| OS | Windows / Linux / macOS | Windows 10/11 |

---

## 📦 การติดตั้ง

### 1. Clone หรือ Download โปรเจกต์

```bash
git clone <repository-url>
cd Github
```

### 2. สร้าง Virtual Environment (แนะนำ)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# หรือ
source venv/bin/activate      # Linux/macOS
```

### 3. ติดตั้ง Dependencies

```bash
pip install ultralytics
pip install grad-cam
pip install opencv-python
pip install torch torchvision
```

หรือติดตั้งทั้งหมดพร้อมกัน:

```bash
pip install ultralytics grad-cam opencv-python torch torchvision
```

### 4. ตรวจสอบการติดตั้ง

```python
import ultralytics
import pytorch_grad_cam
import cv2
import torch
print("✅ ติดตั้งสำเร็จ!")
```

---

## 🗂️ การเตรียม Dataset

จัดโครงสร้างโฟลเดอร์ดังนี้:

```
my_dataset/
├── train/
│   ├── OK/          ← รูปชิ้นงานปกติ สำหรับ Training
│   │   ├── ok_001.jpg
│   │   ├── ok_002.jpg
│   │   └── ...
│   └── NG/          ← รูปชิ้นงานผิดปกติ สำหรับ Training
│       ├── ng_001.jpg
│       ├── ng_002.jpg
│       └── ...
└── val/
    ├── OK/          ← รูปชิ้นงานปกติ สำหรับ Validation
    └── NG/          ← รูปชิ้นงานผิดปกติ สำหรับ Validation
```

**📌 หลักการแบ่งรูปภาพ:**
- Training Set: **80%** ของรูปทั้งหมด
- Validation Set: **20%** ของรูปทั้งหมด
- รูปแต่ละคลาสควรมีจำนวนใกล้เคียงกัน (Balanced Dataset)
- ขนาดรูปที่แนะนำ: อย่างน้อย **224×224 pixels**

---

## 🏋️ การ Train โมเดล

### แก้ไขค่าใน `train.py`

```python
from ultralytics import YOLO

# 1. โหลด Pretrained YOLO Classification Model
model = YOLO('yolo26s-cls.pt')

# 2. เริ่ม Training
results = model.train(
    data='my_dataset',          # 📁 Path ของโฟลเดอร์ dataset
    epochs=50,                   # 🔄 จำนวนรอบการเทรน (เพิ่มได้ถ้าต้องการความแม่นยำสูงขึ้น)
    imgsz=224,                   # 📐 ขนาดรูปภาพ (224 คือมาตรฐาน)
    batch=2,                     # 📦 Batch size (เพิ่มได้ถ้า GPU มี VRAM มาก)
    project='my_classification', # 📂 ชื่อโฟลเดอร์ผลลัพธ์
    name='experiment_1'          # 🏷️ ชื่อการทดลอง
)

# 3. ตรวจสอบ Accuracy
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")
```

### รันสคริปต์

```bash
python train.py
```

### ผลลัพธ์หลัง Training

```
my_classification/
└── experiment_1/
    ├── weights/
    │   ├── best.pt      ← โมเดลที่ดีที่สุด (ใช้ไฟล์นี้!)
    │   └── last.pt      ← โมเดลล่าสุด
    ├── results.png      ← กราฟ Training/Validation Loss & Accuracy
    └── confusion_matrix.png
```

> **💡 หมายเหตุ:** หลัง Train เสร็จ ให้ copy `best.pt` มาไว้ในโฟลเดอร์หลัก และเปลี่ยนชื่อเป็น `best2.pt` หรือแก้ไข path ใน `detect.py`

---

## 🔎 การ Detect และแสดง Heatmap

### แก้ไขค่าใน `detect.py`

```python
# ── ส่วนที่ต้องแก้ไข ──────────────────────────────────────────
model_path = 'best2.pt'    # 🤖 Path ของโมเดลที่ Train แล้ว
img_path   = 'NG.jpg'      # 🖼️ Path ของรูปที่ต้องการตรวจสอบ
# ─────────────────────────────────────────────────────────────
```

### รันสคริปต์

```bash
python detect.py
```

### ผลลัพธ์ที่ได้

```
📺 หน้าต่าง "Anomaly Heatmap with Label" จะเปิดขึ้น แสดง:
   ┌──────────────────────────────────────────┐
   │  Result: NG (97.45%)                     │  ← ชื่อคลาส + ความมั่นใจ
   │                                          │
   │         [ Heatmap Overlay ]              │
   │    🔴🔴🔴🔴🔴🔴                       │  ← บริเวณผิดปกติ (สีแดง)
   │    🔴🔴🟠🟠🟡🟡🟢🟢                  │
   │    🟠🟠🟡🟡🟢🟢🟢🟢                  │
   │    🟡🟡🟢🟢🟢🟢🟢🟢                  │  ← บริเวณปกติ (สีเขียว)
   │    🟢🟢🟢🟢🟢🟢🟢🟢                  │
   │                                          │
   └──────────────────────────────────────────┘

💾 บันทึกผลลัพธ์เป็นไฟล์: result_with_label.jpg
```

---

## 📊 การอ่านผลลัพธ์

### ความหมายของสีใน Heatmap

```
สีบน Heatmap            ความหมาย                     ระดับ
─────────────────────────────────────────────────────────────
🔴  แดงเข้ม           บริเวณที่โมเดลสนใจมากที่สุด      สูงมาก
🟠  ส้ม               บริเวณที่โมเดลสนใจมาก             สูง
🟡  เหลือง            บริเวณที่โมเดลสนใจปานกลาง         กลาง
🟢  เขียว             บริเวณที่โมเดลสนใจน้อย             ต่ำ
🔵  น้ำเงิน           บริเวณที่โมเดลไม่สนใจ              ต่ำมาก
```

### ตัวอย่างการแปลผล

| ผลลัพธ์ | ความหมาย | การดำเนินการ |
|---------|---------|------------|
| `Result: OK (99%)` + สีเขียวกระจาย | ชิ้นงานปกติ ไม่มีตำหนิ | ✅ ผ่าน |
| `Result: NG (95%)` + จุดแดงชัดเจน | ชิ้นงานมีตำหนิ ระบุตำแหน่งได้ | ❌ ไม่ผ่าน |
| `Result: NG (55%)` + สีกระจัดกระจาย | โมเดลไม่มั่นใจ ควร Inspect ด้วยตา | ⚠️ ตรวจสอบอีกครั้ง |

---

## 🎛️ Parameter ที่ปรับได้

### `train.py`

| Parameter | ค่าเริ่มต้น | คำอธิบาย |
|-----------|-----------|---------|
| `epochs` | `50` | จำนวนรอบการ Train (เพิ่มเพื่อความแม่นยำ) |
| `imgsz` | `224` | ขนาดรูปภาพ input (pixel) |
| `batch` | `2` | จำนวนรูปต่อ batch (เพิ่มถ้า RAM/VRAM พอ) |
| `project` | `'my_classification'` | ชื่อโฟลเดอร์ผลลัพธ์ |
| `name` | `'experiment_2'` | ชื่อการทดลอง (สร้าง sub-folder) |

### `detect.py`

| Parameter | ค่าเริ่มต้น | คำอธิบาย |
|-----------|-----------|---------|
| `model_path` | `'best2.pt'` | Path ของโมเดลที่ Train แล้ว |
| `img_path` | `'NG.jpg'` | Path ของรูปที่ต้องการตรวจสอบ |
| `target_layers` | `model[-2]` | Layer ที่ใช้ดึง Gradient (ปกติใช้ layer ก่อนสุดท้าย) |
| `imgsz resize` | `(224, 224)` | ขนาดที่ resize ก่อนส่งเข้าโมเดล |

---

## 🛠️ การแก้ปัญหาที่พบบ่อย

**❌ `RuntimeError: element 0 of tensors does not require grad`**
```python
# เพิ่ม mode ก่อนทำ Grad-CAM
model.model.train()
for param in model.model.parameters():
    param.requires_grad = True
```

**❌ `FileNotFoundError: best2.pt`**
```bash
# ตรวจสอบ path และชื่อไฟล์โมเดล
# โมเดลจะอยู่ที่: my_classification/experiment_N/weights/best.pt
```

**❌ รูป Heatmap ดำทั้งหมด**
```python
# ลอง target layer อื่น เช่น layer ก่อนหน้า
target_layers = [model.model.model[-3]]  # ลองเลื่อน index
```

---

## 📚 เทคโนโลยีที่ใช้

| Library | เวอร์ชัน | การใช้งาน |
|---------|---------|---------|
| [Ultralytics YOLO](https://docs.ultralytics.com/) | 8.x | Classification Model |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | latest | Grad-CAM Heatmap |
| [PyTorch](https://pytorch.org/) | 2.x | Deep Learning Framework |
| [OpenCV](https://opencv.org/) | 4.x | Image Processing & Display |
| [NumPy](https://numpy.org/) | 1.x | Array Operations |

---

<p align="center">
  Made with ❤️ for Quality Control & Anomaly Detection
</p>
