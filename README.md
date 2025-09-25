# ไลบรารีสร้างภาพ/แก้ไข/รีทัช สำหรับ Python

## 0) เตรียมสภาพแวดล้อม (ครั้งเดียว)

```bash
# Windows
python -m venv .venv && .\.venv\Scripts\activate
# macOS/Linux
python3 -m venv .venv && source .venv/bin/activate

python -m pip install -U pip setuptools wheel
```

> แนะนำ Python 3.10–3.11 บน venv/conda และถ้ามี GPU NVIDIA ให้ติดตั้งไดรเวอร์+CUDA ให้พร้อมเพื่อเร่งงานโมเดลเชิงลึก

---

## 1) Core I/O/Editing (พื้นฐานที่ควรมี)

```bash
pip install numpy pillow opencv-python scikit-image imageio matplotlib
```

ทดสอบเร็ว:

```bash
python - << 'PY'
from PIL import Image, ImageFilter
im = Image.new('RGB',(512,512),'#222')
im = im.filter(ImageFilter.GaussianBlur(4))
im.save('test_core.png')
print('saved: test_core.png')
PY
```

---

## 2) Generative (Text-to-Image / Inpainting / Image-to-Image)

**CPU เท่านั้น (ติดตั้งง่ายสุด):**

```bash
pip install diffusers[torch] transformers accelerate safetensors
```

**GPU NVIDIA (ติดตั้ง PyTorch ให้ตรงกับ CUDA ก่อน):**

```bash
pip install diffusers transformers accelerate safetensors xformers
```

ทดสอบ Text-to-Image:

```bash
python - << 'PY'
from diffusers import StableDiffusionPipeline
import torch
m = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(m, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
if torch.cuda.is_available():
    pipe = pipe.to('cuda')
img = pipe("a cute corgi playing guitar, high detail, soft lighting").images[0]
img.save('sd_t2i_sample.png')
print('saved: sd_t2i_sample.png')
PY
```

> โมเดลบางตัวต้องยอมรับ license บน Hugging Face ก่อนดาวน์โหลดครั้งแรก

---

## 3) Retouch/Enhance (อัปสเกล, หน้าเนียน, คืนความละเอียด)

```bash
pip install realesrgan gfpgan basicsr facexlib insightface
```

ตัวอย่างอัปสเกล + ใบหน้าชัด:

```bash
python - << 'PY'
from PIL import Image
from realesrgan import RealESRGAN
import torch

model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=2)
model.load_weights('RealESRGAN_x2.pth')
img = Image.open('input.jpg').convert('RGB')
up = model.predict(img)
up.save('upscaled_x2.png')
print('saved: upscaled_x2.png')
PY
```

> ครั้งแรกจะดาวน์โหลด weight; ตรวจสอบ repo ของ realesrgan/gfpgan หากต้องการไฟล์อื่น

---

## 4) Background Removal / Segmentation / Matting

```bash
pip install rembg opencv-contrib-python
```

ลบฉากหลังจาก CLI:

```bash
rembg i input.png output.png
```

หรือใช้งานผ่าน Python:

```bash
python - << 'PY'
from rembg import remove
from PIL import Image
im = Image.open('input.png')
out = remove(im)
out.save('no_bg.png')
print('saved: no_bg.png')
PY
```

**ตัวแบ่งส่วน/ตรวจจับยอดนิยม:**

```bash
pip install ultralytics
```

ทดสอบ YOLOv8 แบบเร็ว:

```bash
yolo predict model=yolov8n-seg.pt source=input.jpg save=True
```

> ต้องการใช้ SAM ให้ดาวน์โหลดโมเดลและชี้พาธไฟล์น้ำหนัก (vit_h/vit_l เป็นต้น)

---

## 5) เครื่องมือเสริมที่มีประโยชน์

```bash
pip install wand
pip install moviepy
pip install imgaug albumentations
pip install onnxruntime-gpu
```

---

## 6) ชุดคำสั่งติดตั้งรวดเดียว

### A) แก้ไข/รีทัชทั่วไป

```bash
pip install -U numpy pillow opencv-python scikit-image imageio matplotlib rembg
```

### B) Stable Diffusion (CPU/GPU)

```bash
# พื้นฐาน + diffusers
a) pip install -U numpy pillow opencv-python imageio scikit-image matplotlib safetensors accelerate transformers
b) pip install -U diffusers[torch]
# ถ้าใช้ GPU ให้ติดตั้ง torch ตามเว็บ PyTorch แล้วรัน: pip install diffusers xformers
```

### C) รีทัช/อัปสเกลใบหน้า

```bash
pip install -U basicsr facexlib gfpgan realesrgan insightface
```

### D) แยกฉาก/ตัดวัตถุเร็ว

```bash
pip install -U rembg ultralytics opencv-contrib-python
```

---

## 7) สคริปต์ pipeline ตัวอย่าง (`pipeline_demo.py`)

```python
from PIL import Image
from rembg import remove
from realesrgan import RealESRGAN
from diffusers import StableDiffusionPipeline
import torch

# 1) ลบฉากหลัง
img = Image.open('input.png').convert('RGB')
no_bg = remove(img)
no_bg.save('step1_no_bg.png')

# 2) อัปสเกล x2 (ต้องมีไฟล์น้ำหนัก RealESRGAN_x2.pth)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
up = RealESRGAN(device, scale=2)
up.load_weights('RealESRGAN_x2.pth')
up_img = up.predict(no_bg)
up_img.save('step2_upscaled.png')

# 3) สร้างภาพจากข้อความ
model_id = 'runwayml/stable-diffusion-v1-5'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
if torch.cuda.is_available():
    pipe = pipe.to('cuda')
result = pipe('a portrait photo of a young woman, soft light, realistic, 35mm, bokeh')
result.images[0].save('step3_t2i.png')
print('done: step1_no_bg.png, step2_upscaled.png, step3_t2i.png')
```

รัน:

```bash
python pipeline_demo.py
```

---

## 8) เคล็ดลับ

* รันบน CPU ให้ทำงานได้ก่อน ถ้าใช้ GPU ตรวจสอบ CUDA/ไดรเวอร์
* ไฟล์ weight จะ cache ครั้งแรกแล้วใช้ซ้ำได้
* ระบุพาธ weight เช่น `RealESRGAN_x2.pth`, `GFPGANv1.4.pth` ให้ถูกต้อง
* ตรวจสอบ license โมเดลก่อนใช้งานเชิงพาณิชย์

---

## 9) สรุปแพ็กเกจสำคัญ

* **พื้นฐาน:** numpy, pillow, opencv-python, scikit-image, imageio, matplotlib
* **สร้างภาพ:** diffusers, transformers, accelerate, (torch/xformers สำหรับ GPU)
* **รีทัช:** realesrgan, gfpgan, basicsr, insightface, facexlib
* **ลบฉากหลัง:** rembg, ultralytics, opencv-contrib-python
* **เสริม:** wand, moviepy, albumentations, onnxruntime-gpu

> ต้องการเวอร์ชัน Node.js หรือไฟล์ requirements.txt แจ้งได้
