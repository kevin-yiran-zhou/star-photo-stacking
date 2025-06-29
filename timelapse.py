import cv2
import os
import glob
from tqdm import tqdm

# === 参数设置 ===
input_dir = "/home/kevin-zhou/Downloads/interval/origin"  # 图像序列路径
output_path = "/home/kevin-zhou/Downloads/interval/timelapse_enhanced.mp4"  # 输出视频路径
fps = 8  # << 你可以改为 30、60 等

# 获取图像序列
image_files = sorted(glob.glob(os.path.join(input_dir, "*.JPG")))
if not image_files:
    raise RuntimeError("❌ 没有找到图片，请确认路径和格式是否正确")

# 获取图像尺寸
sample_img = cv2.imread(image_files[0])
height, width, _ = sample_img.shape

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"📽️ 开始生成增强版视频，共 {len(image_files)} 张图，每秒 {fps} 帧...\n")

# === 每帧增强：银河提亮、对比增强 ===
for path in tqdm(image_files, desc="写入帧", unit="帧"):
    img = cv2.imread(path)

    # 增强亮度 & 银河细节
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 提升整体亮度和对比度（你可以调整 alpha/beta）
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

    video_writer.write(enhanced)

video_writer.release()
print(f"✅ 视频生成完成，保存至: {output_path}")
