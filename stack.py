import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

# 输入和输出路径
input_dir = "/home/kevin-zhou/Downloads/interval/origin"
output_path = "/home/kevin-zhou/Downloads/interval/stacked.jpg"

# 获取图像列表
image_files = sorted(glob.glob(os.path.join(input_dir, "*.JPG")))
if not image_files:
    raise RuntimeError("❌ 没有找到图片，请确认路径和文件格式")

print(f"📸 找到 {len(image_files)} 张图像，开始最大值堆栈...\n")

# 初始化堆栈图像
accumulator = cv2.imread(image_files[0]).astype(np.float32)

# 遍历图像执行最大值堆栈
for path in tqdm(image_files[1:], desc="最大值堆栈", unit="张"):
    img = cv2.imread(path).astype(np.float32)
    accumulator = np.maximum(accumulator, img)

# 保存结果图像
cv2.imwrite(output_path, accumulator.astype(np.uint8))
print(f"✅ 最大值堆栈完成，保存至: {output_path}")
