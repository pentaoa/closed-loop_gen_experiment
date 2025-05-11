import os
import random
image_set_path = 'stimuli_SX'  # 图像集路径

# 获取所有图片
test_images = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
print(f"一共 {len(test_images)} 张图片")

# 构建图像路径列表
test_images_path = [os.path.join(image_set_path, test_image) for test_image in test_images]

# 从未处理的图片中随机选择10张
available_paths = [path for path in test_images_path]    
sample_image_paths = sorted(random.sample(available_paths, 10))

print("随机选择的10张图片路径:")
for path in sample_image_paths:
    print(path)
