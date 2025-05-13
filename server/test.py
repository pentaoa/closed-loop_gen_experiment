from PIL import Image
import os
import glob

def crop_to_square_and_resize(input_path, output_path, target_size=(1024, 1024)):
    """
    将图片裁剪为正方形并调整大小到目标尺寸
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸，默认为(1024, 1024)
    """
    try:
        # 打开图片
        img = Image.open(input_path)
        
        # 确定裁剪区域（居中裁剪）
        width, height = img.size
        
        # 选择较小的边作为正方形边长
        size = min(width, height)
        
        # 计算裁剪区域的左上角坐标
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        # 裁剪图片
        img_cropped = img.crop((left, top, right, bottom))
        
        # 调整图片大小
        img_resized = img_cropped.resize(target_size, Image.LANCZOS)
        
        # 保存图片
        img_resized.save(output_path)
        
        return True
    
    except Exception as e:
        print(f"处理图片 {input_path} 时出错: {e}")
        return False

def process_directory(input_dir, output_dir=None, target_size=(1024, 1024)):
    """
    处理指定目录中的所有图片
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录，如果为None则与输入目录相同
        target_size: 目标尺寸，默认为(1024, 1024)
    """
    # 如果输出目录未指定，则使用输入目录
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    # 处理每个图片
    success_count = 0
    for image_file in image_files:
        filename = os.path.basename(image_file)
        output_path = os.path.join(output_dir, f"square_{filename}")
        
        if crop_to_square_and_resize(image_file, output_path, target_size):
            success_count += 1
    
    return success_count, len(image_files)

# 主函数
if __name__ == "__main__":
    input_directory = "/mnt/dataset0/xkp/closed-loop/image_pool"
    output_directory = "/mnt/dataset0/xkp/closed-loop/image_pool_square"
    
    print(f"开始处理图片...")
    success, total = process_directory(input_directory, output_directory)
    print(f"处理完成! 成功转换 {success}/{total} 张图片")
    print(f"转换后的图片保存在: {output_directory}")