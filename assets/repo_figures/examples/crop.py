from PIL import Image
import os

def downsample_images(directory):
    # 遍历给定路径中的所有子目录和文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):  # 确保是jpg图片
                file_path = os.path.join(root, file)
                
                # 打开图片并进行降采样
                with Image.open(file_path) as img:
                    # 计算新尺寸为原来的1/4
                    new_size = (img.width // 4, img.height // 4)
                    img_resized = img.resize(new_size, Image.LANCZOS)
                    
                    # 覆盖保存图片
                    img_resized.save(file_path)
                print(f"Processed: {file_path}")

# 使用路径调用函数
directory_path = "/data1/wjs/RFSolver_release/assets/repo_figures/examples"  # 替换为你的图片根目录路径
downsample_images(directory_path)
