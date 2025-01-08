import os
import pydicom
import numpy as np
from PIL import Image
import re

def convert_dcm_to_jpg(input_dir, output_dir):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有DCM文件
    dcm_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dcm_files.append(os.path.join(root, file))
    
    # 提取文件名中的数字并排序
    dcm_files.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    
    # 转换并重命名
    for i, dcm_file in enumerate(dcm_files, 1):
        try:
            # 读取DICOM文件
            ds = pydicom.dcmread(dcm_file)
            
            # 转换为numpy数组
            img_array = ds.pixel_array
            
            # 归一化到0-255
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # 转换为PIL图像
            img = Image.fromarray(img_array)
            
            # 保存为jpg
            output_path = os.path.join(output_dir, f"{i}.jpg")
            img.save(output_path, quality=95)
            print(f"已转换: {dcm_file} -> {output_path}")
            
        except Exception as e:
            print(f"处理文件 {dcm_file} 时出错: {str(e)}")

if __name__ == "__main__":
    # 基础目录
    base_dir = "/Users/xuxiao/Code/Lung Imaging/data"
    
    # 遍历所有包含DCM的文件夹
    for folder in os.listdir(base_dir):
        if folder.endswith('-DCM'):
            input_dir = os.path.join(base_dir, folder)
            # 创建对应的输出目录（将-DCM改为-JPG）
            output_folder = folder.replace('-DCM', '-JPG')
            output_dir = os.path.join(base_dir, output_folder)
            
            print(f"\n处理文件夹: {folder}")
            convert_dcm_to_jpg(input_dir, output_dir)
