import os
import pydicom
import numpy as np
from PIL import Image
import re
import shutil

def get_patient_name(folder_name):
    """从文件夹名称中提取病人姓名"""
    return folder_name.replace('-DCM', '').replace('-JPG', '')

def sample_indices(total_length, target_samples=50):
    """计算等间隔采样的索引"""
    if total_length <= target_samples:
        return list(range(total_length))
    
    # 计算采样间隔
    step = (total_length - 1) / (target_samples - 1)
    indices = [int(round(i * step)) for i in range(target_samples)]
    return sorted(list(set(indices)))  # 去重并排序

def organize_jpg_files(input_dir, output_dir):
    """整理已有的JPG文件，等间隔采样50张"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有JPG文件
    jpg_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    
    # 按系统默认名称排序
    jpg_files.sort()
    
    # 获取采样索引
    sample_idx = sample_indices(len(jpg_files))
    sampled_files = [jpg_files[i] for i in sample_idx]
    
    # 获取病人名字
    folder_name = os.path.basename(input_dir)
    patient_name = get_patient_name(folder_name)
    patient_dir = os.path.join(output_dir, patient_name)
    
    # 创建病人目录
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    
    # 重命名并移动文件
    print(f"\n处理病人 {patient_name} 的图片:")
    print(f"总图片数: {len(jpg_files)}, 采样数: {len(sampled_files)}")
    
    for i, jpg_file in enumerate(sampled_files, 1):
        new_name = f"{i}.jpg"
        output_path = os.path.join(patient_dir, new_name)
        shutil.copy2(jpg_file, output_path)
        print(f"已整理: {jpg_file} -> {output_path}")

def convert_dcm_to_jpg(input_dir, output_dir):
    """转换DCM文件为JPG，等间隔采样50张"""
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
    
    # 获取采样索引
    sample_idx = sample_indices(len(dcm_files))
    sampled_files = [dcm_files[i] for i in sample_idx]
    
    # 获取病人名字
    folder_name = os.path.basename(input_dir)
    patient_name = get_patient_name(folder_name)
    patient_dir = os.path.join(output_dir, patient_name)
    
    # 创建病人目录
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    
    # 转换并重命名
    print(f"\n处理病人 {patient_name} 的DCM文件:")
    print(f"总文件数: {len(dcm_files)}, 采样数: {len(sampled_files)}")
    
    for i, dcm_file in enumerate(sampled_files, 1):
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
            output_path = os.path.join(patient_dir, f"{i}.jpg")
            img.save(output_path, quality=95)
            print(f"已转换: {dcm_file} -> {output_path}")
            
        except Exception as e:
            print(f"处理文件 {dcm_file} 时出错: {str(e)}")

if __name__ == "__main__":
    # 基础目录
    base_dir = "/Users/xuxiao/Code/LIIV/data"
    output_base_dir = "/Users/xuxiao/Code/LIIV/data/processed"
    
    # 遍历所有文件夹
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path) or folder == "processed":
            continue
            
        if folder.endswith('-DCM'):
            # 处理DCM文件
            convert_dcm_to_jpg(folder_path, output_base_dir)
        elif folder.endswith('-JPG'):
            # 处理已有的JPG文件
            organize_jpg_files(folder_path, output_base_dir)
