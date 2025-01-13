import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LungDataset(Dataset):
    def __init__(self, image_dir, label_file=None, transform=None):
        """
        Args:
            image_dir: 图像目录路径
            label_file: 标注文件路径（可选）
            transform: 图像转换
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.labels = None
        
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as f:
                data = json.load(f)
                self.labels = self._process_via_labels(data)
    
    def _process_via_labels(self, data):
        """处理VIA标注格式"""
        labels = {}
        for img_key, img_data in data['_via_img_metadata'].items():
            filename = img_data['filename']
            lung_points = []
            inflammation_points = []
            
            for region in img_data['regions']:
                attrs = region['region_attributes']
                point = region['shape_attributes']
                x, y = point['cx'], point['cy']
                
                # 检查标注类型
                if 'lung' in attrs and isinstance(attrs['lung'], dict) and attrs['lung'].get('a'):
                    lung_points.append([x, y])
                if 'i' in attrs and isinstance(attrs['i'], dict) and attrs['i'].get('b'):
                    inflammation_points.append([x, y])
            
            labels[filename] = {
                'lung': np.array(lung_points),
                'inflammation': np.array(inflammation_points)
            }
        return labels
    
    def _create_mask(self, points, size=(256, 256)):
        """从点创建掩码"""
        if len(points) == 0:
            return np.zeros(size)
        
        mask = np.zeros(size)
        for x, y in points:
            if 0 <= x < size[0] and 0 <= y < size[1]:
                mask[int(y), int(x)] = 1
        return mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None and img_name in self.labels:
            label_data = self.labels[img_name]
            lung_mask = self._create_mask(label_data['lung'])
            inflammation_mask = self._create_mask(label_data['inflammation'])
            return {
                'image': image,
                'lung_mask': torch.FloatTensor(lung_mask),
                'inflammation_mask': torch.FloatTensor(inflammation_mask)
            }
        
        return {'image': image}

def get_data_loaders(image_dir, label_file=None, batch_size=4, train_split=0.8):
    """获取数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = LungDataset(image_dir, label_file, transform)
    
    # 划分训练集和验证集
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
