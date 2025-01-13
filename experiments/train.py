import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from datetime import datetime
from torchvision import transforms

from dataset import LungDataset
from models import UNet
from utils import SegmentationTrainer
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = RESULTS_DIR / f'training_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    
    logger.info("开始训练过程...")
    
    # 创建数据加载器
    dataset = LungDataset(
        str(PROCESSED_DIR / '崔连祥'),  # 指定崔连祥的图像目录
        str(LABELED_DIR / 'cuilianxiang.json'),  # 使用崔连祥的标注文件
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=SEGMENTATION['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=SEGMENTATION['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
    
    # 创建模型
    model = UNet(
        n_channels=SEGMENTATION['input_channels'],
        n_classes=SEGMENTATION['num_classes']
    )
    
    # 创建训练器
    trainer = SegmentationTrainer(
        model=model,
        config=SEGMENTATION
    )
    
    # 训练模型
    best_model_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=CHECKPOINTS_DIR
    )
    
    logger.info(f"训练完成！最佳模型保存在: {best_model_path}")

if __name__ == '__main__':
    train()
