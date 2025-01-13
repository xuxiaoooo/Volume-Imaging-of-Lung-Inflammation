import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from models import CombinedLoss

logger = logging.getLogger(__name__)

class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        
        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            lung_masks = batch['lung_mask'].to(self.device)
            inflammation_masks = batch['inflammation_mask'].to(self.device)
            
            # 合并两种掩码
            masks = torch.stack([lung_masks, inflammation_masks], dim=1)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                lung_masks = batch['lung_mask'].to(self.device)
                inflammation_masks = batch['inflammation_mask'].to(self.device)
                
                # 合并两种掩码
                masks = torch.stack([lung_masks, inflammation_masks], dim=1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader,
        val_loader,
        save_dir: Path
    ) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = save_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, model_path)
                logger.info(f"保存最佳模型到: {model_path}")
        
        return model_path

class SegmentationPredictor:
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict_image(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        # 预处理
        if image.shape != self.config['target_size']:
            image = cv2.resize(image, self.config['target_size'])
        
        # 归一化
        image = image / 255.0
        
        # 转换为tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 二值化
        pred_mask = (pred_mask > threshold).astype(np.uint8)
        
        return pred_mask
    
    def predict_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        metadata_file: Optional[Path] = None,
        save_visualizations: bool = True
    ) -> List[Dict]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取元数据
        metadata = None
        if metadata_file and metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        results = []
        image_files = sorted(list(input_dir.glob('*.jpg')))
        
        for image_file in tqdm(image_files, desc="Predicting"):
            # 读取图像
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"无法读取图像: {image_file}")
                continue
            
            # 预测
            pred_mask = self.predict_image(image)
            
            # 保存结果
            mask_path = output_dir / f"mask_{image_file.name}"
            cv2.imwrite(str(mask_path), pred_mask * 255)
            
            # 可视化
            if save_visualizations:
                vis_path = output_dir / f"vis_{image_file.name}"
                self._save_visualization(image, pred_mask, vis_path)
            
            # 记录结果
            result = {
                'image_file': str(image_file),
                'mask_file': str(mask_path),
                'area': float(np.sum(pred_mask))
            }
            
            # 添加元数据
            if metadata:
                for item in metadata:
                    if item['filename'] == image_file.name:
                        result['spatial_info'] = item['spatial_info']
                        break
            
            results.append(result)
        
        # 保存结果
        with open(output_dir / 'prediction_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: Path
    ):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('预测掩码')
        plt.axis('off')
        
        plt.subplot(133)
        overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
        overlay[..., 0] = image
        overlay[..., 1] = image
        overlay[..., 2] = image
        overlay[mask > 0] = [255, 0, 0]
        plt.imshow(overlay)
        plt.title('叠加显示')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class VolumeClassifier:
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config['threshold']
    
    def calculate_volume_ratio(self, prediction_results: List[Dict]) -> float:
        total_volume = 0
        inflammation_volume = 0
        
        for result in prediction_results:
            if 'spatial_info' in result:
                spatial_info = result['spatial_info']
                pixel_area = spatial_info['pixel_spacing'][0] * spatial_info['pixel_spacing'][1]
                slice_thickness = spatial_info['slice_thickness']
                
                area = result['area']
                volume = area * pixel_area * slice_thickness
                
                inflammation_volume += volume
                total_volume += volume * (area > 0)
        
        return inflammation_volume / total_volume if total_volume > 0 else 0
    
    def classify_patient(self, prediction_results: List[Dict]) -> Dict:
        ratio = self.calculate_volume_ratio(prediction_results)
        label = 1 if ratio > self.threshold else 0
        
        return {
            'volume_ratio': ratio,
            'label': label,
            'class_name': '有炎症' if label else '无炎症'
        }
    
    def classify_dataset(
        self,
        predictions_dir: Path,
        output_file: Optional[Path] = None
    ) -> pd.DataFrame:
        predictions_dir = Path(predictions_dir)
        results = []
        
        for patient_dir in predictions_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            pred_file = patient_dir / 'prediction_results.json'
            if not pred_file.exists():
                continue
            
            with open(pred_file, 'r') as f:
                prediction_results = json.load(f)
            
            classification = self.classify_patient(prediction_results)
            results.append({
                'patient_id': patient_dir.name,
                **classification
            })
        
        df = pd.DataFrame(results)
        
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df
