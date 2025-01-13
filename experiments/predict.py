import torch
import logging
from pathlib import Path
from datetime import datetime

from models import UNet
from utils import SegmentationPredictor, VolumeClassifier
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict():
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = RESULTS_DIR / f'prediction_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    
    logger.info("开始预测过程...")
    
    # 加载模型
    model = UNet(
        n_channels=SEGMENTATION['input_channels'],
        n_classes=SEGMENTATION['num_classes']
    )
    
    # 加载最新的检查点
    checkpoints = list(CHECKPOINTS_DIR.glob('*.pth'))
    if not checkpoints:
        raise FileNotFoundError("未找到模型检查点！")
    
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"加载模型检查点: {latest_checkpoint}")
    
    # 创建预测器
    predictor = SegmentationPredictor(
        model=model,
        config=SEGMENTATION
    )
    
    # 预测所有病人
    for patient_dir in PROCESSED_DATA_DIR.iterdir():
        if not patient_dir.is_dir():
            continue
        
        logger.info(f"处理病人: {patient_dir.name}")
        
        # 预测分割掩码
        results = predictor.predict_directory(
            input_dir=patient_dir,
            output_dir=PREDICTIONS_DIR / patient_dir.name,
            metadata_file=patient_dir / 'metadata.json',
            save_visualizations=True
        )
    
    # 分类
    classifier = VolumeClassifier(CLASSIFICATION)
    predictions = classifier.classify_dataset(
        predictions_dir=PREDICTIONS_DIR,
        output_file=RESULTS_DIR / f'classification_results_{timestamp}.csv'
    )
    
    logger.info(f"预测完成！分类结果已保存到: {RESULTS_DIR}/classification_results_{timestamp}.csv")

if __name__ == '__main__':
    predict()
