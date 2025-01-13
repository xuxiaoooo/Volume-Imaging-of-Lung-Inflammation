from pathlib import Path
import torch

# 基础路径
BASE_DIR = Path('/Users/xuxiao/Code/LIIV')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
LABELED_DIR = DATA_DIR / 'labeled'
PREDICTIONS_DIR = DATA_DIR / 'predictions'
RESULTS_DIR = DATA_DIR / 'results'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'  # 添加检查点目录

# 预处理参数
PREPROCESSING = {
    'target_size': (256, 256),
    'window_center': 40,
    'window_width': 400,
    'hu_threshold': -300,
}

# 分割模型参数
SEGMENTATION = {
    'batch_size': 2,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 2,
    'input_channels': 1,
}

# 分类模型参数
CLASSIFICATION = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'num_classes': 2,  # 有炎症/无炎症
    'threshold': 0.3,  # 炎症体积比阈值
}

# 可视化参数
VISUALIZATION = {
    'figsize': (12, 8),
    'dpi': 100,
    'cmap': 'viridis',
}

# 分析参数
ANALYSIS = {
    'volume_threshold': 0.1,  # 最小有效体积比例
    'slice_spacing': 3,  # mm
    'density_bins': 100,
}
