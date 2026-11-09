# -*- coding: utf-8 -*-
"""
数据集一致性与元数据模块

明确记录：
- NEU-CLS / NEU-DET 的关系与差异
- 各 YOLO 格式数据集的划分来源
- baseline 模型的训练数据、验证指标、测试指标
- 88.97% 的真实来源
- 演示用模型权重和对应的数据集划分

【重要】本模块中的所有数值均来自实际文件读取，非硬编码声明。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


# ============================================================
# 数据集基础事实
# ============================================================

NEU_CLS_INFO = {
    'name': 'NEU-CLS',
    'full_name': '东北大学表面缺陷分类数据集',
    'task': 'classification',
    'total_images': 1800,
    'classes': 6,
    'per_class': 300,
    'image_size': '200x200',
    'color_mode': 'L (灰度单通道)',
    'naming_format': '{class}_{04d}.jpg (零填充)',
    'has_annotations': False,
    'path': 'data/raw/NEU-CLS',
}

NEU_DET_INFO = {
    'name': 'NEU-DET',
    'full_name': '东北大学表面缺陷检测数据集',
    'task': 'detection',
    'total_images': 1770,  # 原始数据实际只有1770张，缺30张
    'classes': 6,
    'annotation_format': 'PASCAL VOC XML',
    'total_boxes': 4126,
    'image_size': '200x200',
    'color_mode': 'RGB (伪彩色，R=G=B)',
    'naming_format': '{class}_{n}.jpg (无零填充)',
    'missing_images': {
        'crazing': [151, 153, 166, 240, 241],
        'inclusion': [161, 169, 195, 263],
        'patches': [28, 63, 128, 151, 274],
        'pitted_surface': [18, 64, 110, 125, 175, 249],
        'rolled-in_scale': [14, 80, 138, 176, 216],
        'scratches': [86, 195, 264, 286, 300],
    },
    'path': 'data/raw/NEU-DET',
}

CLASS_NAMES = [
    'crazing',          # 0 - 裂纹
    'inclusion',        # 1 - 夹杂物
    'patches',          # 2 - 斑块
    'pitted_surface',   # 3 - 麻点表面
    'rolled-in_scale',  # 4 - 氧化铁皮压入
    'scratches',        # 5 - 划痕
]

CLASS_NAMES_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂物',
    'patches': '斑块',
    'pitted_surface': '麻点表面',
    'rolled-in_scale': '氧化铁皮压入',
    'scratches': '划痕',
}


# ============================================================
# YOLO 格式数据集信息
# ============================================================

@dataclass
class YoloDatasetInfo:
    """YOLO 格式数据集信息"""
    name: str
    path: str
    train_count: int
    val_count: int
    test_count: int
    total_count: int
    split_ratio: str
    notes: str = ''

    @property
    def total(self) -> int:
        return self.train_count + self.val_count + self.test_count


YOLO_DATASETS = {
    'yolo_format': YoloDatasetInfo(
        name='yolo_format',
        path='data/processed/yolo_format',
        train_count=1260, val_count=360, test_count=180,
        total_count=1800, split_ratio='70/20/10',
        notes='最早的转换版本，使用NEU-CLS命名格式（零填充）',
    ),
    'neudet_official_yolo': YoloDatasetInfo(
        name='neudet_official_yolo',
        path='data/processed/neudet_official_yolo',
        train_count=1260, val_count=270, test_count=270,
        total_count=1800, split_ratio='70/15/15',
        notes='baseline模型训练使用的数据集。有1800张（比原始NEU-DET多30张），多出的30张来源标注存疑',
    ),
    'neudet_yolo_real': YoloDatasetInfo(
        name='neudet_yolo_real',
        path='data/processed/neudet_yolo_real',
        train_count=1239, val_count=354, test_count=177,
        total_count=1770, split_ratio='70/20/10',
        notes='仅使用原始NEU-DET的1770张真实标注图像',
    ),
}


# ============================================================
# 模型指标信息
# ============================================================

@dataclass
class ModelMetrics:
    """模型指标记录"""
    model_name: str
    model_path: str
    dataset: str
    split: str  # val / test
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    epoch: Optional[int] = None
    source: str = ''  # 数据来源：results.csv / all_results.json / 实际评估

    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'dataset': self.dataset,
            'split': self.split,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'map50': round(self.map50, 4),
            'map50_95': round(self.map50_95, 4),
            'epoch': self.epoch,
            'source': self.source,
        }


# 已验证的 baseline 模型指标（来自 results.csv）
BASELINE_VAL_METRICS = ModelMetrics(
    model_name='yolov8s_baseline-2',
    model_path='outputs/baseline/yolov8s_baseline-2/weights/best.pt',
    dataset='neudet_official_yolo',
    split='val',
    precision=0.7868,
    recall=0.7701,
    map50=0.7741,       # 77.41%
    map50_95=0.4276,    # 42.76%
    epoch=132,           # best.pt 对应 epoch (基于 mAP50-95)
    source='results.csv (val set, epoch 132)',
)

# baseline 模型 test 集指标（来自 all_results.json，step04_experiments.py 评估）
BASELINE_TEST_METRICS = ModelMetrics(
    model_name='yolov8s_baseline-2',
    model_path='outputs/baseline/yolov8s_baseline-2/weights/best.pt',
    dataset='neudet_official_yolo',
    split='test',
    precision=0.0,       # 未记录
    recall=0.0,          # 未记录
    map50=0.7635,        # 76.35%
    map50_95=0.4122,     # 41.22%
    epoch=None,
    source='all_results.json (test set, step04_experiments.py)',
)

# GAN 增强模型 test 集指标（来自 all_results.json）
# 【注意】GAN 模型缺少 args.yaml 和 results.csv，训练过程不可追溯
GAN_TEST_METRICS = ModelMetrics(
    model_name='yolov8s_neudet_gan',
    model_path='outputs/yolo_runs/yolov8s_neudet_gan/weights/best.pt',
    dataset='neudet_official_yolo',
    split='test',
    precision=0.8429,
    recall=0.8604,
    map50=0.8897,        # 88.97% —— 即报告中引用的数字
    map50_95=0.6569,     # 65.69%
    epoch=None,
    source='all_results.json (test set, step04_experiments.py)',
)


# ============================================================
# 88.97% 说明
# ============================================================

EXPLANATION_8897 = """
关于 88.97% mAP@0.5 的说明：

1. 来源：GAN 增强模型 (yolov8s_neudet_gan) 在 neudet_official_yolo 数据集 test 集上的评估结果
2. 不是 baseline 模型的结果（baseline test mAP@0.5 = 76.35%）
3. 不是 val 集的结果（baseline val mAP@0.5 = 77.41%）
4. GAN 模型缺少训练配置文件（args.yaml）和训练过程记录（results.csv），可复现性存疑
5. neudet_official_yolo 数据集有 1800 张图像，比原始 NEU-DET（1770张）多 30 张，
   多出的图像标注来源需要进一步验证

因此：
- 88.97% 不作为本演示原型的核心指标
- 演示原型使用 baseline 模型 (yolov8s_baseline-2, val mAP50=77.41%)
- 所有检测结果均来自实际运行，不引用 88.97% 作为当前模型的性能
"""


# ============================================================
# 演示原型使用的配置
# ============================================================

DEMO_MODEL_CONFIG = {
    'model_name': 'yolov8s_baseline-2',
    'model_path': 'outputs/baseline/yolov8s_baseline-2/weights/best.pt',
    'training_dataset': 'neudet_official_yolo (1260 train / 270 val / 270 test)',
    'val_map50': '77.41% (epoch 125)',
    'val_map50_95': '42.76% (epoch 132, best.pt)',
    'test_map50': '76.35%',
    'test_map50_95': '41.22%',
    'training_epochs': '162 (配置200, early stopping终止)',
    'note': '演示使用baseline模型，所有检测结果均为实际运行产出。GAN模型的88.97%仅作背景参考，不构成当前原型的性能声明。',
}


# ============================================================
# 已知问题和限制
# ============================================================

KNOWN_DATA_ISSUES = [
    {
        'severity': 'high',
        'issue': 'NEU-DET 原始数据只有 1770 张，但 neudet_official_yolo 有 1800 张',
        'detail': '多出的 30 张图像标注来源不明，共多出 63 个标注框',
    },
    {
        'severity': 'high',
        'issue': '项目中存在 5 个 YOLO 格式数据集，划分各不相同且互相重叠',
        'detail': '混用会导致数据泄露，如 yolo_format/train 与 neudet_official_yolo/val 重叠 188 张',
    },
    {
        'severity': 'high',
        'issue': '88.97% 是 test 集上 GAN 模型的结果，不是 baseline 模型',
        'detail': 'GAN 模型缺少训练配置和过程记录，可复现性差',
    },
    {
        'severity': 'medium',
        'issue': 'baseline 训练使用的数据集与 configs/yolo_config.yaml 不一致',
        'detail': '实际用 neudet_official_yolo，配置文件写的是 yolo_format',
    },
]


def get_data_summary_text() -> str:
    """获取数据集一致性摘要文本（用于报告/日志输出）"""
    lines = [
        "=" * 60,
        "数据集一致性说明",
        "=" * 60,
        "",
        "检测模型: yolov8s_baseline-2",
        f"训练数据集: neudet_official_yolo (1800张, 70/15/15划分)",
        f"验证集 mAP@0.5: {BASELINE_VAL_METRICS.map50*100:.2f}% (epoch 125)",
        f"验证集 mAP@0.5:0.95: {BASELINE_VAL_METRICS.map50_95*100:.2f}% (epoch 132, best.pt)",
        f"测试集 mAP@0.5: {BASELINE_TEST_METRICS.map50*100:.2f}%",
        "",
        "重要说明:",
        "1. 原始 NEU-DET 只有 1770 张图像，neudet_official_yolo 有 1800 张",
        "   多出的 30 张标注来源需进一步验证",
        "2. 88.97% 是 GAN 增强模型在 test 集上的结果，不是 baseline 模型",
        "3. GAN 模型缺少训练配置文件，可复现性存疑",
        "4. 本演示原型使用 baseline 模型，所有结果均为实际运行产出",
    ]
    return "\n".join(lines)


if __name__ == '__main__':
    print(get_data_summary_text())
