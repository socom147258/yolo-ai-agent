# 工业缺陷检测系统部署文档

**项目名称**: GAN+YOLO 工业零件缺陷检测系统
**参赛方向**: 制造方向（MANUFACTURING）- 质量检测
**技术方案**: GAN数据增强 + YOLO目标检测联合框架

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境要求](#2-环境要求)
3. [环境安装](#3-环境安装)
4. [数据集准备](#4-数据集准备)
5. [GAN数据增强](#5-gan数据增强)
6. [YOLO检测模型训练](#6-yolo检测模型训练)
7. [模型评估](#7-模型评估)
8. [模型导出与部署](#8-模型导出与部署)
9. [推理使用](#9-推理使用)
10. [项目结构](#10-项目结构)
11. [常见问题](#11-常见问题)

---

## 1. 项目概述

### 1.1 技术方案

本项目采用 **GAN（生成对抗网络）+ YOLO（目标检测）** 联合框架，解决工业场景中缺陷样本稀缺的核心痛点：

1. **数据增强阶段**: 使用 DCGAN/CDGAN 生成逼真的合成缺陷图像，扩充训练数据集
   - **DCGAN**: 基础深度卷积生成对抗网络，支持条件生成
   - **CDGAN**: 双判别器结构，全局判别器评估整图真实性，局部判别器专注缺陷区域纹理细节

2. **检测阶段**: 使用 YOLOv8/YOLOv11 作为核心检测架构
   - 支持注意力机制（SE/CBAM/CA）增强小缺陷检测能力
   - 支持多尺度特征融合
   - 实时检测，输出缺陷类别、位置坐标及置信度

3. **模型轻量化**: 支持模型剪枝、量化，可部署于边缘计算设备

### 1.2 支持数据集

- **NEU钢表面缺陷数据集**: 6种热轧钢带表面缺陷，1800张灰度图像
- **MVTec AD数据集**: 15类工业产品，70余种缺陷类型（需手动下载）

---

## 2. 环境要求

### 2.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4核8线程 | 8核16线程 |
| 内存 | 8GB | 16GB+ |
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060 12GB+ |
| 显存 | 6GB | 12GB+ |
| 硬盘 | 20GB 可用空间 | 50GB SSD |

### 2.2 软件要求

- **操作系统**: Windows 10/11 或 Linux (Ubuntu 20.04+)
- **Python**: 3.9 ~ 3.11
- **CUDA**: 11.7 ~ 12.2 (GPU训练需要)
- **cuDNN**: 对应CUDA版本

### 2.3 Python依赖

核心依赖包：
```
torch>=2.0.0              # PyTorch深度学习框架
torchvision>=0.15.0       # 计算机视觉工具包
ultralytics>=8.1.0        # YOLOv8/v11官方库
opencv-python>=4.8.0      # 图像处理
numpy>=1.24.0             # 数值计算
onnx>=1.14.0              # ONNX模型格式
onnxruntime>=1.15.0       # ONNX推理引擎
pyyaml>=6.0               # YAML配置解析
tqdm>=4.65.0              # 进度条
matplotlib>=3.7.0         # 可视化
```

---

## 3. 环境安装

### 3.1 创建虚拟环境（推荐）

```bash
# 使用conda (推荐)
conda create -n defect_detect python=3.10
conda activate defect_detect

# 或使用venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate
```

### 3.2 安装PyTorch

根据CUDA版本选择对应的安装命令：

**CUDA 11.8:**
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU版本 (无GPU):**
```bash
pip install torch==2.0.1 torchvision==0.15.2
```

验证安装：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # GPU版本应返回True
```

### 3.3 安装项目依赖

```bash
pip install -r requirements.txt
```

### 3.4 验证安装

```bash
python -c "import torch; import ultralytics; import cv2; print('All dependencies installed successfully')"
```

---

## 4. 数据集准备

### 4.1 NEU钢表面缺陷数据集（推荐快速开始）

**方式一：手动下载（推荐）**

1. 访问官方地址：http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
2. 或 Kaggle：https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
3. 下载后解压到 `data/raw/NEU-CLS/` 目录

数据集包含6类缺陷，每类300张灰度图像（200×200像素）：
- crazing (裂纹)
- inclusion (夹杂物)
- patches (斑块)
- pitted_surface (麻点)
- rolled-in_scale (氧化铁皮)
- scratches (划痕)

**方式二：使用下载脚本**
```bash
python scripts/download_data.py --dataset NEU --save_dir data/raw
```

### 4.2 数据预处理

将原始数据转换为YOLO训练格式和GAN训练格式：

```bash
python scripts/preprocess_data.py \
    --dataset NEU \
    --raw_dir data/raw/NEU-CLS \
    --output_dir data/processed/yolo_format \
    --gan_output_dir data/processed/gan_dataset \
    --image_size 64 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seed 42
```

处理后目录结构：
```
data/processed/
├── yolo_format/           # YOLO格式数据集
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   ├── test/
│   └── data.yaml
└── gan_dataset/           # GAN训练数据集
    ├── train/
    │   ├── crazing/
    │   ├── inclusion/
    │   └── ...
    └── val/
```

---

## 5. GAN数据增强

### 5.1 DCGAN训练

```bash
python scripts/train_gan.py \
    --config configs/gan_config.yaml \
    --model dcgan \
    --epochs 200 \
    --batch_size 64
```

训练过程中：
- 样本图片保存在 `outputs/gan_samples/dcgan/`
- 模型检查点保存在 `outputs/checkpoints/dcgan/`
- 训练日志保存在 `outputs/logs/`

### 5.2 CDGAN训练（双判别器，更高质量）

```bash
python scripts/train_gan.py \
    --config configs/gan_config.yaml \
    --model cdgan \
    --epochs 300 \
    --batch_size 32
```

CDGAN特点：
- 全局判别器：评估整体图像真实性
- 局部判别器：专注缺陷区域纹理细节
- 重建损失：提升内容一致性

### 5.3 生成合成缺陷图像

训练完成后，使用最佳模型生成合成数据：

```bash
python scripts/generate_samples.py \
    --checkpoint outputs/checkpoints/dcgan/best_model.pth \
    --model dcgan \
    --num_per_class 500 \
    --output_dir data/processed/gan_generated \
    --batch_size 64
```

生成数量建议：
- 快速验证：每类100张
- 标准训练：每类300-500张
- 完整增强：每类1000张+

### 5.4 混合真实数据与GAN数据

```bash
# 编辑 preprocess_data.py 或直接运行混合功能
python scripts/preprocess_data.py \
    --raw_dir data/raw/NEU-CLS \
    --output_dir data/processed/yolo_format \
    --gan_dir data/processed/gan_generated \
    --gan_ratio 0.5
```

混合后数据集位置：`data/processed/yolo_gan_mixed/`

---

## 6. YOLO检测模型训练

### 6.1 基础训练（仅真实数据）

```bash
python scripts/train_yolo.py \
    --config configs/yolo_config.yaml \
    --model yolov8n \
    --epochs 300 \
    --batch_size 16 \
    --imgsz 640
```

### 6.2 使用GAN增强数据训练

编辑 `configs/yolo_config.yaml`：
```yaml
data:
  use_gan_augmentation: true
  gan_ratio: 0.5
```

或命令行指定：
```bash
python scripts/train_yolo.py \
    --config configs/yolo_config.yaml \
    --model yolov8s \
    --data data/processed/yolo_gan_mixed/data.yaml \
    --epochs 300
```

### 6.3 模型选择

| 模型 | 参数量 | 速度 | mAP | 适用场景 |
|------|--------|------|-----|---------|
| yolov8n | 3.2M | 最快 | 基础 | 边缘部署、实时检测 |
| yolov8s | 11.2M | 快 | 较好 | 平衡速度与精度 |
| yolov8m | 25.9M | 中 | 好 | 精度优先 |
| yolov8l | 43.7M | 较慢 | 很好 | 服务器端推理 |
| yolov8x | 68.2M | 最慢 | 最好 | 最高精度要求 |

### 6.4 训练参数调优

编辑 `configs/yolo_config.yaml`：

```yaml
training:
  epochs: 300          # 训练轮数
  batch_size: 16       # 批次大小（根据显存调整）
  patience: 50         # 早停耐心值
  lr0: 0.01            # 初始学习率
  # 数据增强
  mosaic: 1.0          # Mosaic增强概率
  mixup: 0.0           # MixUp增强概率
  fliplr: 0.5          # 水平翻转概率
```

### 6.5 训练输出

训练结果保存在 `outputs/yolo_runs/{model_name}/`：
- `weights/best.pt` - 最佳模型权重
- `weights/last.pt` - 最新模型权重
- `results.png` - 训练曲线图
- `confusion_matrix.png` - 混淆矩阵
- `results.csv` - 详细指标数据

---

## 7. 模型评估

### 7.1 综合评估

```bash
python scripts/evaluate.py \
    --weights outputs/yolo_runs/yolov8n/weights/best.pt \
    --data data/processed/yolo_format/data.yaml \
    --output_dir outputs/results/evaluation \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.45
```

评估指标：
- **mAP@0.5**: IoU阈值0.5时的平均精度
- **mAP@0.5:0.95**: IoU从0.5到0.95的平均mAP
- **Precision (精确率)**: 预测为正的样本中真正为正的比例
- **Recall (召回率)**: 所有正样本中被正确检测的比例
- **F1-Score**: 精确率和召回率的调和平均

### 7.2 多模型对比

```bash
python scripts/evaluate.py \
    --weights outputs/yolo_runs/yolov8n/weights/best.pt \
    --compare outputs/yolo_runs/yolov8s/weights/best.pt \
               outputs/yolo_runs/yolov8m/weights/best.pt \
    --data data/processed/yolo_format/data.yaml \
    --output_dir outputs/results/comparison
```

### 7.3 评估结果

结果保存在 `outputs/results/evaluation/`：
- `evaluation_results.yaml` - 详细指标数据
- `overall_metrics.png` - 总体指标图
- `per_class_metrics.png` - 每类指标对比图
- `eval/` - Ultralytics详细评估输出

---

## 8. 模型导出与部署

### 8.1 导出为ONNX格式

```bash
python scripts/export_model.py \
    --weights outputs/yolo_runs/yolov8n/weights/best.pt \
    --format onnx \
    --imgsz 640 \
    --simplify \
    --opset 12 \
    --verify
```

### 8.2 导出为TensorRT格式（NVIDIA GPU）

```bash
python scripts/export_model.py \
    --weights outputs/yolo_runs/yolov8n/weights/best.pt \
    --format engine \
    --imgsz 640 \
    --half \
    --workspace 8
```

### 8.3 量化选项

**FP16量化**（GPU推理加速）：
```bash
python scripts/export_model.py --weights best.pt --format onnx --half
```

**INT8量化**（极致压缩，需校准数据）：
```bash
python scripts/export_model.py --weights best.pt --format onnx --int8
```

### 8.4 速度基准测试

```bash
python scripts/export_model.py \
    --weights outputs/yolo_runs/yolov8n/weights/best.pt \
    --benchmark \
    --device 0
```

### 8.5 导出格式对比

| 格式 | 大小 | 速度 | 平台 | 适用场景 |
|------|------|------|------|---------|
| PyTorch (.pt) | 大 | 较慢 | 全平台 | 训练、研究 |
| ONNX (.onnx) | 中 | 中 | 全平台 | 跨框架部署 |
| TensorRT (.engine) | 中 | 最快 | NVIDIA GPU | 生产部署 |
| TorchScript (.torchscript) | 中 | 中 | 全平台 | C++部署 |

---

## 9. 推理使用

### 9.1 单张图像推理

```bash
# PyTorch后端
python scripts/inference.py \
    --weights outputs/yolo_runs/yolov8n/weights/best.pt \
    --source test_image.jpg \
    --output outputs/inference \
    --conf 0.25

# ONNX后端
python scripts/inference.py \
    --weights outputs/export/best.onnx \
    --source test_image.jpg \
    --backend onnx \
    --conf 0.25
```

### 9.2 批量推理（目录）

```bash
python scripts/inference.py \
    --weights best.pt \
    --source data/test_images/ \
    --output outputs/inference_results/ \
    --conf 0.3
```

### 9.3 推理输出

每张图像生成两个文件：
- `{name}_result.jpg` - 带检测框的可视化图像
- `{name}_result.txt` - 检测结果文本

检测结果格式：
```
Defect 1:
  Class: crazing
  Confidence: 0.9234
  Bounding Box: [x1, y1, x2, y2]
```

### 9.4 Python API调用

```python
from scripts.inference import YOLOInference
import cv2

# 初始化
inferencer = YOLOInference(
    weights_path="best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
    imgsz=640,
    device="0"
)

# 读取图像
image = cv2.imread("test.jpg")

# 推理
detections = inferencer.predict(image)

# 可视化
vis_image = inferencer.visualize(image, detections)
cv2.imwrite("result.jpg", vis_image)

# 打印结果
for det in detections:
    print(f"{det['class']}: {det['confidence']:.2f} @ {det['bbox']}")
```

---

## 10. 项目结构

```
YOLO/
├── configs/                    # 配置文件
│   ├── gan_config.yaml         # GAN训练配置
│   └── yolo_config.yaml        # YOLO训练配置
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   │   └── NEU-CLS/
│   └── processed/              # 预处理后数据
│       ├── yolo_format/        # YOLO格式数据
│       ├── yolo_gan_mixed/     # GAN增强混合数据
│       ├── gan_dataset/        # GAN训练数据
│       └── gan_generated/      # GAN生成数据
├── models/                     # 模型定义
│   ├── gan/
│   │   ├── dcgan.py            # DCGAN模型
│   │   └── cdgan.py            # CDGAN模型
│   └── yolo/
│       └── attention.py        # 注意力机制模块
├── scripts/                    # 运行脚本
│   ├── download_data.py        # 数据集下载
│   ├── preprocess_data.py      # 数据预处理
│   ├── train_gan.py            # GAN训练
│   ├── generate_samples.py     # GAN样本生成
│   ├── train_yolo.py           # YOLO训练
│   ├── evaluate.py             # 模型评估
│   ├── export_model.py         # 模型导出
│   └── inference.py            # 推理脚本
├── utils/                      # 工具函数
│   ├── common.py               # 通用工具
│   └── dataset.py              # 数据集类
├── outputs/                    # 输出目录
│   ├── gan_samples/            # GAN样本
│   ├── checkpoints/            # 模型检查点
│   ├── yolo_runs/              # YOLO训练结果
│   ├── results/                # 评估结果
│   ├── export/                 # 导出模型
│   ├── inference/              # 推理结果
│   └── logs/                   # 日志文件
├── docs/                       # 文档
│   └── deployment.md           # 部署文档
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明
```

---

## 11. 常见问题

### Q1: CUDA out of memory 怎么办？

A: 减小batch_size或image_size：
```bash
python scripts/train_yolo.py --batch_size 8 --imgsz 416
```

### Q2: GAN训练不稳定怎么办？

A: 尝试以下调整：
- 降低学习率：`lr: 0.0001`
- 使用标签平滑：`label_smoothing: 0.9`
- 增加判别器更新频率
- 使用Wasserstein loss (需修改模型)

### Q3: 检测精度低怎么提升？

A: 可以从以下方面优化：
1. 增加训练轮数
2. 使用更大的模型 (yolov8s/m/l)
3. 增加GAN生成数据量
4. 调整数据增强参数
5. 使用注意力机制
6. 调优置信度阈值

### Q4: ONNX推理结果与PyTorch不一致？

A: 常见原因：
- 预处理方式不同 (归一化、颜色空间)
- NMS实现差异
- 尝试使用 `--simplify` 简化ONNX模型
- 检查opset版本兼容性

### Q5: 如何在边缘设备部署？

A: 推荐流程：
1. 选择轻量模型 (yolov8n)
2. 导出为ONNX格式
3. 使用INT8量化
4. 使用NCNN/TNN/MNN等推理框架
5. 或使用TensorRT部署在Jetson等边缘GPU

### Q6: 训练结果如何复现？

A: 确保以下几点：
- 设置相同的随机种子 (seed: 42)
- 使用相同版本的依赖包
- 使用相同的训练数据划分
- 启用deterministic模式

---

## 技术支持

如有问题，请参考：
- Ultralytics官方文档: https://docs.ultralytics.com/
- PyTorch官方文档: https://pytorch.org/docs/
- ONNX官方文档: https://onnx.ai/
