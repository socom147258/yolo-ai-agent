# 基于 GAN-YOLO 与 AI Agent 的工业表面缺陷智能检测及质量分析系统

## 项目简介

本项目是一个面向工业表面缺陷检测的完整 AI 系统，以 **cDCGAN 条件生成对抗网络** 为数据增强核心，以 **YOLOv8s** 为检测基线，以 **AI Agent 智能体** 为交互协调层，构建了集检测、判定、分析、报告于一体的智能化质量检测解决方案。

### 三大核心创新

1. **cDCGAN 条件生成 + Copy-Paste 协同的数据增强方法** — 解决工业缺陷小样本和类别不均衡问题
2. **AI Agent 智能体 + 三级质量判定体系** — 自然语言交互、多工具调用、PASS/REVIEW/FAIL 三级判定
3. **面向制造业应用的全流程可视化质量分析系统** — 完整的软件系统，而非单纯的检测算法

---

## 目录结构

```
YOLO/
├── app.py                          # Streamlit 主程序（5大功能模块）
├── requirements.txt                # Python 依赖清单
│
├── models/
│   ├── gan/
│   │   └── cdgan_pipeline.py       # cDCGAN 模型 + Copy-Paste 合成 + 自动标注
│   └── agent/
│       ├── __init__.py
│       └── detection_agent.py      # AI Agent 核心：AIAgent + QualityRuleEngine + ToolRegistry
│
├── scripts/
│   ├── generate_final_report.py    # 生成最终竞赛报告（Word）
│   ├── step02_gan_augmentation.py  # Step 2: GAN 数据增强全流程
│   ├── step03_agent_setup.py       # Step 3: Agent 功能验证
│   ├── step04_experiments.py       # Step 4: 完整实验评估
│   └── step05_generate_report.py   # Step 5: 生成最终报告
│
├── configs/
│   └── data_config.yaml            # 数据集配置
│
├── docs/
│   └── competition_report_final.docx  # 最终竞赛报告
│
├── data/
│   └── processed/                  # 处理后的数据集（运行时生成）
│
└── outputs/
    ├── baseline/                   # 基线模型训练输出
    ├── phase2/                     # GAN 增强模型输出
    ├── final_experiments/          # 实验结果汇总
    ├── agent/                      # Agent 数据库
    ├── export/                     # 模型导出（ONNX等）
    └── temp/                       # 临时文件
```

---

## 核心模块说明

### 1. AI Agent 智能体 (`models/agent/detection_agent.py`)

系统的智能核心，包含三个主要类：

| 类名 | 功能 | 关键方法 |
|------|------|---------|
| `QualityRuleEngine` | 三级质量判定引擎 | `evaluate()` - 输入检测结果，输出PASS/REVIEW/FAIL |
| `ToolRegistry` | 工具注册表（7个工具） | `detect`, `count_defects`, `classify_filter`, `quality_evaluate`, `explain_result`, `save_record`, `generate_report` |
| `AIAgent` | AI Agent 协调层 | `execute_task()` - 输入自然语言指令+图片，输出自然语言回答 |

#### Agent 工作流程

```
用户输入 → 任务理解 → 任务规划 → 工具执行 → 结果整合 → 自然语言输出
              ↓           ↓          ↓          ↑
           意图识别    工具序列    上下文传递   多工具结果整合
```

#### 三级质量判定维度

- **缺陷类别严重等级**（1-5级，裂纹最严重）
- **缺陷数量密度**（总缺陷数 + 严重缺陷数）
- **缺陷面积占比**（总面积% + 最大单缺陷%）
- **检测置信度**（低置信度触发复核建议）

---

### 2. cDCGAN + Copy-Paste 数据增强 (`models/gan/cdgan_pipeline.py`)

| 类/函数 | 功能 |
|---------|------|
| `Generator` | cDCGAN 生成器（噪声+类别标签 → 64×64缺陷图） |
| `Discriminator` | cDCGAN 判别器（图像+类别标签 → 真假概率） |
| `train_cdcgan()` | cDCGAN 训练函数 |
| `generate_defect_images()` | 按类别批量生成缺陷图像 |
| `copy_paste_synthesize()` | Copy-Paste 合成 + 自动标注（基于mask的边界框生成） |
| `mix_real_gan_data()` | 真实数据与合成数据混合 |

#### Copy-Paste 自动标注五步流程

```
1. 缺陷区域提取（Otsu阈值 + 最大连通域）
2. 缺陷变换与粘贴（缩放/旋转/随机位置）
3. 边缘融合（羽化 alpha 融合）
4. 自动标注生成（基于 mask 计算边界框 → YOLO格式）
5. 质量检查与筛选（面积/边界/完整性校验）
```

---

### 3. Streamlit 可视化界面 (`app.py`)

五大功能模块：

| 模块 | 功能说明 |
|------|---------|
| 🤖 AI Agent 对话 | 自然语言交互，支持多轮对话、工具调用展示、检测可视化 |
| 🔍 智能检测 | 单张/批量检测，实时显示检测框和三级质量判定 |
| 📊 数据分析 | 缺陷统计、质量分布图表、历史记录查询 |
| ⚙️ 质量规则 | 可视化配置三级判定规则参数，即时生效 |
| 📋 报告中心 | 检测记录管理、详情查看、统计报告 |

---

## 环境配置

### 系统要求

- Python 3.10+
- Windows / Linux / macOS
- 推荐 GPU（CUDA 11.8+），也支持 CPU 运行

### 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| ultralytics | 8.x | YOLOv8 检测框架 |
| torch | 2.x | PyTorch 深度学习 |
| torchvision | 0.x | 视觉工具库 |
| streamlit | 1.x | Web 界面框架 |
| python-docx | 1.x | Word 文档生成 |
| pillow | 10.x | 图像处理 |
| opencv-python | 4.x | 计算机视觉 |
| numpy | 1.x | 数值计算 |
| pandas | 2.x | 数据分析 |
| matplotlib | 3.x | 数据可视化 |

---

## 快速开始

### 1. 准备数据

将 NEU-DET 数据集放入 `data/NEU-DET/` 目录，结构如下：

```
data/NEU-DET/
├── crazing/
├── inclusion/
├── patches/
├── pitted_surface/
├── rolled-in_scale/
└── scratches/
```

### 2. 运行 GAN 数据增强

```bash
python scripts/step02_gan_augmentation.py
```

完成后生成：
- cDCGAN 训练好的生成器权重
- GAN 合成图像 + 自动标注
- GAN 增强版 YOLO 模型

### 3. 运行 Agent 功能验证

```bash
python scripts/step03_agent_setup.py
```

### 4. 运行完整实验评估

```bash
python scripts/step04_experiments.py
```

### 5. 生成最终竞赛报告

```bash
python scripts/generate_final_report.py
```

### 6. 启动可视化界面

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

---

## 实验结果

### NEU-DET 测试集性能

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|------|---------|-------------|-----------|--------|
| YOLOv8s 基线 | 76.35% | 44.12% | 78.5% | 72.3% |
| YOLOv8s + cDCGAN 增强 | **88.97%** | **52.84%** | **91.2%** | **86.7%** |
| 提升 | +12.62% | +8.72% | +12.7% | +14.4% |

### 各类别性能提升

| 缺陷类别 | 基线 | GAN增强 | 提升 |
|---------|------|---------|------|
| 裂纹 crazing | 68.2% | 85.6% | **+17.4%** |
| 夹杂 inclusion | 72.5% | 86.9% | +14.4% |
| 斑块 patches | 82.1% | 92.3% | +10.2% |
| 麻点 pitted_surface | 78.8% | 89.1% | +10.3% |
| 氧化皮 rolled-in_scale | 75.3% | 87.5% | +12.2% |
| 划痕 scratches | 81.2% | 92.4% | +11.2% |

### AI Agent 功能测试

| 测试项 | 结果 |
|--------|------|
| 功能测试（8项） | 100% 通过 |
| 意图识别准确率 | 90% |
| 单张推理（CPU） | ~55ms |
| 系统响应时间 | ~65ms |
| 质量判定耗时 | <1ms |

---

## 竞赛报告

完整竞赛报告位于 `docs/competition_report_final.docx`，包含六大章节：

1. **项目背景与目标** — 制造业痛点、四大目标
2. **系统架构与技术原理** — 五层架构、cDCGAN、YOLOv8、数据增强、AI Agent
3. **数据集** — NEU-DET 概况、数据划分与防泄漏说明
4. **实验结果与分析** — 基线对比、GAN对比、类别分析、Agent验证
5. **模型可靠性评估** — 鲁棒性、推理性能、系统稳定性
6. **创新点与总结** — 三大核心创新、后续方向、总结

---

## 注意事项

1. **质量规则参数**：系统中的三级质量判定规则为实验演示设定，不代表任何企业的正式产品验收标准。实际应用时需根据具体产品质量标准进行配置。

2. **AI Agent 实现方式**：当前版本采用基于关键词匹配的有限任务规划实现，不依赖外部大语言模型，可在本地独立运行。如需更强的自然语言理解能力，可扩展接入 LLM。

3. **数据隔离**：GAN 训练和合成仅使用训练集数据，测试集严格隔离，确保评估结果客观可靠。

4. **模型文件**：由于模型权重文件较大，未包含在代码包中。运行前需先训练模型或下载预训练权重。

---

## 技术栈

- **检测模型**：Ultralytics YOLOv8s
- **生成模型**：cDCGAN（条件深度卷积生成对抗网络）
- **智能体框架**：自研 Agent 协调层（任务理解 + 工具调用 + 结果解释）
- **Web 界面**：Streamlit
- **数据库**：SQLite
- **文档生成**：python-docx
- **深度学习框架**：PyTorch

---

## 版本信息

- 版本：v3.0（最终竞赛版）
- 更新日期：2026-11-03
- 主要更新：
  - v1.0：GAN + YOLOv8 基线检测系统
  - v2.0：增加规则引擎自动化 + SQLite + Streamlit 四模块
  - v3.0：升级为真正 AI Agent（任务理解+工具调用）+ 三级质量判定 + 五大模块界面 + 报告全面重构
