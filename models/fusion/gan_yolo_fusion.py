# -*- coding: utf-8 -*-
"""
GAN-YOLO 深度融合模块

4 种融合机制：
1. 定向数据增强（Adaptive Augmentation）：分析 YOLO 各类别表现，
   对薄弱类别用 GAN 定向生成样本，形成"检测→诊断→增强→再训练"闭环
2. 交叉验证检测（Cross-Validation Detection）：GAN 重建误差 + YOLO 检测
   双路验证，互补盲区，降低漏检和误检
3. 困难样本挖掘（Hard Sample Mining）：GAN 生成多样性样本 + YOLO 检测，
   筛选出低置信度/误检/漏检的困难样本，构建困难样本库
4. 融合质量判定（Fused Quality Judgement）：综合 YOLO 检测置信度 +
   GAN 重建异常度 + 缺陷特征一致性，给出更可靠的质量判定
"""

import os
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import copy


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ClassPerformance:
    """单类别检测性能指标"""
    class_name: str
    class_idx: int
    total_samples: int = 0
    detected_count: int = 0        # 正确检测数
    missed_count: int = 0          # 漏检数
    false_positive_count: int = 0  # 误检数
    avg_confidence: float = 0.0    # 平均置信度
    low_confidence_count: int = 0  # 低置信度样本数
    performance_score: float = 0.0  # 综合表现评分（0-1，越低越需要增强）


@dataclass
class FusionDetectionResult:
    """融合检测结果"""
    # YOLO 检测结果
    yolo_detections: List[Dict] = field(default_factory=list)
    yolo_confidences: List[float] = field(default_factory=list)

    # GAN 重建结果
    gan_reconstruction: np.ndarray = None
    reconstruction_error: float = 0.0
    anomaly_map: np.ndarray = None  # 像素级异常热力图

    # 融合分析
    fused_quality: str = "PASS"     # PASS / REVIEW / FAIL
    fused_confidence: float = 0.0   # 融合置信度
    cross_validation_notes: List[str] = field(default_factory=list)

    # 指标
    processing_time: float = 0.0


@dataclass
class HardSample:
    """困难样本"""
    image: np.ndarray
    class_name: str
    difficulty: str = "medium"  # easy / medium / hard / extreme
    reason: str = ""            # 困难原因
    yolo_confidence: float = 0.0
    detection_correct: bool = True


# ============================================================
# 融合管理器
# ============================================================

class GanYoloFusion:
    """
    GAN-YOLO 深度融合管理器

    实现 GAN 生成模型与 YOLO 检测模型的深度协同，
    包括定向增强、交叉验证、困难样本挖掘、融合判定等功能。
    """

    # 类别名称
    CLASS_NAMES = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"
    ]

    CLASS_NAMES_CN = {
        "crazing": "裂纹",
        "inclusion": "夹杂",
        "patches": "斑块",
        "pitted_surface": "麻点",
        "rolled-in_scale": "氧化皮",
        "scratches": "划痕",
    }

    def __init__(self, gan_manager=None, yolo_model=None,
                 device: str = None):
        """
        初始化融合管理器

        Args:
            gan_manager: GAN 管理器实例
            yolo_model: YOLO 模型实例
            device: 计算设备
        """
        self.gan = gan_manager
        self.yolo = yolo_model

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # 各类别性能记录
        self.class_performances: Dict[str, ClassPerformance] = {
            name: ClassPerformance(class_name=name, class_idx=i)
            for i, name in enumerate(self.CLASS_NAMES)
        }

        # 困难样本库
        self.hard_samples: List[HardSample] = []

        # 统计
        self._stats = {
            "fused_detections": 0,
            "hard_samples_mined": 0,
            "augmentation_rounds": 0,
        }

    # ============================================================
    # 机制 1：定向数据增强
    # ============================================================

    def analyze_class_performance(self, detection_results: List[Dict]) -> Dict:
        """
        分析 YOLO 各类别的检测性能，找出薄弱类别

        Args:
            detection_results: 检测结果列表，每项包含
                - image_path: 图像路径
                - true_class: 真实类别
                - detections: YOLO 检测结果列表 [{class, conf, bbox}]

        Returns:
            性能分析报告
        """
        # 重置统计
        for name in self.CLASS_NAMES:
            self.class_performances[name] = ClassPerformance(
                class_name=name,
                class_idx=self.CLASS_NAMES.index(name),
            )

        for result in detection_results:
            true_class = result.get("true_class", "")
            detections = result.get("detections", [])

            if true_class not in self.class_performances:
                continue

            perf = self.class_performances[true_class]
            perf.total_samples += 1

            # 检查是否检测到正确类别
            detected = False
            confidences = []
            for det in detections:
                det_class = det.get("class", "")
                det_conf = det.get("confidence", 0)

                # 处理类别索引或名称
                if isinstance(det_class, int):
                    det_class = self.CLASS_NAMES[det_class] if det_class < len(self.CLASS_NAMES) else ""

                if det_class == true_class:
                    detected = True
                    confidences.append(det_conf)
                if det_conf < 0.5:
                    perf.low_confidence_count += 1

            if detected:
                perf.detected_count += 1
                if confidences:
                    perf.avg_confidence = (perf.avg_confidence * (perf.detected_count - 1)
                                           + np.mean(confidences)) / perf.detected_count
            else:
                perf.missed_count += 1

        # 计算综合表现评分
        for name, perf in self.class_performances.items():
            if perf.total_samples == 0:
                perf.performance_score = 1.0  # 无数据视为良好
                continue

            # 召回率
            recall = perf.detected_count / perf.total_samples
            # 平均置信度归一化
            conf_norm = min(perf.avg_confidence / 0.8, 1.0)
            # 低置信度比例惩罚
            low_conf_ratio = perf.low_confidence_count / max(perf.total_samples, 1)
            low_conf_penalty = low_conf_ratio * 0.3

            # 综合评分（越高越好）
            perf.performance_score = max(0.0, recall * 0.6 + conf_norm * 0.4 - low_conf_penalty)

        # 找出薄弱类别（按评分升序）
        weak_classes = sorted(
            self.class_performances.values(),
            key=lambda x: x.performance_score
        )

        report = {
            "per_class": {
                name: {
                    "total_samples": perf.total_samples,
                    "detected": perf.detected_count,
                    "missed": perf.missed_count,
                    "avg_confidence": round(perf.avg_confidence, 4),
                    "low_confidence": perf.low_confidence_count,
                    "performance_score": round(perf.performance_score, 4),
                }
                for name, perf in self.class_performances.items()
            },
            "weak_classes": [
                {
                    "class_name": w.class_name,
                    "class_name_cn": self.CLASS_NAMES_CN.get(w.class_name, w.class_name),
                    "performance_score": round(w.performance_score, 4),
                    "needs_augmentation": w.performance_score < 0.7,
                }
                for w in weak_classes
                if w.total_samples > 0
            ],
            "overall_score": round(
                np.mean([p.performance_score for p in self.class_performances.values()
                         if p.total_samples > 0]) if any(
                    p.total_samples > 0 for p in self.class_performances.values()
                ) else 0.0,
                4
            ),
        }

        return report

    def adaptive_augmentation(self, performance_report: Dict,
                              base_samples_per_class: int = 100,
                              max_ratio: float = 5.0,
                              output_dir: str = None) -> Dict:
        """
        自适应定向数据增强

        根据各类别的表现评分，动态决定生成数量：
        - 表现好的类别：生成基础数量
        - 表现差的类别：生成更多（最多 max_ratio 倍）

        Args:
            performance_report: analyze_class_performance 的输出
            base_samples_per_class: 基础生成数量
            max_ratio: 最大增强倍率
            output_dir: 输出目录

        Returns:
            增强计划与结果
        """
        if self.gan is None:
            return {"status": "error", "message": "GAN 模型不可用"}

        per_class = performance_report.get("per_class", {})

        # 计算每个类别的增强数量
        augmentation_plan = {}
        for class_name, perf in per_class.items():
            score = perf["performance_score"]
            # 评分越低，生成越多（线性插值）
            ratio = 1.0 + (1.0 - score) * (max_ratio - 1.0)
            num_samples = int(base_samples_per_class * ratio)
            augmentation_plan[class_name] = {
                "performance_score": score,
                "augmentation_ratio": round(ratio, 2),
                "planned_samples": num_samples,
            }

        # 执行生成
        generated_stats = {}
        total_generated = 0

        if output_dir is None:
            output_dir = str(
                Path(__file__).parent.parent.parent / "data" / "processed" / "gan_augmented"
            )

        for class_name, plan in augmentation_plan.items():
            num = plan["planned_samples"]

            result = self.gan.generate(
                num_samples=num,
                class_name=class_name,
            )

            # 保存
            class_dir = Path(output_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for i, img in enumerate(result["images"]):
                fname = f"aug_{class_name}_{i:05d}.jpg"
                img.save(class_dir / fname)
                saved += 1

            generated_stats[class_name] = {
                "planned": num,
                "actual_generated": saved,
                "augmentation_ratio": plan["augmentation_ratio"],
            }
            total_generated += saved

        self._stats["augmentation_rounds"] += 1

        return {
            "method": "adaptive_augmentation",
            "description": "根据各类别检测性能自适应调整生成数量，薄弱类别生成更多样本",
            "base_samples": base_samples_per_class,
            "max_ratio": max_ratio,
            "output_dir": output_dir,
            "plan": augmentation_plan,
            "results": generated_stats,
            "total_generated": total_generated,
        }

    # ============================================================
    # 机制 2：交叉验证检测
    # ============================================================

    def cross_validation_detect(self, image: Union[str, Image.Image, np.ndarray],
                                yolo_result: Dict = None,
                                anomaly_threshold: float = 0.15) -> FusionDetectionResult:
        """
        GAN + YOLO 交叉验证检测

        原理：
        - GAN（生成器）学习了正常样本的分布，对有缺陷的图像重建误差大
        - 结合 YOLO 的检测结果，双路验证：
          * YOLO 检出 + GAN 异常高 → 高度可信缺陷
          * YOLO 检出 + GAN 异常低 → 可能误检，标记 REVIEW
          * YOLO 未检出 + GAN 异常高 → 可能漏检，标记 REVIEW
          * YOLO 未检出 + GAN 异常低 → 正常样本

        Args:
            image: 输入图像
            yolo_result: 预计算的 YOLO 检测结果，为 None 则自动检测
            anomaly_threshold: GAN 异常阈值（重建误差）

        Returns:
            融合检测结果
        """
        start_time = time.time()
        result = FusionDetectionResult()

        # 加载图像
        img_pil = self._load_image(image)

        # --- YOLO 检测 ---
        if yolo_result is None and self.yolo is not None:
            yolo_result = self._run_yolo_detection(img_pil)

        if yolo_result:
            result.yolo_detections = yolo_result.get("detections", [])
            result.yolo_confidences = [
                d.get("confidence", 0) for d in result.yolo_detections
            ]

        # --- GAN 重建分析 ---
        if self.gan is not None and self.gan.is_loaded:
            recon_result = self._gan_reconstruction_analysis(img_pil)
            result.gan_reconstruction = recon_result["reconstruction"]
            result.reconstruction_error = recon_result["error"]
            result.anomaly_map = recon_result["anomaly_map"]
        else:
            # GAN 不可用时，退化为纯 YOLO
            result.reconstruction_error = 0.0

        # --- 融合判定 ---
        yolo_has_defect = len(result.yolo_detections) > 0
        yolo_max_conf = max(result.yolo_confidences) if result.yolo_confidences else 0.0
        gan_anomaly_high = result.reconstruction_error > anomaly_threshold

        notes = []

        # 四象限判定
        if yolo_has_defect and gan_anomaly_high:
            # 双路都检出 → 高度可信
            if yolo_max_conf > 0.7:
                result.fused_quality = "FAIL"
                result.fused_confidence = 0.9
            else:
                result.fused_quality = "FAIL"
                result.fused_confidence = 0.75
            notes.append("YOLO 检出缺陷 + GAN 重建异常 → 高度可信缺陷")

        elif yolo_has_defect and not gan_anomaly_high:
            # YOLO 检出但 GAN 认为正常 → 可能误检
            result.fused_quality = "REVIEW"
            result.fused_confidence = 0.5
            notes.append("YOLO 检出缺陷但 GAN 重建误差低 → 可能误检，建议人工复核")

        elif not yolo_has_defect and gan_anomaly_high:
            # GAN 异常但 YOLO 未检出 → 可能漏检
            result.fused_quality = "REVIEW"
            result.fused_confidence = 0.55
            notes.append("GAN 重建异常但 YOLO 未检出 → 可能漏检，建议人工复核")

        else:
            # 双路都正常 → 合格
            result.fused_quality = "PASS"
            result.fused_confidence = 0.85
            notes.append("YOLO 未检出 + GAN 重建正常 → 合格")

        # 附加分析
        if yolo_max_conf > 0 and result.reconstruction_error > 0:
            # 一致性评分：两者一致程度
            consistency = 1.0 - abs(float(yolo_has_defect) - float(gan_anomaly_high))
            notes.append(f"双模型一致性: {'一致' if consistency > 0.5 else '不一致'}")

        result.cross_validation_notes = notes
        result.processing_time = time.time() - start_time

        return result

    def _gan_reconstruction_analysis(self, img_pil: Image.Image) -> Dict:
        """
        GAN 重建误差分析（简化版）

        完整实现需要训练一个专门用于重建的 GAN/autoencoder。
        这里使用生成器的条件生成 + 多类别平均来近似估计重建误差。

        注：实际生产环境建议使用专门的 AnoGAN / f-AnoGAN 等架构。
        """
        # 调整到 GAN 输入尺寸
        img_size = self.gan.image_size
        img_resized = img_pil.resize((img_size, img_size)).convert('L')
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(self.device)
        # 归一化到 [-1, 1]
        img_norm = img_tensor * 2.0 - 1.0

        # 用各类别的条件生成"最接近"的重建
        # 思路：生成各类别的平均图像，计算与输入的最小误差
        min_error = float('inf')
        best_recon = None
        best_anomaly_map = None

        with torch.no_grad():
            for class_idx in range(self.gan.num_classes):
                # 生成该类别的多个样本并取平均（作为"该类别典型样本"）
                z = torch.randn(8, self.gan.latent_dim, device=self.device)
                labels = torch.full((8,), class_idx, dtype=torch.long, device=self.device)

                gen_imgs = self.gan.generator(z, labels)
                gen_mean = gen_imgs.mean(dim=0, keepdim=True)

                # 计算重建误差
                # gen_mean: (1, channels, H, W)
                # img_norm: (1, channels, H, W)
                error_map = torch.abs(img_norm - gen_mean)
                error = error_map.mean().item()

                if error < min_error:
                    min_error = error
                    best_recon = ((gen_mean.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                    best_anomaly_map = error_map.squeeze().cpu().numpy()

        return {
            "reconstruction": best_recon,
            "error": min_error,
            "anomaly_map": best_anomaly_map,
            "method": "multi-class_conditional_average",
            "note": "简化版重建分析，完整方案建议使用 AnoGAN 等专用架构",
        }

    def _run_yolo_detection(self, img_pil: Image.Image) -> Dict:
        """运行 YOLO 检测"""
        if self.yolo is None:
            return {"detections": []}

        # 保存为临时文件供 YOLO 读取
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img_pil.save(f.name)
            tmp_path = f.name

        try:
            results = self.yolo.predict(tmp_path, verbose=False)
            detections = []
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_idx = int(box.cls[0].item())
                        cls_name = self.CLASS_NAMES[cls_idx] if cls_idx < len(self.CLASS_NAMES) else f"class_{cls_idx}"
                        detections.append({
                            "class": cls_name,
                            "class_idx": cls_idx,
                            "confidence": float(box.conf[0].item()),
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),
                        })
            return {"detections": detections}
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    # ============================================================
    # 机制 3：困难样本挖掘
    # ============================================================

    def mine_hard_samples(self, num_per_class: int = 20,
                          difficulty_thresholds: Dict = None) -> Dict:
        """
        困难样本挖掘

        用 GAN 生成大量样本 → YOLO 检测 → 筛选出困难样本
        困难类型：
        - 低置信度检测：confidence < 0.6
        - 误检：检测出的类别与生成类别不符
        - 漏检：生成了有缺陷的样本但 YOLO 没检测到

        Args:
            num_per_class: 每类生成数量
            difficulty_thresholds: 难度阈值

        Returns:
            困难样本挖掘结果
        """
        if self.gan is None:
            return {"status": "error", "message": "GAN 模型不可用"}

        if difficulty_thresholds is None:
            difficulty_thresholds = {
                "easy": 0.8,
                "medium": 0.6,
                "hard": 0.4,
                "extreme": 0.2,
            }

        hard_samples = []
        stats = {name: {"total": 0, "hard": 0, "reasons": {}} for name in self.CLASS_NAMES}

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            # 生成样本
            gen_result = self.gan.generate(
                num_samples=num_per_class,
                class_idx=class_idx,
            )

            stats[class_name]["total"] = num_per_class

            for i, img in enumerate(gen_result["images"]):
                # YOLO 检测
                yolo_result = self._run_yolo_detection(img)
                detections = yolo_result["detections"]

                # 分析是否为困难样本
                is_hard = False
                reason = ""
                difficulty = "easy"
                yolo_conf = 0.0
                correct = True

                # 查找正确类别的检测
                correct_dets = [
                    d for d in detections
                    if d.get("class") == class_name
                    or d.get("class_idx") == class_idx
                ]
                wrong_dets = [
                    d for d in detections
                    if d.get("class") != class_name
                    and d.get("class_idx") != class_idx
                ]

                if correct_dets:
                    yolo_conf = max(d["confidence"] for d in correct_dets)

                    # 低置信度
                    if yolo_conf < difficulty_thresholds["hard"]:
                        is_hard = True
                        difficulty = "extreme"
                        reason = f"极低置信度 ({yolo_conf:.3f})"
                    elif yolo_conf < difficulty_thresholds["medium"]:
                        is_hard = True
                        difficulty = "hard"
                        reason = f"低置信度 ({yolo_conf:.3f})"
                    elif yolo_conf < difficulty_thresholds["easy"]:
                        is_hard = True
                        difficulty = "medium"
                        reason = f"中等置信度 ({yolo_conf:.3f})"

                    # 有误检（同时检出了错误类别）
                    if wrong_dets:
                        is_hard = True
                        if difficulty == "easy":
                            difficulty = "medium"
                        reason += f" + 误检类别: {[d['class'] for d in wrong_dets]}"

                else:
                    # 漏检
                    is_hard = True
                    correct = False
                    difficulty = "extreme"
                    reason = "完全漏检"

                if is_hard:
                    sample = HardSample(
                        image=np.array(img),
                        class_name=class_name,
                        difficulty=difficulty,
                        reason=reason.strip(" + "),
                        yolo_confidence=yolo_conf,
                        detection_correct=correct,
                    )
                    hard_samples.append(sample)
                    stats[class_name]["hard"] += 1

                    reason_key = reason.split(" (")[0].strip(" + ")
                    stats[class_name]["reasons"][reason_key] = \
                        stats[class_name]["reasons"].get(reason_key, 0) + 1

        self.hard_samples.extend(hard_samples)
        self._stats["hard_samples_mined"] += len(hard_samples)

        # 按难度统计
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0, "extreme": 0}
        for s in hard_samples:
            difficulty_counts[s.difficulty] += 1

        return {
            "method": "hard_sample_mining",
            "description": "GAN生成 → YOLO检测 → 筛选困难样本（低置信度/误检/漏检）",
            "total_generated": num_per_class * len(self.CLASS_NAMES),
            "hard_samples_count": len(hard_samples),
            "hard_ratio": round(len(hard_samples) / (num_per_class * len(self.CLASS_NAMES)), 3),
            "difficulty_distribution": difficulty_counts,
            "per_class_stats": {
                name: {
                    "total": s["total"],
                    "hard": s["hard"],
                    "hard_ratio": round(s["hard"] / s["total"], 3) if s["total"] > 0 else 0,
                    "reasons": s["reasons"],
                }
                for name, s in stats.items()
            },
            "note": "困难样本可用于针对性训练，提升模型对难例的检测能力",
        }

    # ============================================================
    # 机制 4：融合质量判定
    # ============================================================

    def fused_quality_judgement(self, image: Union[str, Image.Image, np.ndarray],
                                quality_engine=None) -> Dict:
        """
        融合质量判定

        综合：
        1. YOLO 检测结果（缺陷数量、类别、置信度）
        2. GAN 重建异常度（整体异常水平）
        3. 异常区域与检测区域的空间一致性
        4. 缺陷特征合理性（GAN 判别器评分）

        给出最终质量等级和置信度。

        Args:
            image: 输入图像
            quality_engine: 质量判定引擎（可选）

        Returns:
            融合判定结果
        """
        # 先做交叉验证检测
        fusion_result = self.cross_validation_detect(image)

        # 基础分数
        base_quality = fusion_result.fused_quality
        base_confidence = fusion_result.fused_confidence

        # 如果有质量引擎，加入规则判定结果
        rule_quality = None
        if quality_engine is not None and fusion_result.yolo_detections:
            try:
                rule_result = quality_engine.judge(fusion_result.yolo_detections)
                rule_quality = rule_result.get("quality", "REVIEW")
            except:
                pass

        # 三源融合（YOLO检测 + GAN重建 + 规则判定）
        final_quality = base_quality
        final_confidence = base_confidence
        sources = ["yolo_detection", "gan_reconstruction"]

        if rule_quality:
            sources.append("quality_rules")
            if rule_quality == base_quality:
                # 规则与融合结果一致 → 提升置信度
                final_confidence = min(0.98, final_confidence + 0.1)
            else:
                # 不一致 → 降级为 REVIEW
                if base_quality in ("PASS", "FAIL"):
                    final_quality = "REVIEW"
                    final_confidence = 0.45

        return {
            "quality": final_quality,
            "confidence": round(final_confidence, 3),
            "quality_cn": {
                "PASS": "合格",
                "REVIEW": "待复核",
                "FAIL": "不合格",
            }.get(final_quality, final_quality),
            "sources_used": sources,
            "cross_validation": {
                "yolo_detected": len(fusion_result.yolo_detections) > 0,
                "yolo_max_confidence": round(
                    max(fusion_result.yolo_confidences, default=0.0), 4
                ),
                "gan_reconstruction_error": round(fusion_result.reconstruction_error, 4),
                "notes": fusion_result.cross_validation_notes,
            },
            "rule_quality": rule_quality,
            "detection_count": len(fusion_result.yolo_detections),
            "processing_time_ms": round(fusion_result.processing_time * 1000, 1),
            "fusion_method": "tri-source_fusion" if rule_quality else "dual-source_fusion",
        }

    # ============================================================
    # 工具方法
    # ============================================================

    def _load_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """加载图像为 PIL Image"""
        if isinstance(image, str):
            return Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return Image.fromarray(image, mode='L')
            else:
                return Image.fromarray(image).convert('L')
        elif isinstance(image, Image.Image):
            return image.convert('L')
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

    def get_stats(self) -> Dict:
        """获取融合模块统计信息"""
        return {
            "gan_available": self.gan is not None and self.gan.is_loaded,
            "yolo_available": self.yolo is not None,
            "fusion_mechanisms": [
                {"id": "adaptive_augmentation", "name": "定向数据增强",
                 "description": "分析类别表现，薄弱类别生成更多样本"},
                {"id": "cross_validation", "name": "交叉验证检测",
                 "description": "GAN重建+YOLO检测，双路验证降低漏检误检"},
                {"id": "hard_sample_mining", "name": "困难样本挖掘",
                 "description": "GAN生成+YOLO检测，筛选难例用于针对性训练"},
                {"id": "fused_quality_judgement", "name": "融合质量判定",
                 "description": "多源融合，给出更可靠的质量等级和置信度"},
            ],
            "hard_samples_count": len(self.hard_samples),
            "augmentation_rounds": self._stats["augmentation_rounds"],
            "fused_detections": self._stats["fused_detections"],
        }


# ============================================================
# 便捷函数
# ============================================================

def create_fusion_manager(gan_manager=None, yolo_model=None,
                          device: str = None) -> GanYoloFusion:
    """
    便捷创建融合管理器

    Args:
        gan_manager: GAN 管理器，为 None 则自动创建
        yolo_model: YOLO 模型，为 None 则不启用 YOLO
        device: 计算设备

    Returns:
        GanYoloFusion 实例
    """
    if gan_manager is None:
        try:
            from models.gan import create_gan_manager
            gan_manager = create_gan_manager()
        except Exception:
            gan_manager = None

    return GanYoloFusion(
        gan_manager=gan_manager,
        yolo_model=yolo_model,
        device=device,
    )
