# -*- coding: utf-8 -*-
"""
批次处理模块

特点：
- 支持批量图像检测和质量判定
- 跟踪计划数/成功数/失败数/未处理数
- 部分完成不能整批 PASS
- 幂等的复核创建（避免重复创建）
- 批次结果可用于报告生成
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from models.agent.task_context import TaskContext, DetectionItem, ImageInfo
from models.agent.quality_engine_v2 import QualityEngineV2
from models.agent.review_workflow import ReviewWorkflow


@dataclass
class BatchImageResult:
    """单张图像的批次处理结果"""
    image_name: str
    image_path: str
    status: str = 'pending'  # pending / success / failed / skipped
    detection_count: int = 0
    quality: str = ''       # PASS / FAIL / REVIEW / ''
    quality_score: float = 0
    scope: str = ''
    error_message: str = ''
    is_forced: bool = False
    forced_reason: str = ''
    detection_time_ms: float = 0
    quality_time_ms: float = 0
    review_task_id: str = ''
    method_notes: List[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """批次处理结果"""
    batch_id: str
    total_planned: int = 0
    total_success: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    image_results: List[BatchImageResult] = field(default_factory=list)
    start_time: str = ''
    end_time: str = ''
    model_version: str = ''
    rule_version: str = ''
    rule_is_demo: bool = True
    batch_quality: str = ''  # PASS / FAIL / REVIEW / INCOMPLETE
    batch_quality_reason: str = ''
    pass_count: int = 0
    fail_count: int = 0
    review_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'batch_id': self.batch_id,
            'summary': {
                'planned': self.total_planned,
                'success': self.total_success,
                'failed': self.total_failed,
                'skipped': self.total_skipped,
                'pass': self.pass_count,
                'fail': self.fail_count,
                'review': self.review_count,
            },
            'batch_quality': self.batch_quality,
            'batch_quality_reason': self.batch_quality_reason,
            'model_version': self.model_version,
            'rule_version': self.rule_version,
            'rule_is_demo': self.rule_is_demo,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'images': [
                {
                    'image_name': r.image_name,
                    'status': r.status,
                    'detection_count': r.detection_count,
                    'quality': r.quality,
                    'quality_score': r.quality_score,
                    'scope': r.scope,
                    'error': r.error_message,
                    'is_forced': r.is_forced,
                    'forced_reason': r.forced_reason,
                    'review_task_id': r.review_task_id,
                    'method_notes': r.method_notes,
                }
                for r in self.image_results
            ]
        }


class BatchProcessor:
    """
    批次处理器

    安全保证：
    - 部分完成的批次不能输出 PASS
    - 失败/未处理的图像会拉低整体结论
    - 复核任务幂等创建（同一张图不重复创建）
    - 完整记录每一步的成功/失败状态
    """

    def __init__(self, quality_engine: QualityEngineV2,
                 review_workflow: Optional[ReviewWorkflow] = None,
                 model_version: str = 'unknown'):
        self.quality_engine = quality_engine
        self.review_workflow = review_workflow
        self.model_version = model_version
        self._review_cache = set()  # 防止重复创建复核任务

    def process_batch(self, image_paths: List[str], detect_func,
                      batch_id: str = None) -> BatchResult:
        """
        处理一批图像

        Args:
            image_paths: 图像路径列表
            detect_func: 检测函数 (image_path) -> (detections, image_info, error_msg)
            batch_id: 批次ID（自动生成）

        Returns:
            BatchResult
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = BatchResult(
            batch_id=batch_id,
            total_planned=len(image_paths),
            start_time=datetime.now().isoformat(),
            model_version=self.model_version,
            rule_version=f"{self.quality_engine.rule_set.rule_id}_v{self.quality_engine.rule_set.version}",
            rule_is_demo=self.quality_engine.rule_set.is_demo,
        )

        for img_path in image_paths:
            img_result = self._process_single_image(img_path, detect_func, batch_id)
            result.image_results.append(img_result)

            if img_result.status == 'success':
                result.total_success += 1
                if img_result.quality == 'PASS':
                    result.pass_count += 1
                elif img_result.quality == 'FAIL':
                    result.fail_count += 1
                elif img_result.quality == 'REVIEW':
                    result.review_count += 1
            elif img_result.status == 'failed':
                result.total_failed += 1
            else:
                result.total_skipped += 1

        result.end_time = datetime.now().isoformat()

        # 批次质量判定
        self._compute_batch_quality(result)

        return result

    def _process_single_image(self, image_path: str, detect_func,
                               batch_id: str) -> BatchImageResult:
        """处理单张图像"""
        img_name = Path(image_path).name
        result = BatchImageResult(image_name=img_name, image_path=image_path)

        try:
            import time
            t0 = time.time()

            # 调用检测函数
            detections, image_info, error_msg = detect_func(image_path)

            result.detection_time_ms = (time.time() - t0) * 1000

            if error_msg:
                result.status = 'failed'
                result.error_message = error_msg
                return result

            result.detection_count = len(detections)

            # 质量判定
            ctx = TaskContext(f"{batch_id}_{img_name}")
            ctx.model_version = self.model_version
            ctx.rule_version = f"{self.quality_engine.rule_set.rule_id}_v{self.quality_engine.rule_set.version}"
            ctx.set_raw_detections(detections, image_info)

            t1 = time.time()
            quality_result = self.quality_engine.evaluate(ctx, scope='full')
            result.quality_time_ms = (time.time() - t1) * 1000

            result.quality = quality_result['quality']
            result.quality_score = quality_result.get('score', 0)
            result.scope = quality_result.get('scope_label', '单图全量')
            result.is_forced = quality_result.get('is_forced', False)
            result.forced_reason = quality_result.get('forced_reason', '')
            result.method_notes = quality_result.get('method_notes', [])
            result.status = 'success'

            # 创建复核任务（幂等）
            if quality_result['quality'] in ('FAIL', 'REVIEW'):
                result.review_task_id = self._create_review_if_needed(
                    image_path, img_name, quality_result, batch_id
                )

        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)

        return result

    def _create_review_if_needed(self, image_path: str, image_name: str,
                                  quality_result: Dict, batch_id: str) -> str:
        """幂等创建复核任务（同一批次同一张图只创建一次）"""
        if not self.review_workflow:
            return ''

        cache_key = f"{batch_id}_{image_name}"
        if cache_key in self._review_cache:
            return ''  # 已创建过，跳过

        try:
            rid = self.review_workflow.create_task(
                task_id=batch_id,
                record_id=hash(cache_key) % (2**31),
                image_name=image_name,
                image_path=image_path,
                original_quality=quality_result['quality'],
                original_score=quality_result.get('score', 0),
                original_scope=quality_result.get('scope_label', '单图全量'),
                reason=quality_result.get('forced_reason', f"自动检测结果为{quality_result['quality']}"),
                quality_issue=f"{quality_result['quality']}（需人工确认）",
                priority='high' if quality_result['quality'] == 'FAIL' else 'medium',
                model_version=self.model_version,
                rule_version=f"{self.quality_engine.rule_set.rule_id}_v{self.quality_engine.rule_set.version}",
            )
            self._review_cache.add(cache_key)
            return rid
        except Exception:
            return ''

    def _compute_batch_quality(self, result: BatchResult):
        """
        计算批次整体质量等级

        安全规则：
        - 有任何失败/未处理 → INCOMPLETE（不能 PASS）
        - 有任何 FAIL → 批次 FAIL
        - 有任何 REVIEW 且无 FAIL → 批次 REVIEW
        - 全部 PASS 且全部成功 → 批次 PASS
        """
        if result.total_failed > 0 or result.total_skipped > 0:
            result.batch_quality = 'INCOMPLETE'
            result.batch_quality_reason = (
                f"批次未全部完成：成功 {result.total_success}/{result.total_planned}，"
                f"失败 {result.total_failed}，跳过 {result.total_skipped}。"
                f"未完成的批次不能输出 PASS。"
            )
        elif result.fail_count > 0:
            result.batch_quality = 'FAIL'
            result.batch_quality_reason = (
                f"批次中有 {result.fail_count} 张图像判定为 FAIL，整批评定为 FAIL。"
            )
        elif result.review_count > 0:
            result.batch_quality = 'REVIEW'
            result.batch_quality_reason = (
                f"批次中有 {result.review_count} 张图像需人工复核，整批评定为 REVIEW。"
            )
        elif result.pass_count == result.total_success and result.total_success == result.total_planned:
            result.batch_quality = 'PASS'
            result.batch_quality_reason = (
                f"全部 {result.total_planned} 张图像均判定为 PASS，批次判定为 PASS。"
            )
        else:
            result.batch_quality = 'INCOMPLETE'
            result.batch_quality_reason = '无法确定批次质量等级。'


def detect_with_yolo(model, image_path: str) -> Tuple[List[DetectionItem], ImageInfo, str]:
    """
    使用 YOLO 模型检测图像（适配 BatchProcessor 的 detect_func 接口）

    返回: (detections, image_info, error_msg)
    error_msg 为空表示成功
    """
    import cv2
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(image_path)
        w, h = img.size

        # 运行推理
        results = model.predict(source=np.array(img), verbose=False)
        result = results[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            cls_names = result.names

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = cls_names[cls_id]
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                detections.append(DetectionItem(
                    cls_name=cls_name,
                    confidence=conf,
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                ))

        # 图像质量评估（简单版：基于清晰度）
        quality_score = 0.85
        quality_issues = []
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) if img.mode == 'RGB' else np.array(img)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # 简单映射：方差越大越清晰
            quality_score = min(1.0, max(0.3, laplacian_var / 1000))
            if quality_score < 0.5:
                quality_issues.append('图像模糊')
        except Exception:
            pass

        image_info = ImageInfo(
            image_id=Path(image_path).stem,
            image_name=Path(image_path).name,
            width=w,
            height=h,
            quality_score=round(quality_score, 3),
            quality_issues=quality_issues,
        )

        return detections, image_info, ''

    except Exception as e:
        return [], ImageInfo(
            image_id=Path(image_path).stem,
            image_name=Path(image_path).name,
            width=0, height=0
        ), str(e)
