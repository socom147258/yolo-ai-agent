# -*- coding: utf-8 -*-
"""
工具函数模块

提供图像读取、检测结果序列化、响应包装等通用工具函数。
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import UploadFile

from models.agent import DetectionItem, ImageInfo, TaskContext
from api.dependencies import get_class_name_cn


# ============================================================
# 响应包装
# ============================================================

def success_response(data: Any = None, message: str = "success") -> Dict[str, Any]:
    """
    构建成功响应

    Args:
        data: 响应数据
        message: 响应消息

    Returns:
        标准响应字典
    """
    return {
        "code": 0,
        "message": message,
        "data": data,
    }


def error_response(code: int = 1, message: str = "error", data: Any = None) -> Dict[str, Any]:
    """
    构建错误响应

    Args:
        code: 错误码（非 0）
        message: 错误信息
        data: 附加数据

    Returns:
        标准响应字典
    """
    return {
        "code": code,
        "message": message,
        "data": data,
    }


# ============================================================
# 文件处理
# ============================================================

def save_upload_file(upload_file: UploadFile, output_dir: str) -> Tuple[str, str]:
    """
    保存上传的文件到临时目录

    Args:
        upload_file: FastAPI 上传文件对象
        output_dir: 输出目录

    Returns:
        (保存的文件路径, 原始文件名)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名，避免冲突
    original_name = upload_file.filename or "unknown.jpg"
    ext = Path(original_name).suffix or ".jpg"
    unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(output_dir, unique_name)

    # 写入文件
    with open(save_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)

    return save_path, original_name


def save_upload_files(upload_files: List[UploadFile], output_dir: str) -> List[Tuple[str, str]]:
    """
    批量保存上传的文件

    Args:
        upload_files: FastAPI 上传文件对象列表
        output_dir: 输出目录

    Returns:
        [(保存路径, 原始文件名), ...]
    """
    results = []
    for upload_file in upload_files:
        path, name = save_upload_file(upload_file, output_dir)
        results.append((path, name))
    return results


# ============================================================
# 检测结果序列化
# ============================================================

def detection_item_to_dict(item: DetectionItem) -> Dict[str, Any]:
    """
    将 DetectionItem 转换为字典（用于 API 响应）

    Args:
        item: 检测项

    Returns:
        序列化为字典的检测结果
    """
    return {
        "cls_name": item.cls_name,
        "cls_name_cn": get_class_name_cn(item.cls_name),
        "confidence": round(item.confidence, 4),
        "x1": round(item.x1, 2),
        "y1": round(item.y1, 2),
        "x2": round(item.x2, 2),
        "y2": round(item.y2, 2),
        "area": round(item.area, 2),
    }


def detections_to_list(detections: List[DetectionItem]) -> List[Dict[str, Any]]:
    """
    将检测结果列表转换为字典列表

    Args:
        detections: 检测结果列表

    Returns:
        字典列表
    """
    return [detection_item_to_dict(d) for d in detections]


def image_info_to_dict(image_info: ImageInfo) -> Dict[str, Any]:
    """
    将 ImageInfo 转换为字典

    Args:
        image_info: 图像信息

    Returns:
        字典
    """
    return {
        "image_id": image_info.image_id,
        "image_name": image_info.image_name,
        "width": image_info.width,
        "height": image_info.height,
        "quality_score": image_info.quality_score,
        "quality_issues": image_info.quality_issues,
    }


# ============================================================
# 质量结果处理
# ============================================================

def quality_result_to_dict(quality_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理质量判定结果字典，确保字段完整且类型正确

    Args:
        quality_result: 原始质量判定结果

    Returns:
        处理后的质量结果字典
    """
    # 确保所有字段都存在
    result = {
        "quality": quality_result.get("quality", "REVIEW"),
        "score": quality_result.get("score", 0),
        "total_defects": quality_result.get("total_defects", 0),
        "critical_count": quality_result.get("critical_count", 0),
        "low_conf_count": quality_result.get("low_conf_count", 0),
        "total_box_area_pct": quality_result.get("total_box_area_pct", 0.0),
        "max_single_box_area_pct": quality_result.get("max_single_box_area_pct", 0.0),
        "max_severity": quality_result.get("max_severity", 0),
        "by_class": quality_result.get("by_class", {}),
        "reasons_fail": quality_result.get("reasons_fail", []),
        "reasons_review": quality_result.get("reasons_review", []),
        "explanation": quality_result.get("explanation", ""),
        "scope": quality_result.get("scope", ""),
        "scope_label": quality_result.get("scope_label", ""),
        "method_notes": quality_result.get("method_notes", []),
        "is_forced": quality_result.get("is_forced", False),
        "forced_reason": quality_result.get("forced_reason", ""),
        "rule_id": quality_result.get("rule_id", ""),
        "rule_version": quality_result.get("rule_version", ""),
        "rule_source": quality_result.get("rule_source", ""),
        "is_demo_rule": quality_result.get("is_demo_rule", True),
    }
    return result


# ============================================================
# 批次结果处理
# ============================================================

def batch_result_to_dict(batch_result) -> Dict[str, Any]:
    """
    将 BatchResult 转换为 API 响应格式的字典

    Args:
        batch_result: BatchResult 实例

    Returns:
        字典
    """
    raw = batch_result.to_dict()
    summary = raw.get("summary", {})

    # 标准化摘要字段名
    normalized_summary = {
        "planned": summary.get("planned", 0),
        "success": summary.get("success", 0),
        "failed": summary.get("failed", 0),
        "skipped": summary.get("skipped", 0),
        "pass_count": summary.get("pass", 0),
        "fail_count": summary.get("fail", 0),
        "review_count": summary.get("review", 0),
    }

    # 处理各图像结果
    images = []
    for img in raw.get("images", []):
        images.append({
            "image_name": img.get("image_name", ""),
            "status": img.get("status", ""),
            "detection_count": img.get("detection_count", 0),
            "quality": img.get("quality", ""),
            "quality_score": img.get("quality_score", 0),
            "scope": img.get("scope", ""),
            "error": img.get("error", ""),
            "is_forced": img.get("is_forced", False),
            "forced_reason": img.get("forced_reason", ""),
            "review_task_id": img.get("review_task_id", ""),
            "method_notes": img.get("method_notes", []),
        })

    return {
        "batch_id": raw.get("batch_id", ""),
        "summary": normalized_summary,
        "batch_quality": raw.get("batch_quality", ""),
        "batch_quality_reason": raw.get("batch_quality_reason", ""),
        "model_version": raw.get("model_version", ""),
        "rule_version": raw.get("rule_version", ""),
        "rule_is_demo": raw.get("rule_is_demo", True),
        "start_time": raw.get("start_time", ""),
        "end_time": raw.get("end_time", ""),
        "images": images,
    }


# ============================================================
# 复核任务处理
# ============================================================

def review_task_to_dict(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    将复核任务字典转换为 API 响应格式

    Args:
        task: 复核任务原始字典

    Returns:
        标准化字典
    """
    return {
        "id": task.get("id"),
        "review_id": task.get("review_id", ""),
        "task_id": task.get("task_id", ""),
        "record_id": task.get("record_id"),
        "image_name": task.get("image_name", ""),
        "image_path": task.get("image_path", ""),
        "original_quality": task.get("original_quality", ""),
        "original_score": task.get("original_score", 0),
        "original_scope": task.get("original_scope", ""),
        "quality_issue": task.get("quality_issue", ""),
        "reason": task.get("reason", ""),
        "priority": task.get("priority", "normal"),
        "status": task.get("status", "pending"),
        "created_at": task.get("created_at", ""),
        "assigned_to": task.get("assigned_to", ""),
        "resolved_at": task.get("resolved_at", ""),
        "resolved_by": task.get("resolved_by", ""),
        "resolution": task.get("resolution", ""),
        "resolution_note": task.get("resolution_note", ""),
        "model_version": task.get("model_version", ""),
        "rule_version": task.get("rule_version", ""),
    }


# ============================================================
# 检测请求构建
# ============================================================

def build_task_context_from_detections(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    image_quality_score: Optional[float] = None,
    task_id: str = None,
) -> TaskContext:
    """
    从检测结果字典构建 TaskContext（用于质量判定）

    Args:
        detections: 检测结果字典列表
        image_width: 图像宽度
        image_height: 图像高度
        image_quality_score: 图像质量评分
        task_id: 任务ID

    Returns:
        TaskContext 实例
    """
    ctx = TaskContext(task_id=task_id)

    # 构建 DetectionItem 列表
    det_items = []
    for d in detections:
        det_items.append(DetectionItem(
            cls_name=d.get("cls_name", "unknown"),
            confidence=float(d.get("confidence", 0.0)),
            x1=float(d.get("x1", 0.0)),
            y1=float(d.get("y1", 0.0)),
            x2=float(d.get("x2", 0.0)),
            y2=float(d.get("y2", 0.0)),
        ))

    # 构建 ImageInfo
    image_info = ImageInfo(
        image_id=task_id or "manual",
        image_name="manual_input",
        width=image_width,
        height=image_height,
        quality_score=image_quality_score,
    )

    ctx.set_raw_detections(det_items, image_info)
    return ctx


# ============================================================
# 报告相关
# ============================================================

def generate_report_id(prefix: str = "RPT") -> str:
    """
    生成报告ID

    Args:
        prefix: 前缀

    Returns:
        报告ID
    """
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"


def get_report_path(report_id: str, output_dir: str) -> str:
    """
    根据报告ID获取报告文件路径

    Args:
        report_id: 报告ID
        output_dir: 输出目录

    Returns:
        报告文件路径
    """
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return os.path.join(reports_dir, f"{report_id}.docx")
