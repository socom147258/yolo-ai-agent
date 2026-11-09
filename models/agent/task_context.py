# -*- coding: utf-8 -*-
"""
任务上下文管理 —— 解决数据范围与状态污染问题

核心原则：
1. 原始检测集合 (raw_detections) 永远不可被修改
2. 筛选操作产生独立的子集视图 (subset_view)，不覆盖原始数据
3. 每个操作都有明确的 scope 标签（全图/子集/批次）
4. 质量判定必须标明基于哪个范围的数据
5. 从子集分析转回全量判定时，正确恢复全量范围
"""

import uuid
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Scope 枚举
SCOPE_FULL_IMAGE = 'full_image'       # 单图全量
SCOPE_SUBSET = 'subset'               # 类别筛选后的子集
SCOPE_BATCH = 'batch'                 # 批次全量
SCOPE_BATCH_SUBSET = 'batch_subset'   # 批次子集


@dataclass
class DetectionItem:
    """单个检测结果项（标准化格式）"""
    cls_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    det_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """检测框面积（像素平方）"""
        return self.width * self.height

    def to_tuple(self):
        """兼容旧格式：[cls_name, conf, x1, y1, x2, y2]"""
        return [self.cls_name, self.confidence, self.x1, self.y1, self.x2, self.y2]

    def copy(self) -> 'DetectionItem':
        """创建深拷贝（安全副本，修改不影响原对象）"""
        return DetectionItem(
            cls_name=self.cls_name,
            confidence=self.confidence,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            det_id=self.det_id,  # 保留原ID，方便追溯
        )

    @classmethod
    def from_tuple(cls, t):
        """从旧格式元组创建"""
        return cls(cls_name=t[0], confidence=float(t[1]),
                   x1=float(t[2]), y1=float(t[3]),
                   x2=float(t[4]), y2=float(t[5]))


@dataclass
class ImageInfo:
    """图像信息"""
    image_id: str
    image_name: str
    width: int
    height: int
    image_path: Optional[str] = None
    quality_score: Optional[float] = None  # 图像质量评分 (0-1)，None表示未评估
    quality_issues: List[str] = field(default_factory=list)  # 图像质量问题列表


@dataclass
class SubsetView:
    """子集视图——筛选结果的独立引用，不修改原始数据"""
    view_id: str
    scope: str               # subset / batch_subset
    parent_scope: str        # 父范围（筛选前的范围）
    filter_type: str         # 筛选类型：class / confidence / area
    filter_params: Dict[str, Any]  # 筛选参数
    det_ids: List[str]       # 子集包含的检测ID列表（引用，非拷贝）
    description: str         # 人类可读描述

    def is_empty(self) -> bool:
        return len(self.det_ids) == 0


class TaskContext:
    """
    任务上下文 —— Agent 执行过程中的状态管理

    核心设计：
    - raw_detections: 原始检测结果，只读，永远不被修改
    - current_scope: 当前分析范围（full_image / subset / batch）
    - current_view: 当前分析对应的子集视图（None表示全量）
    - 所有筛选操作都产生新的 SubsetView，不破坏原始数据
    """

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        self.created_at = datetime.now().isoformat()

        # 图像信息
        self.image: Optional[ImageInfo] = None

        # 原始检测结果（不可修改的基线数据）
        self._raw_detections: List[DetectionItem] = []
        self._raw_detection_set = False  # 是否已设置原始检测

        # 批次数据（如果是批次任务）
        self._batch_images: Dict[str, 'ImageInfo'] = {}
        self._batch_detections: Dict[str, List[DetectionItem]] = {}

        # 当前分析范围
        self.current_scope: str = SCOPE_FULL_IMAGE
        self._current_view: Optional[SubsetView] = None

        # 模型与规则版本
        self.model_version: str = "unknown"
        self.rule_version: str = "demo_v1"
        self.rule_source: str = "演示规则"  # 规则来源：演示/企业标准/自定义

        # 质量规则配置快照（判定时使用的规则版本）
        self.rule_snapshot: Optional[Dict] = None

        # 执行历史中的所有子集视图
        self._views: List[SubsetView] = []

        # 检测失败/异常信息
        self.detection_errors: List[str] = []
        self.batch_errors: Dict[str, str] = {}  # image_id -> error_msg

    # ==================== 原始检测设置 ====================

    def set_raw_detections(self, detections: List[DetectionItem], image: ImageInfo):
        """设置原始检测结果（只能设置一次）"""
        if self._raw_detection_set:
            raise RuntimeError("原始检测结果已设置，不可覆盖。如需重新检测请创建新的任务上下文。")
        self._raw_detections = list(detections)
        self.image = image
        self._raw_detection_set = True
        # 默认 scope 为全图
        self.current_scope = SCOPE_FULL_IMAGE
        self._current_view = None

    def set_batch_detections(self, image_results: Dict[str, List[DetectionItem]],
                              images: Dict[str, ImageInfo]):
        """设置批次检测结果"""
        self._batch_detections = {k: list(v) for k, v in image_results.items()}
        self._batch_images = dict(images)
        self.current_scope = SCOPE_BATCH
        self._raw_detection_set = True
        # 批次模式下，image 存第一张的信息（方便显示）
        first_id = list(images.keys())[0] if images else None
        if first_id:
            self.image = images[first_id]

    @property
    def has_raw_detections(self) -> bool:
        return self._raw_detection_set

    # ==================== 获取当前范围的检测 ====================

    def get_current_detections(self) -> List[DetectionItem]:
        """
        获取当前分析范围内的检测结果
        返回的是深拷贝列表，修改返回值不会影响上下文原始数据
        """
        if self.current_scope == SCOPE_FULL_IMAGE:
            return [d.copy() for d in self._raw_detections]
        elif self.current_scope == SCOPE_SUBSET:
            if self._current_view is None:
                return []
            id_set = set(self._current_view.det_ids)
            return [d.copy() for d in self._raw_detections if d.det_id in id_set]
        elif self.current_scope == SCOPE_BATCH:
            # 批次全量：汇总所有图像的检测
            all_dets = []
            for dets in self._batch_detections.values():
                all_dets.extend(d.copy() for d in dets)
            return all_dets
        elif self.current_scope == SCOPE_BATCH_SUBSET:
            if self._current_view is None:
                return []
            id_set = set(self._current_view.det_ids)
            all_dets = []
            for dets in self._batch_detections.values():
                all_dets.extend(d.copy() for d in dets if d.det_id in id_set)
            return all_dets
        else:
            return []

    def get_raw_detections(self) -> List[DetectionItem]:
        """获取原始全量检测结果（深拷贝，外部修改不影响上下文）"""
        return [d.copy() for d in self._raw_detections]

    # ==================== 筛选操作 ====================

    def filter_by_class(self, target_class: str) -> SubsetView:
        """
        按类别筛选，生成新的子集视图
        不修改原始数据，不自动切换 scope（由调用方决定是否切换）
        """
        if self.current_scope in (SCOPE_FULL_IMAGE, SCOPE_SUBSET):
            source = self.get_current_detections()
            parent_scope = self.current_scope
            new_scope = SCOPE_SUBSET
        else:
            source = self.get_current_detections()
            parent_scope = self.current_scope
            new_scope = SCOPE_BATCH_SUBSET

        filtered = [d for d in source if d.cls_name == target_class]
        det_ids = [d.det_id for d in filtered]

        view = SubsetView(
            view_id=f"view_{uuid.uuid4().hex[:8]}",
            scope=new_scope,
            parent_scope=parent_scope,
            filter_type='class',
            filter_params={'target_class': target_class},
            det_ids=det_ids,
            description=f"类别筛选：{target_class}（{len(filtered)}个）"
        )
        self._views.append(view)
        return view

    def switch_to_view(self, view: SubsetView):
        """切换到某个子集视图"""
        self._current_view = view
        self.current_scope = view.scope

    def switch_to_full(self):
        """切换回全量范围（单图或批次）"""
        self._current_view = None
        if self._batch_detections:
            self.current_scope = SCOPE_BATCH
        else:
            self.current_scope = SCOPE_FULL_IMAGE

    # ==================== 范围信息 ====================

    @property
    def scope_label(self) -> str:
        """当前范围的人类可读标签"""
        labels = {
            SCOPE_FULL_IMAGE: '单图全量',
            SCOPE_SUBSET: '单图子集',
            SCOPE_BATCH: '批次全量',
            SCOPE_BATCH_SUBSET: '批次子集',
        }
        return labels.get(self.current_scope, self.current_scope)

    @property
    def image_count(self) -> int:
        """当前范围涉及的图像数量"""
        if self.current_scope in (SCOPE_FULL_IMAGE, SCOPE_SUBSET):
            return 1 if self.image else 0
        else:
            return len(self._batch_images)

    @property
    def is_batch(self) -> bool:
        return self.current_scope in (SCOPE_BATCH, SCOPE_BATCH_SUBSET)

    @property
    def is_subset(self) -> bool:
        return self.current_scope in (SCOPE_SUBSET, SCOPE_BATCH_SUBSET)

    # ==================== 范围判定安全检查 ====================

    def can_judge_quality(self, target_scope: str = None) -> tuple:
        """
        检查是否可以执行质量判定（针对目标范围）
        返回 (can_judge: bool, reason: str)

        安全规则：
        - 子集范围不能输出整图/整批质量结论
        - 检测失败的不能输出 PASS
        - 批次未完成的不能输出整批 PASS
        """
        target = target_scope or self.current_scope

        if not self._raw_detection_set and not self._batch_detections:
            return False, "没有检测数据，无法判定"

        # 子集判定只能输出子集结论
        if target in (SCOPE_SUBSET, SCOPE_BATCH_SUBSET):
            return True, "子集范围判定（仅代表筛选后的子集，不代表整体）"

        # 全量判定需要检查是否有检测错误
        if target == SCOPE_FULL_IMAGE:
            if self.detection_errors:
                return False, f"检测存在错误：{'; '.join(self.detection_errors)}"
            return True, "单图全量判定"

        if target == SCOPE_BATCH:
            if self.batch_errors:
                err_count = len(self.batch_errors)
                total = len(self._batch_images) + len(self.batch_errors)
                return False, f"批次中 {err_count}/{total} 张图像检测失败，不能输出整批结论"
            return True, f"批次全量判定（共 {len(self._batch_images)} 张图像）"

        return True, "判定"

    # ==================== 图像尺寸 ====================

    @property
    def img_width(self) -> int:
        return self.image.width if self.image else 640

    @property
    def img_height(self) -> int:
        return self.image.height if self.image else 640

    # ==================== 辅助 ====================

    def get_detection_by_id(self, det_id: str) -> Optional[DetectionItem]:
        """根据 ID 查找检测项"""
        for d in self._raw_detections:
            if d.det_id == det_id:
                return d
        for dets in self._batch_detections.values():
            for d in dets:
                if d.det_id == det_id:
                    return d
        return None

    def snapshot(self) -> Dict:
        """生成当前上下文的快照（用于报告和追踪）"""
        current_dets = self.get_current_detections()
        return {
            'task_id': self.task_id,
            'scope': self.current_scope,
            'scope_label': self.scope_label,
            'is_subset': self.is_subset,
            'image_count': self.image_count,
            'detection_count': len(current_dets),
            'model_version': self.model_version,
            'rule_version': self.rule_version,
            'rule_source': self.rule_source,
            'subset_description': self._current_view.description if self._current_view else None,
            'detection_errors': list(self.detection_errors),
            'batch_error_count': len(self.batch_errors),
        }
