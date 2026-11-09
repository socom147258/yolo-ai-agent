# -*- coding: utf-8 -*-
"""
工具 Schema 定义 —— 每个工具的参数和返回值结构

所有工具必须遵循明确的输入输出 Schema，支持：
- 参数校验
- 返回值结构验证
- 超时控制
- 有限重试
- 费用统计（如使用 LLM）
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(str, Enum):
    SUCCESS = 'success'
    ERROR = 'error'
    TIMEOUT = 'timeout'
    VALIDATION_ERROR = 'validation_error'
    NOT_FOUND = 'not_found'


@dataclass
class ToolParamSchema:
    """参数 Schema 定义"""
    name: str
    type: str                    # str / int / float / bool / list / dict
    required: bool = False
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None  # 枚举值
    min_value: Optional[float] = None    # 数值最小值
    max_value: Optional[float] = None    # 数值最大值


@dataclass
class ToolSchema:
    """工具完整 Schema"""
    name: str
    display_name: str
    description: str
    category: str = "analysis"   # detection / analysis / quality / report / review
    params: List[ToolParamSchema] = field(default_factory=list)
    returns: List[str] = field(default_factory=list)  # 返回字段列表
    timeout_sec: int = 30
    max_retries: int = 0
    cost_credits: int = 0        # 费用（积分），0表示免费工具

    def validate_params(self, params: Dict[str, Any]) -> tuple:
        """
        校验参数
        返回 (is_valid: bool, errors: List[str], cleaned_params: Dict)
        """
        errors = []
        cleaned = {}

        # 构建参数 schema 字典
        param_schemas = {p.name: p for p in self.params}

        # 检查必选参数
        for ps in self.params:
            if ps.required and ps.name not in params:
                errors.append(f"缺少必选参数: {ps.name}")

        # 检查未知参数（只警告，不拒绝）
        for k in params:
            if k not in param_schemas:
                # 未知参数直接透传（可能是内部上下文参数）
                cleaned[k] = params[k]

        # 类型校验和默认值
        for ps in self.params:
            val = params.get(ps.name, ps.default)

            if val is None:
                if ps.required:
                    errors.append(f"参数 {ps.name} 不能为空")
                cleaned[ps.name] = ps.default
                continue

            # 类型转换和校验
            try:
                if ps.type == 'str':
                    val = str(val)
                elif ps.type == 'int':
                    val = int(val)
                elif ps.type == 'float':
                    val = float(val)
                elif ps.type == 'bool':
                    if isinstance(val, str):
                        val = val.lower() in ('true', '1', 'yes', '是')
                    else:
                        val = bool(val)
            except (ValueError, TypeError):
                errors.append(f"参数 {ps.name} 类型错误，应为 {ps.type}")
                continue

            # 范围校验
            if ps.type in ('int', 'float'):
                if ps.min_value is not None and val < ps.min_value:
                    errors.append(f"参数 {ps.name} 不能小于 {ps.min_value}")
                if ps.max_value is not None and val > ps.max_value:
                    errors.append(f"参数 {ps.name} 不能大于 {ps.max_value}")

            # 枚举校验
            if ps.choices and val not in ps.choices:
                errors.append(f"参数 {ps.name} 值 {val} 不在允许范围内: {ps.choices}")

            cleaned[ps.name] = val

        return len(errors) == 0, errors, cleaned


@dataclass
class ToolResult:
    """工具执行结果（标准化格式）"""
    tool_name: str
    status: ToolStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    execution_time_ms: float = 0.0
    retry_count: int = 0
    cost_credits: int = 0

    def to_dict(self) -> Dict:
        return {
            'tool': self.tool_name,
            'status': self.status.value,
            'data': self.data,
            'error': self.error_message,
            'execution_time_ms': round(self.execution_time_ms, 2),
            'retry_count': self.retry_count,
            'cost_credits': self.cost_credits,
        }

    @classmethod
    def success(cls, tool_name: str, data: Dict, exec_ms: float = 0) -> 'ToolResult':
        return cls(tool_name=tool_name, status=ToolStatus.SUCCESS,
                   data=data, execution_time_ms=exec_ms)

    @classmethod
    def error(cls, tool_name: str, message: str, exec_ms: float = 0) -> 'ToolResult':
        return cls(tool_name=tool_name, status=ToolStatus.ERROR,
                   error_message=message, execution_time_ms=exec_ms)

    @classmethod
    def validation_error(cls, tool_name: str, errors: List[str]) -> 'ToolResult':
        return cls(tool_name=tool_name, status=ToolStatus.VALIDATION_ERROR,
                   error_message="参数校验失败: " + "; ".join(errors))

    @classmethod
    def timeout(cls, tool_name: str, exec_ms: float) -> 'ToolResult':
        return cls(tool_name=tool_name, status=ToolStatus.TIMEOUT,
                   error_message=f"执行超时（{exec_ms:.0f}ms）",
                   execution_time_ms=exec_ms)


# ============================================================
# 预定义工具 Schema
# ============================================================

TOOL_SCHEMAS: Dict[str, ToolSchema] = {}


def register_tool(schema: ToolSchema):
    """注册工具 Schema"""
    TOOL_SCHEMAS[schema.name] = schema


# ---------- 1. 图像质量检查 ----------
register_tool(ToolSchema(
    name='image_quality_check',
    display_name='图像质量检查',
    description='检查输入图像的质量（模糊度、亮度、对比度等），用于判断是否适合检测',
    category='detection',
    params=[
        ToolParamSchema('image_path', 'str', required=True, description='图像文件路径'),
    ],
    returns=['quality_score', 'issues', 'is_usable'],
    timeout_sec=10,
))

# ---------- 2. 图像缺陷检测 ----------
register_tool(ToolSchema(
    name='detect_image',
    display_name='图像缺陷检测',
    description='对单张图像运行 YOLO 缺陷检测，返回检测框、类别和置信度',
    category='detection',
    params=[
        ToolParamSchema('image_path', 'str', required=True, description='图像文件路径'),
        ToolParamSchema('conf_threshold', 'float', default=0.25,
                        min_value=0.01, max_value=0.99, description='置信度阈值'),
        ToolParamSchema('iou_threshold', 'float', default=0.45,
                        min_value=0.01, max_value=0.99, description='NMS IoU 阈值'),
    ],
    returns=['detections', 'total_count', 'inference_time_ms', 'img_width', 'img_height'],
    timeout_sec=60,
))

# ---------- 3. 批次检测 ----------
register_tool(ToolSchema(
    name='detect_batch',
    display_name='批次缺陷检测',
    description='对一批图像依次执行缺陷检测',
    category='detection',
    params=[
        ToolParamSchema('image_paths', 'list', required=True, description='图像路径列表'),
        ToolParamSchema('conf_threshold', 'float', default=0.25,
                        min_value=0.01, max_value=0.99, description='置信度阈值'),
    ],
    returns=['results', 'success_count', 'fail_count', 'total_time_ms'],
    timeout_sec=300,
))

# ---------- 4. 缺陷统计 ----------
register_tool(ToolSchema(
    name='count_defects',
    display_name='缺陷统计',
    description='统计当前范围内的缺陷数量和类别分布',
    category='analysis',
    params=[
        ToolParamSchema('scope', 'str', default='current',
                        choices=['current', 'full', 'batch'],
                        description='统计范围：当前范围/全量/批次'),
    ],
    returns=['total_count', 'by_class', 'confidence_stats'],
    timeout_sec=5,
))

# ---------- 5. 类别筛选 ----------
register_tool(ToolSchema(
    name='filter_by_class',
    display_name='按类别筛选',
    description='按缺陷类别筛选检测结果，生成子集视图（不修改原始数据）',
    category='analysis',
    params=[
        ToolParamSchema('target_class', 'str', required=True,
                        choices=['crazing', 'inclusion', 'patches', 'pitted_surface',
                                 'rolled-in_scale', 'scratches'],
                        description='目标缺陷类别'),
        ToolParamSchema('switch_scope', 'bool', default=True,
                        description='是否切换到筛选后的子集范围'),
    ],
    returns=['filtered_count', 'target_class', 'view_id', 'description'],
    timeout_sec=5,
))

# ---------- 6. 面积分析 ----------
register_tool(ToolSchema(
    name='area_analysis',
    display_name='面积分析',
    description='分析缺陷的检测框面积占比（注意：这是检测框覆盖率，非真实缺陷面积）',
    category='analysis',
    params=[
        ToolParamSchema('scope', 'str', default='current',
                        choices=['current', 'full'],
                        description='分析范围'),
        ToolParamSchema('overlap_method', 'str', default='union',
                        choices=['union', 'sum'],
                        description='重叠框计算方式：并集去重/简单求和'),
    ],
    returns=['total_area_pct', 'max_single_area_pct', 'avg_area_pct',
             'area_by_class', 'method_note'],
    timeout_sec=5,
))

# ---------- 7. 置信度分布 ----------
register_tool(ToolSchema(
    name='confidence_dist',
    display_name='置信度分布',
    description='分析检测结果的置信度分布',
    category='analysis',
    params=[
        ToolParamSchema('scope', 'str', default='current',
                        choices=['current', 'full'],
                        description='分析范围'),
    ],
    returns=['avg_confidence', 'min_confidence', 'max_confidence',
             'low_conf_count', 'high_conf_count', 'distribution'],
    timeout_sec=5,
))

# ---------- 8. 质量判定 ----------
register_tool(ToolSchema(
    name='quality_judge',
    display_name='质量判定',
    description='根据质量规则对当前范围的检测结果进行三级判定（PASS/REVIEW/FAIL）',
    category='quality',
    params=[
        ToolParamSchema('scope', 'str', default='current',
                        choices=['current', 'full'],
                        description='判定范围'),
        ToolParamSchema('rule_set', 'str', default='default',
                        description='规则集名称'),
    ],
    returns=['quality', 'score', 'reasons_fail', 'reasons_review',
             'scope_label', 'rule_version', 'method_notes'],
    timeout_sec=10,
))

# ---------- 9. 规则检索 ----------
register_tool(ToolSchema(
    name='rule_lookup',
    display_name='规则检索',
    description='查询当前生效的质量规则配置和阈值',
    category='quality',
    params=[
        ToolParamSchema('rule_set', 'str', default='default',
                        description='规则集名称'),
    ],
    returns=['rule_version', 'rule_source', 'is_demo', 'thresholds', 'notes'],
    timeout_sec=3,
))

# ---------- 10. 检测详情 ----------
register_tool(ToolSchema(
    name='detection_details',
    display_name='检测详情',
    description='输出详细的检测结果列表',
    category='analysis',
    params=[
        ToolParamSchema('scope', 'str', default='current',
                        choices=['current', 'full'],
                        description='范围'),
        ToolParamSchema('limit', 'int', default=20,
                        min_value=1, max_value=100, description='最大返回条数'),
    ],
    returns=['details', 'total_count', 'shown_count'],
    timeout_sec=5,
))

# ---------- 11. 保存记录 ----------
register_tool(ToolSchema(
    name='save_record',
    display_name='保存记录',
    description='将检测和判定结果保存到数据库',
    category='report',
    params=[
        ToolParamSchema('include_review', 'bool', default=False,
                        description='如果是REVIEW是否创建复核任务'),
    ],
    returns=['record_id', 'review_task_id', 'message'],
    timeout_sec=10,
))

# ---------- 12. 生成报告 ----------
register_tool(ToolSchema(
    name='generate_report',
    display_name='生成报告',
    description='生成结构化检测报告（可下载Word格式）',
    category='report',
    params=[
        ToolParamSchema('report_type', 'str', default='summary',
                        choices=['summary', 'detail', 'batch_summary'],
                        description='报告类型'),
        ToolParamSchema('format', 'str', default='json',
                        choices=['json', 'docx'],
                        description='报告格式'),
    ],
    returns=['report_data', 'download_path', 'message'],
    timeout_sec=15,
))

# ---------- 13. 复核任务创建 ----------
register_tool(ToolSchema(
    name='create_review',
    display_name='创建复核任务',
    description='为 REVIEW 级别的检测创建人工复核任务',
    category='review',
    params=[
        ToolParamSchema('reason', 'str', default='', description='复核原因'),
        ToolParamSchema('priority', 'str', default='normal',
                        choices=['low', 'normal', 'high', 'urgent'],
                        description='优先级'),
    ],
    returns=['review_id', 'status', 'message'],
    timeout_sec=5,
))

# ---------- 14. 切换回全量 ----------
register_tool(ToolSchema(
    name='switch_full_scope',
    display_name='切换全量范围',
    description='从子集分析切换回全量范围，避免子集结论误用为整体结论',
    category='analysis',
    params=[],
    returns=['scope', 'scope_label', 'detection_count'],
    timeout_sec=3,
))

# ---------- 15. 批次对比 ----------
register_tool(ToolSchema(
    name='compare_batches',
    display_name='批次对比',
    description='对比两个批次的检测统计结果（缺陷率、类别分布等）',
    category='analysis',
    params=[
        ToolParamSchema('batch_a_id', 'str', required=True, description='批次A ID'),
        ToolParamSchema('batch_b_id', 'str', required=True, description='批次B ID'),
    ],
    returns=['comparison', 'differences'],
    timeout_sec=10,
))

# ---------- 16. GAN 生成缺陷样本 ----------
register_tool(ToolSchema(
    name='gan_generate',
    display_name='GAN缺陷图像生成',
    description='使用条件 GAN 生成指定类别的缺陷样本图像，用于数据增强或可视化展示',
    category='augmentation',
    params=[
        ToolParamSchema('defect_class', 'str', required=True,
                        choices=['crazing', 'inclusion', 'patches', 'pitted_surface',
                                 'rolled-in_scale', 'scratches'],
                        description='缺陷类别名称'),
        ToolParamSchema('num_samples', 'int', default=4,
                        min_value=1, max_value=64,
                        description='生成样本数量'),
        ToolParamSchema('save_images', 'bool', default=True,
                        description='是否保存生成的图像到文件'),
    ],
    returns=['generated_count', 'defect_class', 'image_paths',
             'generate_time_ms', 'model_info'],
    timeout_sec=30,
))

# ---------- 17. GAN 模型信息 ----------
register_tool(ToolSchema(
    name='gan_model_info',
    display_name='GAN模型信息查询',
    description='查询当前加载的 GAN 模型信息（架构、类别、检查点等）',
    category='augmentation',
    params=[],
    returns=['loaded', 'architecture', 'image_size', 'num_classes',
             'class_names', 'device', 'total_generated'],
    timeout_sec=5,
))

# ---------- 18. GAN 生成网格 ----------
register_tool(ToolSchema(
    name='gan_generate_grid',
    display_name='GAN生成类别网格',
    description='生成所有缺陷类别的样本网格图，直观对比各类缺陷的生成效果',
    category='augmentation',
    params=[
        ToolParamSchema('samples_per_class', 'int', default=8,
                        min_value=1, max_value=16,
                        description='每个类别的样本数量'),
    ],
    returns=['grid_image_path', 'samples_per_class', 'num_classes',
             'total_samples', 'generate_time_ms'],
    timeout_sec=60,
))

# ---------- 19. GAN-YOLO 融合检测 ----------
register_tool(ToolSchema(
    name='fused_detect',
    display_name='融合检测',
    description='GAN重建+YOLO检测双路验证，降低漏检误检率，给出更可靠的检测结果和置信度',
    category='detection',
    params=[
        ToolParamSchema('image_path', 'str', required=True,
                        description='图像文件路径'),
        ToolParamSchema('anomaly_threshold', 'float', default=0.15,
                        min_value=0.01, max_value=0.5,
                        description='GAN异常阈值，重建误差超过则判定为异常'),
    ],
    returns=['quality', 'confidence', 'detections',
             'reconstruction_error', 'cross_validation_notes',
             'fusion_method'],
    timeout_sec=30,
))

# ---------- 20. 困难样本挖掘 ----------
register_tool(ToolSchema(
    name='mine_hard_samples',
    display_name='困难样本挖掘',
    description='用GAN生成多样性样本，经YOLO检测后筛选出低置信度/误检/漏检的困难样本，用于针对性模型提升',
    category='augmentation',
    params=[
        ToolParamSchema('num_per_class', 'int', default=20,
                        min_value=5, max_value=100,
                        description='每类生成数量'),
    ],
    returns=['hard_samples_count', 'hard_ratio', 'difficulty_distribution',
             'per_class_stats', 'total_generated'],
    timeout_sec=120,
))

# ---------- 21. 自适应增强 ----------
register_tool(ToolSchema(
    name='adaptive_augmentation',
    display_name='自适应数据增强',
    description='分析各类别检测性能，对薄弱类别用GAN定向生成更多样本，实现自适应数据增强',
    category='augmentation',
    params=[
        ToolParamSchema('base_samples', 'int', default=100,
                        min_value=10, max_value=500,
                        description='基础生成数量（表现最好的类别生成数量）'),
        ToolParamSchema('max_ratio', 'float', default=5.0,
                        min_value=1.5, max_value=10.0,
                        description='最大增强倍率（表现最差的类别生成 base×max_ratio）'),
    ],
    returns=['total_generated', 'weak_classes', 'augmentation_plan',
             'output_dir', 'method'],
    timeout_sec=300,
))

# ---------- 22. 文件夹批量检测 ----------
register_tool(ToolSchema(
    name='batch_detect_folder',
    display_name='文件夹批量检测',
    description='读取指定文件夹中的所有图片，依次执行缺陷检测和质量判定，返回批量统计结果',
    category='detection',
    params=[
        ToolParamSchema('folder_path', 'str', required=True,
                        description='包含图片的文件夹路径（本地绝对路径）'),
        ToolParamSchema('conf_threshold', 'float', default=0.25,
                        min_value=0.01, max_value=0.99, description='置信度阈值'),
    ],
    returns=['total_images', 'total_detections', 'by_class_total',
             'pass_count', 'review_count', 'fail_count', 'pass_rate',
             'results', 'summary'],
    timeout_sec=600,
))


# ---------- 批量分析工具 ----------

register_tool(ToolSchema(
    name='batch_quality_stats',
    display_name='批次质量统计',
    description='基于已有的批量检测结果，统计质量分布（合格/待复核/不合格的数量和比例）',
    category='analysis',
    params=[],
    returns=['total_images', 'pass_count', 'review_count', 'fail_count',
             'pass_rate', 'review_rate', 'fail_rate', 'summary'],
    timeout_sec=5,
))

register_tool(ToolSchema(
    name='batch_defect_distribution',
    display_name='批次缺陷分布',
    description='基于已有的批量检测结果，统计各类缺陷的数量和占比分布',
    category='analysis',
    params=[],
    returns=['total_detections', 'by_class', 'by_class_pct', 'top_class', 'summary'],
    timeout_sec=5,
))

register_tool(ToolSchema(
    name='batch_list_failed',
    display_name='列出不合格图片',
    description='基于已有的批量检测结果，列出不合格（FAIL）或待复核（REVIEW）的图片详情',
    category='analysis',
    params=[
        ToolParamSchema('quality_filter', 'str', default='FAIL',
                        choices=['FAIL', 'REVIEW', 'FAIL_OR_REVIEW'],
                        description='质量筛选条件'),
        ToolParamSchema('limit', 'int', default=20,
                        min_value=1, max_value=100, description='返回结果数量上限'),
    ],
    returns=['total_matched', 'images', 'summary'],
    timeout_sec=5,
))

register_tool(ToolSchema(
    name='batch_details',
    display_name='批次检测详情',
    description='基于已有的批量检测结果，返回每张图片的详细检测结果',
    category='analysis',
    params=[
        ToolParamSchema('limit', 'int', default=20,
                        min_value=1, max_value=100, description='返回结果数量上限'),
        ToolParamSchema('offset', 'int', default=0,
                        min_value=0, description='分页偏移量'),
    ],
    returns=['total_images', 'results', 'has_more', 'summary'],
    timeout_sec=5,
))


def get_tool_schema(tool_name: str) -> Optional[ToolSchema]:
    """获取工具 Schema"""
    return TOOL_SCHEMAS.get(tool_name)


def list_tool_schemas(category: str = None) -> List[ToolSchema]:
    """列出所有工具 Schema"""
    if category:
        return [s for s in TOOL_SCHEMAS.values() if s.category == category]
    return list(TOOL_SCHEMAS.values())
