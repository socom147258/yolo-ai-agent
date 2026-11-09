# -*- coding: utf-8 -*-
"""
Pydantic 请求/响应模型定义

所有 API 接口的请求体和响应体都在此处定义，
确保类型安全和文档自动生成。
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================
# 通用响应格式
# ============================================================

class ApiResponse(BaseModel):
    """统一 API 响应格式"""
    code: int = Field(default=0, description="响应码，0 表示成功，非 0 表示错误")
    message: str = Field(default="success", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="响应数据")


# ============================================================
# 检测相关
# ============================================================

class DetectionItemSchema(BaseModel):
    """单个检测结果项"""
    cls_name: str = Field(description="缺陷类别名称")
    cls_name_cn: str = Field(default="", description="缺陷类别中文名称")
    confidence: float = Field(description="置信度 0-1")
    x1: float = Field(description="检测框左上角 x 坐标")
    y1: float = Field(description="检测框左上角 y 坐标")
    x2: float = Field(description="检测框右下角 x 坐标")
    y2: float = Field(description="检测框右下角 y 坐标")
    area: float = Field(default=0.0, description="检测框面积（像素平方）")


class ImageInfoSchema(BaseModel):
    """图像信息"""
    image_id: str = Field(description="图像ID")
    image_name: str = Field(description="图像文件名")
    width: int = Field(description="图像宽度（像素）")
    height: int = Field(description="图像高度（像素）")
    quality_score: Optional[float] = Field(default=None, description="图像质量评分 0-1")
    quality_issues: List[str] = Field(default_factory=list, description="图像质量问题列表")


class QualityResultSchema(BaseModel):
    """质量判定结果"""
    quality: str = Field(description="质量等级: PASS / FAIL / REVIEW")
    score: int = Field(description="质量评分 0-100")
    total_defects: int = Field(description="缺陷总数")
    critical_count: int = Field(description="严重缺陷数量")
    low_conf_count: int = Field(description="低置信度检测数量")
    total_box_area_pct: float = Field(description="总检测框面积占比（%）")
    max_single_box_area_pct: float = Field(description="最大单检测框面积占比（%）")
    max_severity: int = Field(description="最高严重等级")
    by_class: Dict[str, Any] = Field(default_factory=dict, description="按类别统计")
    reasons_fail: List[str] = Field(default_factory=list, description="不合格原因列表")
    reasons_review: List[str] = Field(default_factory=list, description="复核原因列表")
    explanation: str = Field(default="", description="自然语言解释")
    scope: str = Field(default="", description="判定范围标识")
    scope_label: str = Field(default="", description="判定范围标签（人类可读）")
    method_notes: List[str] = Field(default_factory=list, description="方法说明备注")
    is_forced: bool = Field(default=False, description="是否为强制判定")
    forced_reason: str = Field(default="", description="强制判定原因")
    rule_id: str = Field(default="", description="规则ID")
    rule_version: str = Field(default="", description="规则版本")
    rule_source: str = Field(default="", description="规则来源")
    is_demo_rule: bool = Field(default=True, description="是否为演示规则")


class DetectResponseData(BaseModel):
    """单图检测响应数据"""
    image_info: ImageInfoSchema
    detections: List[DetectionItemSchema] = Field(default_factory=list)
    detection_count: int = Field(default=0)
    quality: QualityResultSchema
    detection_time_ms: float = Field(default=0.0)
    quality_time_ms: float = Field(default=0.0)
    review_task_id: str = Field(default="", description="复核任务ID（如果创建了的话）")


# ============================================================
# 质量判定请求
# ============================================================

class QualityJudgeRequest(BaseModel):
    """质量判定请求（给定检测结果进行判定）"""
    detections: List[DetectionItemSchema] = Field(description="检测结果列表")
    image_width: int = Field(description="图像宽度")
    image_height: int = Field(description="图像高度")
    image_quality_score: Optional[float] = Field(default=None, description="图像质量评分")
    scope: str = Field(default="full_image", description="判定范围")
    scope_label: str = Field(default="单图全量", description="范围标签")


# ============================================================
# 批次检测
# ============================================================

class BatchImageResultSchema(BaseModel):
    """单张图像的批次处理结果"""
    image_name: str = Field(description="图像文件名")
    status: str = Field(description="处理状态: success / failed / skipped")
    detection_count: int = Field(default=0, description="检测到的缺陷数量")
    quality: str = Field(default="", description="质量等级")
    quality_score: float = Field(default=0, description="质量评分")
    scope: str = Field(default="", description="判定范围")
    error: str = Field(default="", description="错误信息")
    is_forced: bool = Field(default=False, description="是否强制判定")
    forced_reason: str = Field(default="", description="强制判定原因")
    review_task_id: str = Field(default="", description="复核任务ID")
    method_notes: List[str] = Field(default_factory=list, description="方法备注")
    detection_time_ms: float = Field(default=0.0, description="检测耗时(ms)")
    quality_time_ms: float = Field(default=0.0, description="质量判定耗时(ms)")


class BatchSummarySchema(BaseModel):
    """批次摘要"""
    planned: int = Field(default=0, description="计划检测数")
    success: int = Field(default=0, description="成功数")
    failed: int = Field(default=0, description="失败数")
    skipped: int = Field(default=0, description="跳过数")
    pass_count: int = Field(default=0, description="PASS 数量")
    fail_count: int = Field(default=0, description="FAIL 数量")
    review_count: int = Field(default=0, description="REVIEW 数量")


class BatchDetectResponseData(BaseModel):
    """批次检测响应数据"""
    batch_id: str = Field(description="批次ID")
    summary: BatchSummarySchema
    batch_quality: str = Field(description="批次整体质量等级")
    batch_quality_reason: str = Field(default="", description="批次质量判定原因")
    model_version: str = Field(default="", description="模型版本")
    rule_version: str = Field(default="", description="规则版本")
    rule_is_demo: bool = Field(default=True, description="是否演示规则")
    start_time: str = Field(default="", description="开始时间")
    end_time: str = Field(default="", description="结束时间")
    images: List[BatchImageResultSchema] = Field(default_factory=list, description="各图像结果")


# ============================================================
# 复核任务
# ============================================================

class ReviewTaskSchema(BaseModel):
    """复核任务"""
    id: Optional[int] = Field(default=None, description="数据库自增ID")
    review_id: str = Field(description="复核任务ID")
    task_id: str = Field(default="", description="关联任务ID")
    record_id: Optional[int] = Field(default=None, description="关联记录ID")
    image_name: str = Field(default="", description="图像文件名")
    image_path: str = Field(default="", description="图像路径")
    original_quality: str = Field(default="", description="原始质量判定")
    original_score: int = Field(default=0, description="原始评分")
    original_scope: str = Field(default="", description="原始判定范围")
    quality_issue: str = Field(default="", description="质量问题描述")
    reason: str = Field(default="", description="创建原因")
    priority: str = Field(default="normal", description="优先级: low/normal/high/urgent")
    status: str = Field(default="pending", description="状态: pending/in_progress/confirmed/rejected/corrected/escalated")
    created_at: str = Field(default="", description="创建时间")
    assigned_to: str = Field(default="", description="分配给")
    resolved_at: str = Field(default="", description="解决时间")
    resolved_by: str = Field(default="", description="解决人")
    resolution: str = Field(default="", description="解决方案")
    resolution_note: str = Field(default="", description="解决备注")
    model_version: str = Field(default="", description="模型版本")
    rule_version: str = Field(default="", description="规则版本")


class ReviewTaskListData(BaseModel):
    """复核任务列表响应数据"""
    tasks: List[ReviewTaskSchema] = Field(default_factory=list, description="复核任务列表")
    total: int = Field(default=0, description="总数")
    limit: int = Field(default=50, description="每页数量")


class ReviewConfirmRequest(BaseModel):
    """确认复核请求"""
    actor: str = Field(default="operator", description="操作人")
    role: str = Field(default="operator", description="角色")
    note: str = Field(default="", description="备注")


class ReviewRejectRequest(BaseModel):
    """驳回复核请求"""
    actor: str = Field(default="inspector", description="操作人")
    role: str = Field(default="inspector", description="角色")
    reason: str = Field(default="", description="驳回原因")
    corrected_quality: str = Field(default="PASS", description="修正后的质量等级")


# ============================================================
# Agent 对话
# ============================================================

class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    message: str = Field(description="用户自然语言消息")
    task_id: Optional[str] = Field(default=None, description="任务ID（可选，用于多轮对话）")


class AgentChatResponseData(BaseModel):
    """Agent 对话响应数据"""
    task_id: str = Field(description="任务ID")
    reply: str = Field(description="Agent 回复")
    final_quality: str = Field(default="", description="最终质量结论")
    final_scope: str = Field(default="", description="结论范围")
    total_steps: int = Field(default=0, description="执行步骤数")
    total_time_ms: float = Field(default=0.0, description="总耗时(ms)")
    llm_fallback: bool = Field(default=False, description="是否降级到规则模式")
    llm_fallback_reason: str = Field(default="", description="降级原因")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="执行步骤详情")


# ============================================================
# 系统配置
# ============================================================

class SystemConfigData(BaseModel):
    """系统配置响应数据"""
    model_version: str = Field(description="模型版本")
    model_path: str = Field(description="模型路径")
    rule_id: str = Field(description="规则ID")
    rule_version: str = Field(description="规则版本")
    rule_source: str = Field(description="规则来源")
    is_demo_rule: bool = Field(description="是否为演示规则")
    llm_provider: str = Field(description="LLM 提供商")
    llm_model: str = Field(default="", description="LLM 模型名")
    llm_available: bool = Field(description="LLM 是否可用")
    data_path: str = Field(description="数据路径")
    output_dir: str = Field(description="输出目录")
    review_db_path: str = Field(description="复核数据库路径")
    trace_db_path: str = Field(description="轨迹数据库路径")


# ============================================================
# 报告
# ============================================================

class ReportGenerateData(BaseModel):
    """报告生成响应数据"""
    report_id: str = Field(description="报告ID")
    report_path: str = Field(description="报告文件路径")
    report_type: str = Field(description="报告类型: single / batch")
    file_size: int = Field(default=0, description="文件大小（字节）")
    generated_at: str = Field(default="", description="生成时间")


# ============================================================
# 健康检查
# ============================================================

class HealthData(BaseModel):
    """健康检查响应数据"""
    status: str = Field(default="ok", description="服务状态")
    version: str = Field(default="1.0.0", description="API 版本")
    model_loaded: bool = Field(default=False, description="模型是否已加载")
    llm_available: bool = Field(default=False, description="LLM 是否可用")
