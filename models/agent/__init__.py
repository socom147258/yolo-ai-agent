# -*- coding: utf-8 -*-
"""
AI Agent 智能质量检测系统
"""

# v2 核心模块
from .task_context import (
    TaskContext,
    DetectionItem,
    ImageInfo,
    SubsetView,
    SCOPE_FULL_IMAGE,
    SCOPE_SUBSET,
    SCOPE_BATCH,
    SCOPE_BATCH_SUBSET,
)

from .tool_schema import (
    ToolSchema,
    ToolParamSchema,
    ToolResult,
    ToolStatus,
    get_tool_schema,
    list_tool_schemas,
)

from .execution_trace import (
    ExecutionTrace,
    TraceStep,
    TraceStore,
)

from .quality_engine_v2 import (
    QualityEngineV2,
    QualityRuleSet,
    RuleStore,
    CLASS_SEVERITY,
    CLASS_NAMES_CN,
)

from .agent_executor_v2 import (
    AgentExecutorV2,
    AgentConfig,
)

from .review_workflow import (
    ReviewWorkflow,
    ReviewStatus,
    ReviewPriority,
)

# v1 兼容
from .detection_agent import AIAgent, QualityRuleEngine as QualityRuleEngineV1, ToolRegistry

DetectionAgent = AIAgent

__all__ = [
    'TaskContext', 'DetectionItem', 'ImageInfo', 'SubsetView',
    'SCOPE_FULL_IMAGE', 'SCOPE_SUBSET', 'SCOPE_BATCH', 'SCOPE_BATCH_SUBSET',
    'ToolSchema', 'ToolParamSchema', 'ToolResult', 'ToolStatus',
    'ExecutionTrace', 'TraceStep', 'TraceStore',
    'QualityEngineV2', 'QualityRuleSet', 'RuleStore',
    'AgentExecutorV2', 'AgentConfig',
    'ReviewWorkflow', 'ReviewStatus', 'ReviewPriority',
    'AIAgent', 'DetectionAgent', 'QualityRuleEngineV1', 'ToolRegistry',
]
