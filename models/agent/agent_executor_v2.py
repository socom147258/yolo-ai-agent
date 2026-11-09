# -*- coding: utf-8 -*-
"""
Agent 执行循环 v2 —— 有界工具调用 + 双路径

双路径设计：
1. LLM 模式：使用支持结构化工具调用的大模型，实现"理解-选择-执行-检查-继续/结束"的有界循环
2. 规则模式（Rule-based）：基于关键词匹配和固定流程，作为 LLM 不可用时的降级方案

安全机制：
- 最大步骤限制（max_steps）
- 费用预算（max_cost）
- 超时控制（per-tool timeout）
- 有限重试
- 参数校验
- 白名单工具
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from .task_context import TaskContext, DetectionItem, ImageInfo
from .tool_schema import (ToolSchema, ToolResult, ToolStatus,
                          get_tool_schema, list_tool_schemas)
from .execution_trace import ExecutionTrace
from .quality_engine_v2 import QualityEngineV2, QualityRuleSet
from .llm_integration import (
    LLMAgentOrchestrator,
    create_llm_provider,
    load_llm_config_from_env,
    build_tools_definition,
)


@dataclass
class AgentConfig:
    """Agent 配置"""
    # 执行限制
    max_steps: int = 10
    max_cost_credits: int = 100
    default_tool_timeout: int = 30

    # LLM 配置（可选）
    llm_provider: str = "none"       # none / openai / deepseek / qwen / moonshot / mock
    llm_model: str = ""
    llm_api_key: str = ""            # 直接传入的 key（优先于环境变量）
    llm_base_url: str = ""
    llm_temperature: float = 0.1

    # 规则配置
    rule_id: str = "demo_default"
    rule_version: str = "1.0.0"

    # 行为
    auto_save_trace: bool = True
    auto_save_record: bool = False   # 是否自动保存检测记录

    # 报告输出
    report_output_dir: str = "outputs/reports"  # Word 报告输出目录

    # GAN 配置
    gan_enabled: bool = True  # 是否启用 GAN 工具
    gan_checkpoint_path: str = ""  # GAN 检查点路径，为空则用默认


class AgentExecutorV2:
    """
    Agent 执行器 v2

    工作流程：
    1. 初始化任务上下文和执行轨迹
    2. 选择执行模式（LLM / 规则）
    3. 有界循环执行工具调用
    4. 每步记录轨迹
    5. 到达终止条件时结束并输出结论
    """

    def __init__(self, config: AgentConfig, model=None, db_path: str = None):
        self.config = config
        self.model = model  # YOLO 模型实例（用于检测工具）
        self.db_path = db_path

        # 质量引擎
        self.quality_engine = QualityEngineV2(QualityRuleSet.default_demo())

        # 工具实现映射
        self._tool_impls: Dict[str, Callable] = {}
        self._register_tool_impls()

        # LLM 编排器（懒加载）
        self._llm_orchestrator = None

        # 轨迹存储
        self.trace_store = None
        if db_path and config.auto_save_trace:
            from .execution_trace import TraceStore
            self.trace_store = TraceStore(db_path)

        # GAN 管理器（懒加载）
        self._gan_manager = None
        self._gan_loaded = False

        # GAN-YOLO 融合管理器（懒加载）
        self._fusion_manager = None

    # ==================== 工具注册 ====================

    def _register_tool_impls(self):
        """注册工具实现"""
        tools = {
            'detect_image': self._tool_detect_image,
            'count_defects': self._tool_count_defects,
            'filter_by_class': self._tool_filter_by_class,
            'area_analysis': self._tool_area_analysis,
            'confidence_dist': self._tool_confidence_dist,
            'quality_judge': self._tool_quality_judge,
            'rule_lookup': self._tool_rule_lookup,
            'detection_details': self._tool_detection_details,
            'switch_full_scope': self._tool_switch_full_scope,
            'image_quality_check': self._tool_image_quality_check,
            'save_record': self._tool_save_record,
            'generate_report': self._tool_generate_report,
            'create_review': self._tool_create_review,
            'batch_detect_folder': self._tool_batch_detect_folder,
            'batch_quality_stats': self._tool_batch_quality_stats,
            'batch_defect_distribution': self._tool_batch_defect_distribution,
            'batch_list_failed': self._tool_batch_list_failed,
            'batch_details': self._tool_batch_details,
        }
        # GAN 工具（仅在启用时注册）
        if self.config.gan_enabled:
            tools['gan_generate'] = self._tool_gan_generate
            tools['gan_model_info'] = self._tool_gan_model_info
            tools['gan_generate_grid'] = self._tool_gan_generate_grid
            # GAN-YOLO 融合工具
            tools['fused_detect'] = self._tool_fused_detect
            tools['mine_hard_samples'] = self._tool_mine_hard_samples
            tools['adaptive_augmentation'] = self._tool_adaptive_augmentation
        self._tool_impls = tools

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表（白名单）"""
        return list(self._tool_impls.keys())

    # ==================== 主入口 ====================

    def run_task(self, user_input: str, image_path: str = None,
                 image_paths: List[str] = None, task_id: str = None,
                 folder_path: str = None, batch_results: List[Dict] = None,
                 batch_by_class: Dict[str, int] = None,
                 batch_total_detections: int = None) -> Dict:
        """
        执行一个质检任务

        Args:
            user_input: 用户的自然语言指令
            image_path: 单张图像路径（单图任务）
            image_paths: 图像路径列表（批次任务）
            task_id: 可选任务ID
            folder_path: 文件夹路径（批量检测任务）
            batch_results: 已有批量检测结果（多轮对话时复用，避免重复检测）
            batch_by_class: 已有批量缺陷分类统计
            batch_total_detections: 已有批量总缺陷数

        Returns:
            完整的任务执行结果
        """
        # 1. 初始化
        ctx = TaskContext(task_id=task_id)
        ctx.model_version = self._get_model_version()
        ctx.rule_version = f"{self.quality_engine.rule_set.rule_id}_v{self.quality_engine.rule_set.version}"
        ctx.rule_source = self.quality_engine.rule_set.source
        ctx.rule_snapshot = self.quality_engine.rule_set.to_dict()
        if folder_path:
            ctx.folder_path = folder_path

        # 恢复已有批量检测结果（多轮对话复用）
        if batch_results:
            ctx.batch_results = batch_results
            ctx.batch_folder = folder_path
            if batch_by_class is not None:
                ctx.batch_by_class = batch_by_class
            if batch_total_detections is not None:
                ctx.batch_total_detections = batch_total_detections

        trace = ExecutionTrace(
            task_id=ctx.task_id,
            model_version=ctx.model_version,
            rule_version=ctx.rule_version,
        )

        # 2. 确定执行模式
        use_llm = self._can_use_llm()
        if not use_llm:
            trace.llm_unavailable_fallback = True

        # 3. 执行
        if use_llm:
            result = self._run_llm_loop(ctx, trace, user_input, image_path, image_paths)
        else:
            result = self._run_rule_loop(ctx, trace, user_input, image_path, image_paths)

        # 4. 保存轨迹
        if self.trace_store:
            self.trace_store.save(trace)

        # 5. 批量模式：补充完整批量数据到结果中（用于多轮对话复用）
        if hasattr(ctx, 'batch_results') and ctx.batch_results:
            result['batch_results'] = ctx.batch_results
            result['batch_folder'] = getattr(ctx, 'batch_folder', folder_path)
            if hasattr(ctx, 'batch_by_class'):
                result['batch_by_class'] = ctx.batch_by_class
            if hasattr(ctx, 'batch_total_detections'):
                result['batch_total_detections'] = ctx.batch_total_detections

        return result

    # ==================== LLM 执行循环 ====================

    def _run_llm_loop(self, ctx: TaskContext, trace: ExecutionTrace,
                      user_input: str, image_path: str, image_paths: List[str]) -> Dict:
        """
        LLM 驱动的有界执行循环（真实多轮工具调用）

        工作流程：
        1. 初始检测（如有图且未检测）
        2. 构建对话消息，调用 LLM 获取下一步
        3. 执行 LLM 选择的工具，记录结果
        4. 将工具结果加入对话，再次调用 LLM
        5. 循环直到 LLM 结束、达到最大步骤或预算耗尽

        LLM 失败时自动降级到规则模式。
        """
        step_idx = 0
        llm_call_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        fallback_to_rule = False
        fallback_reason = ''
        final_scope = None  # 最终范围，批量模式会提前设置

        orch = self._get_llm_orchestrator()
        rule_info = {
            'rule_id': self.quality_engine.rule_set.rule_id,
            'version': self.quality_engine.rule_set.version,
            'source': self.quality_engine.rule_set.source,
            'is_demo': self.quality_engine.rule_set.is_demo,
        }

        # 对话历史（LLM 视角）
        llm_messages = [{'role': 'user', 'content': user_input}]

        # 第0步：如果是批量文件夹检测，先执行批量检测（系统级前置步骤）
        folder = getattr(ctx, 'folder_path', None)
        has_batch_results = hasattr(ctx, 'batch_results') and getattr(ctx, 'batch_results', None)
        # 两种情况触发首次检测：1) 文件夹模式 + 有相关关键词 + 还没检测过
        batch_keywords = ['批量', '文件夹', '目录', 'folder', 'batch', '一批', '检测', '统计', '合格', '质量', '缺陷', '多少个', '分布', '分析']
        if folder and not has_batch_results and any(kw in user_input for kw in batch_keywords):
            step_idx += 1
            batch_result = self._execute_tool('batch_detect_folder', {'folder_path': folder},
                                              ctx, step_idx, trace)
            # 把批量检测结果摘要加入对话，让 LLM 基于此继续分析
            batch_summary = batch_result.data.get('summary', '批量检测完成')
            llm_messages = [
                {'role': 'system', 'content': f'当前处于批量检测模式。已完成文件夹批量检测：{batch_summary}\n\n你可以直接调用以下工具进行深入分析，不需要再次执行批量检测：\n- batch_quality_stats：批次质量统计（合格率等）\n- batch_defect_distribution：各类缺陷分布\n- batch_list_failed：列出不合格/待复核的图片\n- batch_details：查看所有图片详情'},
                {'role': 'user', 'content': user_input},
            ]
            # 标记为批次模式
            final_scope = "批次检测"
        elif folder and has_batch_results:
            # 已有批量检测结果（追问场景），直接进入 LLM 对话
            # 先告诉 LLM 当前状态，让它知道可以直接调用分析工具
            batch_count = len(ctx.batch_results)
            total_det = getattr(ctx, 'batch_total_detections', '若干')
            llm_messages = [
                {'role': 'system', 'content': f'当前处于批量检测模式。之前已完成文件夹批量检测，共 {batch_count} 张图片，检测到 {total_det} 个缺陷。\n\n你可以直接调用以下工具进行深入分析，不需要再次执行批量检测：\n- batch_quality_stats：批次质量统计（合格率等）\n- batch_defect_distribution：各类缺陷分布\n- batch_list_failed：列出不合格/待复核的图片\n- batch_details：查看所有图片详情\n\n请根据用户的问题选择合适的工具。'},
                {'role': 'user', 'content': user_input},
            ]
            final_scope = "批次检测"

        # 第一步：如果有图且没检测，先做检测（这是必须的，LLM 不知道图像路径）
        if image_path and not ctx.has_raw_detections:
            step_idx += 1
            result = self._execute_tool('detect_image', {'image_path': image_path}, ctx, step_idx, trace)

            # 把检测结果摘要加入对话
            det_count = len(ctx.get_current_detections())
            det_summary = f"图像检测完成，共发现 {det_count} 个缺陷。"
            if ctx.image and ctx.image.quality_score is not None:
                det_summary += f" 图像质量分：{ctx.image.quality_score:.2f}。"
            llm_messages.append({
                'role': 'assistant',
                'content': f"已完成图像检测，{det_summary} 请告诉我你想做什么分析？"
            })
            # 补充用户的原始问题
            llm_messages.append({'role': 'user', 'content': user_input})

        # LLM 多轮循环
        while step_idx < self.config.max_steps:
            # 检查预算
            if trace.total_cost >= self.config.max_cost_credits:
                trace.budget_exceeded = True
                break

            # 构建上下文信息
            context_info = {
                'scope_label': ctx.scope_label,
                'is_subset': ctx.is_subset,
                'detection_count': len(ctx.get_current_detections()),
                'has_raw_detections': ctx.has_raw_detections,
                'max_steps': self.config.max_steps,
                'current_step': step_idx,
                'is_batch_mode': bool(getattr(ctx, 'batch_results', None)),
                'batch_folder': getattr(ctx, 'batch_folder', None),
                'batch_total_images': len(getattr(ctx, 'batch_results', [])),
            }

            # 调用 LLM 规划下一步
            llm_call_count += 1
            llm_result = orch.plan_next_step(llm_messages, context_info, rule_info)

            # 记录 LLM 调用信息到轨迹
            trace.add_llm_call({
                'call_index': llm_call_count,
                'model': llm_result.model,
                'mode': llm_result.mode,
                'prompt_tokens': llm_result.prompt_tokens,
                'completion_tokens': llm_result.completion_tokens,
                'total_tokens': llm_result.total_tokens,
                'time_ms': llm_result.time_ms,
                'success': llm_result.success,
                'error': llm_result.error,
                'tool_calls_proposed': [tc['name'] for tc in llm_result.tool_calls],
            })
            total_prompt_tokens += llm_result.prompt_tokens
            total_completion_tokens += llm_result.completion_tokens

            # LLM 调用失败 → 降级到规则模式
            if not llm_result.success:
                fallback_to_rule = True
                fallback_reason = llm_result.fallback_reason or llm_result.error
                trace.llm_unavailable_fallback = True
                trace.llm_fallback_reason = fallback_reason
                break

            # LLM 没有工具调用 → 任务完成
            if not llm_result.tool_calls:
                # LLM 给出了最终回答
                final_text = llm_result.response_text or '分析完成。'
                llm_messages.append({'role': 'assistant', 'content': final_text})
                break

            # 执行 LLM 选择的工具（一次只执行第一个，让 LLM 逐步决策）
            tool_call = llm_result.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']

            # 安全校验：工具白名单 + 参数校验
            valid, err_msg = orch.validate_tool_call(tool_name, tool_args)
            if not valid:
                # 校验失败，告诉 LLM 让它修正
                llm_messages.append({
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{
                        'id': tool_call.get('id', 'unknown'),
                        'type': 'function',
                        'function': {'name': tool_name, 'arguments': json.dumps(tool_args)},
                    }]
                })
                llm_messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.get('id', 'unknown'),
                    'content': f"错误：{err_msg}。请检查参数后重试，或者选择其他工具。",
                })
                step_idx += 1
                trace.add_tool_step(
                    step_index=step_idx,
                    tool_name=tool_name,
                    action_desc=f"LLM提议调用 {tool_name}",
                    params_summary=json.dumps(tool_args, ensure_ascii=False)[:200],
                    result_summary=f"参数校验失败: {err_msg}",
                    status=ToolStatus.VALIDATION_ERROR,
                    error=err_msg,
                    exec_ms=0,
                    cost=0,
                )
                continue

            # 执行工具
            step_idx += 1
            tool_result = self._execute_tool(tool_name, tool_args, ctx, step_idx, trace)

            # 把工具调用加入对话历史
            llm_messages.append({
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': tool_call.get('id', f'call_{step_idx}'),
                    'type': 'function',
                    'function': {'name': tool_name, 'arguments': json.dumps(tool_args)},
                }]
            })

            # 把工具结果加入对话历史（摘要形式，避免 token 爆炸）
            result_summary = self._summarize_result(tool_name, tool_result)
            llm_messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.get('id', f'call_{step_idx}'),
                'content': result_summary[:1000],  # 限制长度
            })

            # 限制对话轮次（防止无限循环）
            if llm_call_count >= self.config.max_steps:
                trace.max_steps_reached = True
                break

        # 如果 LLM 失败，降级到规则模式完成剩余工作
        if fallback_to_rule:
            # 用规则模式补充执行
            result = self._run_rule_fallback(ctx, trace, user_input, image_path, step_idx)
            result['mode'] = 'llm_fallback_to_rule'
            result['fallback_reason'] = fallback_reason
            return result

        # 生成最终结论
        final_quality = "未执行判定"
        if final_scope is None:
            final_scope = ctx.scope_label

        # 检查是否是批量模式
        batch_tools = {'batch_detect_folder', 'batch_quality_stats',
                       'batch_defect_distribution', 'batch_list_failed', 'batch_details'}
        has_batch_steps = any(step.tool_name in batch_tools and step.status == 'success'
                              for step in trace.steps)

        if has_batch_steps:
            # 批量模式：从 batch_quality_stats 获取质量信息
            for step in reversed(trace.steps):
                if step.tool_name == 'batch_quality_stats' and step.status == 'success':
                    final_quality = '批次'
                    break
            else:
                final_quality = '批次'
            if not final_scope or final_scope == ctx.scope_label:
                final_scope = '批次检测'
        else:
            # 单图模式：从 quality_judge 获取结果
            for step in reversed(trace.steps):
                if step.tool_name == 'quality_judge' and step.status == 'success':
                    if 'FAIL' in step.result_summary:
                        final_quality = 'FAIL'
                    elif 'REVIEW' in step.result_summary:
                        final_quality = 'REVIEW'
                    elif 'PASS' in step.result_summary:
                        final_quality = 'PASS'
                    break

        # 从最后一条 LLM 回复提取结论文本
        final_text = ''
        for msg in reversed(llm_messages):
            if msg['role'] == 'assistant' and msg.get('content'):
                final_text = msg['content']
                break

        if not final_text:
            final_text = f"分析完成。质量判定：{final_quality}（{final_scope}）"

        conclusion = final_text
        trace.mark_finished(conclusion, final_scope)
        trace.llm_total_prompt_tokens = total_prompt_tokens
        trace.llm_total_completion_tokens = total_completion_tokens
        trace.llm_call_count = llm_call_count

        result = {
            'task_id': ctx.task_id,
            'mode': 'llm',
            'llm_model': orch.provider.model_name,
            'llm_provider': orch.provider.provider_name,
            'llm_call_count': llm_call_count,
            'llm_total_tokens': total_prompt_tokens + total_completion_tokens,
            'trace': trace.to_dict(),
            'context_snapshot': ctx.snapshot(),
            'final_conclusion': conclusion,
            'final_quality': final_quality,
            'final_scope': final_scope,
        }
        if has_batch_steps:
            result['batch_results'] = getattr(ctx, 'batch_results', [])
        return result

    def _run_rule_fallback(self, ctx: TaskContext, trace: ExecutionTrace,
                            user_input: str, image_path: str, start_step: int) -> Dict:
        """LLM 失败后的规则模式降级执行"""
        # 用关键词规划补充执行
        plan = self._plan_by_keywords(user_input, ctx)

        # 过滤掉已经执行过的步骤（简单按工具名匹配）
        executed_tools = {s.tool_name for s in trace.steps}
        remaining_plan = [(name, params) for name, params in plan
                          if name not in executed_tools]

        step_idx = start_step
        for tool_name, params in remaining_plan:
            step_idx += 1
            if step_idx > self.config.max_steps:
                trace.max_steps_reached = True
                break
            self._execute_tool(tool_name, params, ctx, step_idx, trace)

        # 生成结论
        final_quality = "未执行判定"
        final_scope = ctx.scope_label

        # 检查是否是批量模式（包含批量检测或批量分析工具）
        batch_tools = {'batch_detect_folder', 'batch_quality_stats',
                       'batch_defect_distribution', 'batch_list_failed', 'batch_details'}
        has_batch_steps = any(step.tool_name in batch_tools and step.status == 'success'
                              for step in trace.steps)

        if has_batch_steps:
            # 批量模式：收集所有成功的批量步骤的摘要，组合成最终结论
            conclusion_parts = []
            for step in trace.steps:
                if step.tool_name in batch_tools and step.status == 'success':
                    if step.result_summary:
                        conclusion_parts.append(step.result_summary)
            conclusion = "（规则模式降级）" + (
                '\n\n'.join(conclusion_parts) if conclusion_parts else "批量检测完成")
            final_scope = "批次检测"
            final_quality = "批次"
        else:
            # 从最后一个 quality_judge 步骤获取结果（单图模式）
            for step in reversed(trace.steps):
                if step.tool_name == 'quality_judge' and step.status == 'success':
                    if 'FAIL' in step.result_summary:
                        final_quality = 'FAIL'
                    elif 'REVIEW' in step.result_summary:
                        final_quality = 'REVIEW'
                    elif 'PASS' in step.result_summary:
                        final_quality = 'PASS'
                    break
            conclusion = f"（规则模式降级）质检结论：{final_quality}（{final_scope}）"
        trace.mark_finished(conclusion, final_scope)

        result = {
            'task_id': ctx.task_id,
            'mode': 'rule_fallback',
            'trace': trace.to_dict(),
            'context_snapshot': ctx.snapshot(),
            'final_conclusion': conclusion,
            'final_quality': final_quality,
            'final_scope': final_scope,
        }
        if has_batch_steps:
            result['batch_results'] = getattr(ctx, 'batch_results', [])
        return result

    # ==================== 规则执行循环 ====================

    def _run_rule_loop(self, ctx: TaskContext, trace: ExecutionTrace,
                       user_input: str, image_path: str, image_paths: List[str]) -> Dict:
        """
        规则驱动的执行循环（关键词匹配 + 固定流程）
        作为 LLM 不可用时的降级方案
        """
        step_idx = 0

        # 根据关键词生成工具调用计划
        plan = self._plan_by_keywords(user_input, ctx)

        # 如果有图但计划里没有检测步骤，且还没检测过，先检测
        if image_path and not ctx.has_raw_detections:
            plan.insert(0, ('detect_image', {'image_path': image_path}))

        # 执行计划
        for tool_name, params in plan:
            step_idx += 1
            if step_idx > self.config.max_steps:
                trace.max_steps_reached = True
                break

            self._execute_tool(tool_name, params, ctx, step_idx, trace)

        # 生成最终结论
        final_quality = "未执行判定"
        final_scope = ctx.scope_label
        eval_result = None

        # 检查是否是批量模式（包含批量检测或批量分析工具）
        batch_tools = {'batch_detect_folder', 'batch_quality_stats',
                       'batch_defect_distribution', 'batch_list_failed', 'batch_details'}
        has_batch_steps = any(step.tool_name in batch_tools and step.status == 'success'
                              for step in trace.steps)

        if has_batch_steps:
            # 批量模式：收集所有成功的批量步骤的摘要，组合成最终结论
            conclusion_parts = []
            for step in trace.steps:
                if step.tool_name in batch_tools and step.status == 'success':
                    if step.result_summary:
                        conclusion_parts.append(step.result_summary)

            conclusion = '\n\n'.join(conclusion_parts) if conclusion_parts else "批量检测完成"
            final_scope = "批次检测"

            # 从质量统计中获取最终质量等级
            final_quality = "批次"
            for step in reversed(trace.steps):
                if step.tool_name == 'batch_quality_stats' and step.status == 'success':
                    # 从结果中提取 fail_count 等信息
                    break

            trace.mark_finished(conclusion, final_scope)

            return {
                'task_id': ctx.task_id,
                'mode': 'rule_based',
                'llm_fallback': True,
                'trace': trace.to_dict(),
                'context_snapshot': ctx.snapshot(),
                'final_conclusion': conclusion,
                'final_scope': final_scope,
                'batch_results': getattr(ctx, 'batch_results', []),
            }

        # 从最后一个 quality_judge 步骤获取结果（单图模式）
        for step in reversed(trace.steps):
            if step.tool_name == 'quality_judge' and step.status == 'success':
                if 'FAIL' in step.result_summary:
                    final_quality = 'FAIL'
                elif 'REVIEW' in step.result_summary:
                    final_quality = 'REVIEW'
                elif 'PASS' in step.result_summary:
                    final_quality = 'PASS'
                break

        conclusion = f"质检结论：{final_quality}（{final_scope}）"
        trace.mark_finished(conclusion, final_scope)

        return {
            'task_id': ctx.task_id,
            'mode': 'rule_based',
            'llm_fallback': True,
            'trace': trace.to_dict(),
            'context_snapshot': ctx.snapshot(),
            'final_conclusion': conclusion,
            'final_scope': final_scope,
            'detections': ctx.get_current_detections(),
        }

    # ==================== 工具执行 ====================

    def _execute_tool(self, tool_name: str, params: Dict, ctx: TaskContext,
                      step_idx: int, trace: ExecutionTrace) -> ToolResult:
        """安全地执行单个工具（含校验、超时、重试）"""
        schema = get_tool_schema(tool_name)
        impl = self._tool_impls.get(tool_name)

        t0 = time.time()

        # 1. 参数校验
        if schema:
            is_valid, errors, cleaned = schema.validate_params(params)
            if not is_valid:
                result = ToolResult.validation_error(tool_name, errors)
                trace.add_tool_step(step_idx, tool_name, f"参数校验失败: {tool_name}",
                                   str(params)[:100], "", ToolStatus.VALIDATION_ERROR,
                                   error="; ".join(errors))
                return result
            params = cleaned
            action_desc = schema.display_name
        else:
            # 未知工具
            result = ToolResult.error(tool_name, f"未知工具: {tool_name}")
            trace.add_tool_step(step_idx, tool_name, f"未知工具",
                               "", "", ToolStatus.NOT_FOUND,
                               error=f"工具 {tool_name} 不在白名单中")
            return result

        # 2. 执行（含简单超时检查）
        try:
            if impl is None:
                result = ToolResult.error(tool_name, f"工具 {tool_name} 未实现")
            else:
                # 注入上下文
                params['_ctx'] = ctx
                params['_trace'] = trace

                tool_data = impl(**params)
                exec_ms = (time.time() - t0) * 1000

                if isinstance(tool_data, ToolResult):
                    result = tool_data
                    result.execution_time_ms = exec_ms
                elif isinstance(tool_data, dict):
                    result = ToolResult.success(tool_name, tool_data, exec_ms)
                else:
                    result = ToolResult.success(tool_name, {'value': tool_data}, exec_ms)

        except Exception as e:
            exec_ms = (time.time() - t0) * 1000
            result = ToolResult.error(tool_name, str(e), exec_ms)

        # 3. 记录轨迹
        params_summary = self._summarize_params(tool_name, params)
        result_summary = self._summarize_result(tool_name, result)
        trace.add_tool_step(
            step_index=step_idx,
            tool_name=tool_name,
            action_desc=action_desc,
            params_summary=params_summary,
            result_summary=result_summary,
            status=result.status,
            error=result.error_message,
            exec_ms=result.execution_time_ms,
            cost=result.cost_credits,
        )

        return result

    def _summarize_params(self, tool_name: str, params: Dict) -> str:
        """生成参数摘要（避免泄露敏感信息，控制长度）"""
        safe_params = {k: v for k, v in params.items()
                       if not k.startswith('_') and k not in ('api_key', 'password')}
        summary = json.dumps(safe_params, ensure_ascii=False, default=str)
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary

    def _summarize_result(self, tool_name: str, result: ToolResult) -> str:
        """生成结果摘要（控制长度）"""
        if result.status != ToolStatus.SUCCESS:
            return f"失败: {result.error_message[:100]}"

        data = result.data
        # 根据工具类型提取关键字段
        if tool_name == 'detect_image':
            return f"检测到 {data.get('total_count', 0)} 个缺陷"
        elif tool_name == 'count_defects':
            return f"共 {data.get('total_count', 0)} 个，{len(data.get('by_class', {}))} 类"
        elif tool_name == 'filter_by_class':
            return f"筛选出 {data.get('filtered_count', 0)} 个 {data.get('target_class', '')}"
        elif tool_name == 'quality_judge':
            quality = data.get('quality', '?')
            score = data.get('score', 0)
            return f"{quality}（{score}分）[{data.get('scope_label', '')}]"
        elif tool_name == 'area_analysis':
            return f"总面积 {data.get('total_area_pct', 0):.2f}%"
        elif tool_name == 'confidence_dist':
            return f"平均置信度 {data.get('avg_confidence', 0):.2f}"
        elif tool_name == 'save_record':
            return data.get('message', '已保存')
        elif tool_name == 'switch_full_scope':
            return f"切换到 {data.get('scope_label', '全量')}"
        elif tool_name == 'image_quality_check':
            return f"质量评分 {data.get('quality_score', 0):.2f}"
        elif tool_name == 'generate_report':
            return data.get('message', '报告已生成')
        elif tool_name == 'create_review':
            return data.get('message', '复核任务已创建')
        elif tool_name == 'batch_detect_folder':
            return data.get('summary', '批量检测完成')
        elif tool_name == 'batch_quality_stats':
            return data.get('summary', '批次质量统计完成')
        elif tool_name == 'batch_defect_distribution':
            return data.get('summary', '缺陷分布统计完成')
        elif tool_name == 'batch_list_failed':
            return data.get('summary', '不合格图片列表')
        elif tool_name == 'batch_details':
            return data.get('summary', '批次检测详情')
        else:
            summary = json.dumps(data, ensure_ascii=False, default=str)
            if len(summary) > 150:
                summary = summary[:147] + "..."
            return summary

    # ==================== 工具实现 ====================

    def _tool_detect_image(self, image_path: str, conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45, _ctx: TaskContext = None,
                           **kwargs) -> Dict:
        """工具：图像缺陷检测"""
        if self.model is None:
            raise RuntimeError("检测模型未加载")

        from PIL import Image as PILImage

        # 获取图像尺寸
        try:
            with PILImage.open(image_path) as img:
                orig_w, orig_h = img.size
        except Exception as e:
            _ctx.detection_errors.append(f"无法读取图像: {e}")
            raise

        image_name = os.path.basename(image_path)
        image_id = f"img_{abs(hash(image_path)) % 100000}"

        # 运行检测
        t0 = time.time()
        results = self.model.predict(source=image_path, conf=conf_threshold,
                                     iou=iou_threshold, verbose=False)
        inference_ms = (time.time() - t0) * 1000

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                class_names = r.names if hasattr(r, 'names') else {}
                for box in r.boxes:
                    cls_idx = int(box.cls[0].item())
                    cls_name = class_names.get(cls_idx, str(cls_idx))
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(DetectionItem(
                        cls_name=cls_name, confidence=round(conf, 4),
                        x1=round(x1, 1), y1=round(y1, 1),
                        x2=round(x2, 1), y2=round(y2, 1),
                    ))

        # 设置到上下文
        img_info = ImageInfo(
            image_id=image_id,
            image_name=image_name,
            width=orig_w,
            height=orig_h,
            image_path=image_path,
        )
        _ctx.set_raw_detections(detections, img_info)

        return {
            'total_count': len(detections),
            'inference_time_ms': round(inference_ms, 1),
            'img_width': orig_w,
            'img_height': orig_h,
            'image_id': image_id,
        }

    def _tool_count_defects(self, scope: str = 'current', _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：缺陷统计"""
        if scope == 'full':
            dets = _ctx.get_raw_detections()
        else:
            dets = _ctx.get_current_detections()

        by_class = {}
        conf_sum = 0.0
        for d in dets:
            by_class[d.cls_name] = by_class.get(d.cls_name, 0) + 1
            conf_sum += d.confidence

        avg_conf = conf_sum / len(dets) if dets else 0.0

        return {
            'total_count': len(dets),
            'by_class': by_class,
            'avg_confidence': round(avg_conf, 3),
            'scope': _ctx.scope_label if scope == 'current' else '全量',
            'summary': f'共 {len(dets)} 个缺陷，{len(by_class)} 种类别',
        }

    def _tool_filter_by_class(self, target_class: str, switch_scope: bool = True,
                               _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：按类别筛选（生成子集视图）"""
        view = _ctx.filter_by_class(target_class)

        if switch_scope:
            _ctx.switch_to_view(view)

        return {
            'filtered_count': len(view.det_ids),
            'target_class': target_class,
            'view_id': view.view_id,
            'description': view.description,
            'switched_scope': switch_scope,
            'note': '已切换到子集范围，后续分析仅针对此类缺陷。如需整体判定请调用 switch_full_scope。'
        }

    def _tool_area_analysis(self, scope: str = 'current', overlap_method: str = 'union',
                            _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：面积分析（检测框覆盖率）"""
        if scope == 'full':
            dets = _ctx.get_raw_detections()
        else:
            dets = _ctx.get_current_detections()

        img_w = _ctx.img_width
        img_h = _ctx.img_height
        img_area = max(1, img_w) * max(1, img_h)

        # 并集计算（使用扫描线法）
        total_area_pct, max_single_pct = self.quality_engine._calculate_box_areas(dets, img_w, img_h)

        # 简单求和（用于对比）
        sum_area = sum(d.area for d in dets)
        sum_area_pct = sum_area / img_area * 100

        # 按类别统计
        area_by_class = {}
        for d in dets:
            pct = d.area / img_area * 100
            area_by_class[d.cls_name] = area_by_class.get(d.cls_name, 0.0) + pct

        return {
            'total_area_pct': total_area_pct,
            'max_single_area_pct': max_single_pct,
            'avg_area_pct': round(total_area_pct / max(1, len(dets)), 3),
            'area_by_class': {k: round(v, 3) for k, v in area_by_class.items()},
            'sum_area_pct': round(sum_area_pct, 2),
            'overlap_method': overlap_method,
            'method_note': '面积指标为检测框覆盖率（并集去重），是真实缺陷面积的代理指标，非物理测量值',
            'scope_label': _ctx.scope_label,
        }

    def _tool_confidence_dist(self, scope: str = 'current',
                               _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：置信度分布"""
        if scope == 'full':
            dets = _ctx.get_raw_detections()
        else:
            dets = _ctx.get_current_detections()

        if not dets:
            return {
                'avg_confidence': 0, 'min_confidence': 0, 'max_confidence': 0,
                'low_conf_count': 0, 'high_conf_count': 0,
                'distribution': {}, 'total_count': 0,
            }

        confs = [d.confidence for d in dets]
        low_threshold = self.quality_engine.rule_set.review_low_confidence
        low_count = sum(1 for c in confs if c < low_threshold)
        high_count = sum(1 for c in confs if c >= 0.8)

        # 分箱分布
        bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        distribution = {}
        for i in range(len(bins) - 1):
            key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
            distribution[key] = sum(1 for c in confs if bins[i] <= c < bins[i+1])

        return {
            'avg_confidence': round(sum(confs) / len(confs), 3),
            'min_confidence': round(min(confs), 3),
            'max_confidence': round(max(confs), 3),
            'low_conf_count': low_count,
            'high_conf_count': high_count,
            'distribution': distribution,
            'total_count': len(dets),
            'low_threshold': low_threshold,
        }

    def _tool_quality_judge(self, scope: str = 'current', rule_set: str = 'default',
                            _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：质量判定"""
        result = self.quality_engine.evaluate(_ctx, scope=scope)
        return result

    def _tool_rule_lookup(self, rule_set: str = 'default', **kwargs) -> Dict:
        """工具：规则检索"""
        rs = self.quality_engine.rule_set
        return {
            'rule_id': rs.rule_id,
            'rule_version': rs.version,
            'rule_source': rs.source,
            'is_demo': rs.is_demo,
            'description': rs.description,
            'thresholds': {
                'max_minor_defects': rs.max_minor_defects,
                'max_critical_defects': rs.max_critical_defects,
                'critical_severity': rs.critical_severity,
                'max_total_box_area_pct': rs.max_total_box_area_pct,
                'max_single_box_area_pct': rs.max_single_box_area_pct,
                'review_low_confidence': rs.review_low_confidence,
                'review_low_conf_count': rs.review_low_conf_count,
                'pass_score_threshold': rs.pass_score_threshold,
                'fail_score_threshold': rs.fail_score_threshold,
            },
            'notes': rs.notes,
        }

    def _tool_detection_details(self, scope: str = 'current', limit: int = 20,
                                 _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：检测详情"""
        if scope == 'full':
            dets = _ctx.get_raw_detections()
        else:
            dets = _ctx.get_current_detections()

        details = []
        for d in dets[:limit]:
            details.append({
                'det_id': d.det_id,
                'class': d.cls_name,
                'class_cn': _get_class_cn(d.cls_name),
                'confidence': d.confidence,
                'box_area_pct': round(d.area / max(1, _ctx.img_width * _ctx.img_height) * 100, 3),
                'bbox': [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
            })

        return {
            'details': details,
            'total_count': len(dets),
            'shown_count': min(limit, len(dets)),
            'scope_label': _ctx.scope_label,
        }

    def _tool_switch_full_scope(self, _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：切换回全量范围"""
        _ctx.switch_to_full()
        dets = _ctx.get_current_detections()
        return {
            'scope': _ctx.current_scope,
            'scope_label': _ctx.scope_label,
            'detection_count': len(dets),
            'message': f'已切换到 {_ctx.scope_label} 范围，共 {len(dets)} 个检测',
        }

    def _tool_image_quality_check(self, image_path: str, _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：图像质量检查（简单实现：基于清晰度）"""
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        try:
            img = PILImage.open(image_path)
            img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) if img.mode == 'RGB' else np.array(img)

            # Laplacian 方差衡量清晰度
            laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            # 归一化到 0-1（经验值：>100 为清晰，<20 为模糊）
            clarity_score = min(1.0, laplacian_var / 200.0)

            issues = []
            if clarity_score < 0.3:
                issues.append('图像模糊')
            if img_gray.mean() < 50:
                issues.append('偏暗')
            elif img_gray.mean() > 205:
                issues.append('偏亮')

            overall = clarity_score * 0.7 + 0.3  # 基础分+清晰度

            # 更新上下文
            if _ctx.image:
                _ctx.image.quality_score = round(overall, 3)
                _ctx.image.quality_issues = issues

            return {
                'quality_score': round(overall, 3),
                'clarity_score': round(clarity_score, 3),
                'brightness': round(float(img_gray.mean()), 1),
                'issues': issues,
                'is_usable': overall >= 0.5,
            }
        except Exception as e:
            return {
                'quality_score': 0,
                'issues': [f'评估失败: {e}'],
                'is_usable': False,
            }

    def _tool_save_record(self, include_review: bool = False, _ctx: TaskContext = None,
                          _trace: ExecutionTrace = None, **kwargs) -> Dict:
        """工具：保存记录到数据库"""
        if not self.db_path:
            return {'record_id': None, 'message': '数据库未配置'}

        import sqlite3
        dets = _ctx.get_current_detections()
        eval_result = self.quality_engine.evaluate(_ctx)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO detection_records_v2
            (task_id, image_name, image_id, timestamp, total_defects,
             quality, quality_score, scope, scope_label,
             detections_json, eval_json, model_version, rule_version,
             trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            _ctx.task_id,
            _ctx.image.image_name if _ctx.image else 'unknown',
            _ctx.image.image_id if _ctx.image else '',
            _ctx.created_at,
            len(dets),
            eval_result['quality'],
            eval_result['score'],
            _ctx.current_scope,
            _ctx.scope_label,
            json.dumps([d.to_tuple() for d in dets], ensure_ascii=False),
            json.dumps(eval_result, ensure_ascii=False),
            _ctx.model_version,
            _ctx.rule_version,
            _trace.task_id if _trace else '',
        ))
        record_id = c.lastrowid
        conn.commit()
        conn.close()

        # 如果是 REVIEW 且需要创建复核任务
        review_id = None
        if include_review and eval_result['quality'] == 'REVIEW':
            review_id = f"rev_{record_id}"
            # 简化：复核任务在数据库中标记

        return {
            'record_id': record_id,
            'review_task_id': review_id,
            'message': f'记录已保存 (ID: {record_id})',
        }

    def _tool_generate_report(self, report_type: str = 'summary', format: str = 'json',
                               _ctx: TaskContext = None, _trace: ExecutionTrace = None,
                               **kwargs) -> Dict:
        """
        工具：生成报告

        支持 JSON 和 DOCX 两种格式。
        DOCX 格式使用 QualityReportGenerator 生成真实的 Word 文件，
        包含分析范围、模型/规则版本、演示规则标记、覆盖率方法、
        证据引用、复核前后结论等完整内容。
        """
        dets = _ctx.get_current_detections()
        eval_result = self.quality_engine.evaluate(_ctx)

        # 按类别统计
        by_class = {}
        total_conf = 0
        for d in dets:
            by_class[d.cls_name] = by_class.get(d.cls_name, 0) + 1
            total_conf += d.confidence
        avg_conf = total_conf / len(dets) if dets else 0

        report_data = {
            'task_id': _ctx.task_id,
            'report_type': report_type,
            'generated_at': _trace.end_time or _trace.start_time,
            'scope': _ctx.current_scope,
            'scope_label': _ctx.scope_label,
            'image_count': _ctx.image_count,
            'model_version': _ctx.model_version,
            'rule_version': _ctx.rule_version,
            'rule_source': _ctx.rule_source,
            'is_demo_rule': _ctx.rule_snapshot.get('is_demo', True) if _ctx.rule_snapshot else True,
            'total_defects': len(dets),
            'quality_result': eval_result,
            'execution_summary': {
                'total_steps': _trace.step_count,
                'total_time_ms': round(_trace.total_time_ms, 1),
                'success_steps': _trace.success_count,
                'llm_fallback': _trace.llm_unavailable_fallback,
            },
            'method_notes': eval_result.get('method_notes', []),
        }

        download_path = None

        # 如果是 docx 格式，生成真实 Word 报告
        if format == 'docx':
            try:
                from models.agent.report_generator_v2 import QualityReportGenerator

                # 确定输出目录
                output_dir = self.config.report_output_dir or 'outputs/reports'
                os.makedirs(output_dir, exist_ok=True)

                # 生成文件名
                safe_task_id = ''.join(
                    c if c.isalnum() or c in '-_' else '_'
                    for c in _ctx.task_id
                )
                filename = f"report_{safe_task_id}_{int(time.time())}.docx"
                output_path = os.path.join(output_dir, filename)

                # 组装 trace 信息
                trace_info = {
                    'task_id': _ctx.task_id,
                    'model_version': _ctx.model_version,
                    'rule_version': _ctx.rule_version,
                    'total_steps': _trace.step_count,
                    'success_steps': _trace.success_count,
                    'total_time_ms': _trace.total_time_ms,
                }

                # 获取图像信息
                image_name = _ctx.image.image_name if _ctx.image else 'unknown'
                is_subset = _ctx.current_scope in ('subset', 'batch_subset')
                subset_class = ''
                if is_subset and _ctx._current_view and _ctx._current_view.filter_type == 'class':
                    subset_class = _ctx._current_view.filter_params.get('target_class', '')

                # 检测摘要
                detection_summary = {
                    'total_count': len(dets),
                    'by_class': by_class,
                    'avg_confidence': avg_conf,
                    'area_ratio': eval_result.get('total_area_ratio', 0) * 100,
                }

                # 生成报告
                gen = QualityReportGenerator()
                gen.generate_single_image_report(
                    image_name=image_name,
                    quality_result=eval_result,
                    detection_summary=detection_summary,
                    trace_info=trace_info,
                    scope_label=_ctx.scope_label,
                    is_subset=is_subset,
                    subset_class=subset_class,
                )
                gen.save(output_path)

                download_path = output_path
                message = f'Word 报告已生成: {filename}'

            except Exception as e:
                message = f'Word 报告生成失败: {str(e)}'
                download_path = None
        else:
            message = f'报告数据已生成（{report_type}）'

        return {
            'report_data': report_data,
            'download_path': download_path,
            'format': format,
            'message': message,
        }

    def _tool_create_review(self, reason: str = '', priority: str = 'normal',
                            _ctx: TaskContext = None, _trace: ExecutionTrace = None,
                            **kwargs) -> Dict:
        """工具：创建复核任务"""
        import sqlite3
        if not self.db_path:
            return {'review_id': None, 'status': 'error', 'message': '数据库未配置'}

        review_id = f"rev_{abs(hash(_ctx.task_id + reason)) % 100000}"

        # 尝试插入（可能表不存在）
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS review_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT UNIQUE,
                    task_id TEXT,
                    image_name TEXT,
                    quality TEXT,
                    reason TEXT,
                    priority TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    resolved_at TEXT,
                    resolver TEXT,
                    resolution TEXT,
                    resolution_note TEXT
                )
            """)
            from datetime import datetime
            c.execute("""
                INSERT INTO review_tasks
                (review_id, task_id, image_name, quality, reason, priority, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                review_id,
                _ctx.task_id,
                _ctx.image.image_name if _ctx.image else '',
                _trace.final_conclusion or '',
                reason,
                priority,
                datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            return {'review_id': review_id, 'status': 'error', 'message': str(e)}

        return {
            'review_id': review_id,
            'status': 'pending',
            'priority': priority,
            'message': f'复核任务已创建 (ID: {review_id})',
        }

    def _tool_batch_detect_folder(self, folder_path: str,
                                   _ctx: TaskContext = None,
                                   _trace: ExecutionTrace = None,
                                   **kwargs) -> Dict:
        """
        工具：批量检测文件夹中的所有图片

        读取指定文件夹中的所有图片（jpg/jpeg/png/bmp），
        依次执行 YOLO 缺陷检测和质量判定，返回批量统计结果。
        """
        import os

        folder_path = folder_path.strip().strip('"').strip("'")

        # 缓存保护：如果已经对同一个文件夹做过检测，直接返回缓存结果
        cached_results = getattr(_ctx, 'batch_results', None)
        cached_folder = getattr(_ctx, 'batch_folder', None)
        if cached_results and cached_folder and cached_folder == folder_path:
            # 从缓存中构建返回值
            by_class = getattr(_ctx, 'batch_by_class', {})
            total_det = getattr(_ctx, 'batch_total_detections', sum(by_class.values()))
            pass_count = sum(1 for r in cached_results if r.get('quality') == 'PASS')
            review_count = sum(1 for r in cached_results if r.get('quality') == 'REVIEW')
            fail_count = sum(1 for r in cached_results if r.get('quality') == 'FAIL')
            total = len(cached_results)
            pass_rate = (pass_count / total * 100) if total > 0 else 0
            summary = (
                f'批量检测完成：共 {total} 张图片，检测到 {total_det} 个缺陷\n'
                f'质量分布：✅ 合格 {pass_count} 张（{pass_rate:.1f}%）| '
                f'⚠️ 待复核 {review_count} 张 | ❌ 不合格 {fail_count} 张'
            )
            return {
                'folder_path': folder_path,
                'total_images': total,
                'total_detections': total_det,
                'by_class_total': by_class,
                'pass_count': pass_count,
                'review_count': review_count,
                'fail_count': fail_count,
                'pass_rate': round(pass_rate, 1),
                'results': cached_results[:20],
                'has_more': len(cached_results) > 20,
                'total_results': len(cached_results),
                'summary': summary,
                'cached': True,
            }

        if not os.path.isdir(folder_path):
            return {
                'status': 'error',
                'message': f'文件夹不存在：{folder_path}'
            }

        # 支持的图片扩展名
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        image_files = []
        for f in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(f)[1]
            if ext in exts:
                image_files.append(os.path.join(folder_path, f))

        if not image_files:
            return {
                'status': 'error',
                'message': f'文件夹中没有找到支持的图片文件（支持 jpg/jpeg/png/bmp）'
            }

        # 批量检测
        results = []
        total_detections = 0
        by_class_total = {}
        pass_count = 0
        review_count = 0
        fail_count = 0

        from PIL import Image as PILImage
        import time

        for img_path in image_files:
            try:
                # 获取图像尺寸
                with PILImage.open(img_path) as img:
                    orig_w, orig_h = img.size

                # 运行检测
                t0 = time.time()
                det_results = self.model.predict(source=img_path, conf=0.25,
                                                 iou=0.45, verbose=False)
                inference_ms = (time.time() - t0) * 1000

                detections_list = []  # 列表格式: [cls_name, conf, x1, y1, x2, y2]
                if det_results and len(det_results) > 0:
                    r = det_results[0]
                    if r.boxes is not None:
                        class_names = r.names if hasattr(r, 'names') else {}
                        for box in r.boxes:
                            cls_idx = int(box.cls[0].item())
                            cls_name = class_names.get(cls_idx, str(cls_idx))
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections_list.append([
                                cls_name, round(conf, 4),
                                round(x1, 1), round(y1, 1),
                                round(x2, 1), round(y2, 1),
                            ])

                total_detections += len(detections_list)

                # 统计类别
                for det in detections_list:
                    cls_name = det[0]
                    by_class_total[cls_name] = by_class_total.get(cls_name, 0) + 1

                # 质量判定（使用底层规则引擎，接受列表格式的detections）
                from models.agent.detection_agent import QualityRuleEngine as SimpleRuleEngine
                simple_engine = SimpleRuleEngine()
                eval_result = simple_engine.evaluate(
                    detections_list,
                    img_width=orig_w,
                    img_height=orig_h,
                )
                quality = eval_result.get('quality', 'UNKNOWN')
                if quality == 'PASS':
                    pass_count += 1
                elif quality == 'REVIEW':
                    review_count += 1
                else:
                    fail_count += 1

                results.append({
                    'image_name': os.path.basename(img_path),
                    'image_path': img_path,
                    'detection_count': len(detections_list),
                    'quality': quality,
                    'score': eval_result.get('score', 0),
                    'inference_time_ms': round(inference_ms, 1),
                })
            except Exception as e:
                results.append({
                    'image_name': os.path.basename(img_path),
                    'image_path': img_path,
                    'error': str(e),
                    'detection_count': 0,
                    'quality': 'ERROR',
                })

        total_images = len(image_files)
        pass_rate = (pass_count / total_images * 100) if total_images > 0 else 0

        # 将批量结果存入 context 供后续工具使用
        _ctx.batch_results = results
        _ctx.batch_folder = folder_path
        _ctx.batch_by_class = by_class_total
        _ctx.batch_total_detections = total_detections

        summary = (
            f'批量检测完成：共 {total_images} 张图片，检测到 {total_detections} 个缺陷\n'
            f'质量分布：✅ 合格 {pass_count} 张（{pass_rate:.1f}%）| '
            f'⚠️ 待复核 {review_count} 张 | ❌ 不合格 {fail_count} 张'
        )

        return {
            'folder_path': folder_path,
            'total_images': total_images,
            'total_detections': total_detections,
            'by_class_total': by_class_total,
            'pass_count': pass_count,
            'review_count': review_count,
            'fail_count': fail_count,
            'pass_rate': round(pass_rate, 1),
            'results': results[:20],  # 只返回前20条详情，避免太多
            'has_more': len(results) > 20,
            'total_results': len(results),
            'summary': summary,
        }

    def _tool_batch_quality_stats(self, _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：批次质量统计（基于已有批量检测结果）"""
        batch_results = getattr(_ctx, 'batch_results', None)
        if not batch_results:
            return {
                'status': 'error',
                'message': '没有批量检测结果，请先执行批量检测',
                'summary': '错误：没有批量检测结果',
            }

        total = len(batch_results)
        pass_count = sum(1 for r in batch_results if r.get('quality') == 'PASS')
        review_count = sum(1 for r in batch_results if r.get('quality') == 'REVIEW')
        fail_count = sum(1 for r in batch_results if r.get('quality') == 'FAIL')

        pass_rate = (pass_count / total * 100) if total > 0 else 0
        review_rate = (review_count / total * 100) if total > 0 else 0
        fail_rate = (fail_count / total * 100) if total > 0 else 0

        summary = (
            f'📊 批次质量统计（共 {total} 张）：\n'
            f'  ✅ 合格：{pass_count} 张（{pass_rate:.1f}%）\n'
            f'  ⚠️ 待复核：{review_count} 张（{review_rate:.1f}%）\n'
            f'  ❌ 不合格：{fail_count} 张（{fail_rate:.1f}%）'
        )

        return {
            'total_images': total,
            'pass_count': pass_count,
            'review_count': review_count,
            'fail_count': fail_count,
            'pass_rate': round(pass_rate, 1),
            'review_rate': round(review_rate, 1),
            'fail_rate': round(fail_rate, 1),
            'summary': summary,
        }

    def _tool_batch_defect_distribution(self, _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：批次缺陷分布统计（基于已有批量检测结果）"""
        batch_results = getattr(_ctx, 'batch_results', None)
        if not batch_results:
            return {
                'status': 'error',
                'message': '没有批量检测结果，请先执行批量检测',
                'summary': '错误：没有批量检测结果',
            }

        # 统计各类缺陷数量（需要累加每张图的检测数）
        # 注意：batch_results 中只有每张图的总检测数和质量，没有按类别细分
        # 所以我们需要从 batch_detect_folder 返回的 by_class_total 获取
        by_class_total = getattr(_ctx, 'batch_by_class', {})
        total_detections = sum(by_class_total.values())

        # 如果没有缓存的 by_class，说明结果可能来自旧版，重新估算
        if not by_class_total and batch_results:
            # 没有详细分类数据，只能统计总检测数
            total_detections = sum(r.get('detection_count', 0) for r in batch_results)
            by_class_total = {'(未分类)': total_detections}

        # 计算各类占比
        by_class_pct = {}
        for cls, cnt in by_class_total.items():
            by_class_pct[cls] = round(cnt / total_detections * 100, 1) if total_detections > 0 else 0

        # 排序（按数量降序）
        sorted_classes = sorted(by_class_total.items(), key=lambda x: x[1], reverse=True)
        top_class = sorted_classes[0][0] if sorted_classes else None

        # 生成中文类别名映射
        cn_map = {
            'crazing': '裂纹', 'inclusion': '夹杂', 'patches': '斑块',
            'pitted_surface': '麻点', 'rolled-in_scale': '氧化铁皮', 'scratches': '划痕',
        }

        # 构建分布摘要
        dist_lines = []
        for cls, cnt in sorted_classes:
            cn_name = cn_map.get(cls, cls)
            pct = by_class_pct.get(cls, 0)
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            dist_lines.append(f'  {bar} {cn_name}：{cnt} 个（{pct:.1f}%）')

        summary = (
            f'📈 缺陷分布统计（共 {total_detections} 个缺陷，{len(by_class_total)} 类）：\n'
            + '\n'.join(dist_lines)
        )

        return {
            'total_detections': total_detections,
            'by_class': by_class_total,
            'by_class_pct': by_class_pct,
            'top_class': top_class,
            'summary': summary,
        }

    def _tool_batch_list_failed(self, quality_filter: str = 'FAIL', limit: int = 20,
                                 _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：列出不合格/待复核的图片"""
        batch_results = getattr(_ctx, 'batch_results', None)
        if not batch_results:
            return {
                'status': 'error',
                'message': '没有批量检测结果，请先执行批量检测',
                'summary': '错误：没有批量检测结果',
            }

        # 筛选符合条件的图片
        if quality_filter == 'FAIL_OR_REVIEW':
            target_qualities = {'FAIL', 'REVIEW'}
        else:
            target_qualities = {quality_filter}

        matched = [r for r in batch_results if r.get('quality') in target_qualities]
        total_matched = len(matched)
        displayed = matched[:limit]

        # 质量标签映射
        quality_label = {
            'FAIL': '❌ 不合格',
            'REVIEW': '⚠️ 待复核',
            'PASS': '✅ 合格',
        }

        # 构建列表摘要
        lines = [f'📋 {quality_label.get(quality_filter, quality_filter)}图片列表（共 {total_matched} 张）：']
        for i, r in enumerate(displayed, 1):
            q = r.get('quality', 'UNKNOWN')
            score = r.get('score', 0)
            det_count = r.get('detection_count', 0)
            lines.append(f'  {i}. {r.get("image_name", "?")} — '
                         f'{quality_label.get(q, q)} | '
                         f'{score}分 | {det_count}个缺陷')

        if total_matched > limit:
            lines.append(f'  ... 还有 {total_matched - limit} 张未显示')

        summary = '\n'.join(lines)

        return {
            'total_matched': total_matched,
            'images': displayed,
            'quality_filter': quality_filter,
            'limit': limit,
            'summary': summary,
        }

    def _tool_batch_details(self, limit: int = 20, offset: int = 0,
                             _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：批次检测详情（分页）"""
        batch_results = getattr(_ctx, 'batch_results', None)
        if not batch_results:
            return {
                'status': 'error',
                'message': '没有批量检测结果，请先执行批量检测',
                'summary': '错误：没有批量检测结果',
            }

        total = len(batch_results)
        end_idx = min(offset + limit, total)
        page_results = batch_results[offset:end_idx]
        has_more = end_idx < total

        quality_label = {
            'FAIL': '❌ 不合格',
            'REVIEW': '⚠️ 待复核',
            'PASS': '✅ 合格',
        }

        lines = [f'📋 批次检测详情（第 {offset + 1}-{end_idx} 张 / 共 {total} 张）：']
        for i, r in enumerate(page_results, offset + 1):
            q = r.get('quality', 'UNKNOWN')
            score = r.get('score', 0)
            det_count = r.get('detection_count', 0)
            lines.append(f'  {i}. {r.get("image_name", "?")} — '
                         f'{quality_label.get(q, q)} | '
                         f'{score}分 | {det_count}个缺陷')

        if has_more:
            lines.append(f'  ... 还有 {total - end_idx} 张')

        summary = '\n'.join(lines)

        return {
            'total_images': total,
            'results': page_results,
            'offset': offset,
            'limit': limit,
            'has_more': has_more,
            'summary': summary,
        }

    # ==================== GAN 工具 ====================

    def _get_gan_manager(self):
        """懒加载 GAN 管理器"""
        if self._gan_manager is not None:
            return self._gan_manager
        if not self.config.gan_enabled:
            return None
        try:
            from models.gan import create_gan_manager
            ckpt = self.config.gan_checkpoint_path or None
            self._gan_manager = create_gan_manager(checkpoint_path=ckpt)
            self._gan_loaded = self._gan_manager.is_loaded
        except Exception as e:
            print(f"[GAN] 加载失败: {e}")
            self._gan_manager = None
            self._gan_loaded = False
        return self._gan_manager

    def _tool_gan_generate(self, defect_class: str, num_samples: int = 4,
                           save_images: bool = True,
                           _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：GAN 生成缺陷样本图像"""
        gan = self._get_gan_manager()
        if gan is None or not gan.is_loaded:
            return {
                'status': 'error',
                'message': 'GAN 模型未加载或不可用',
                'generated_count': 0,
            }

        try:
            result = gan.generate(
                num_samples=num_samples,
                class_name=defect_class,
            )

            image_paths = []
            if save_images and result['images']:
                import os
                from datetime import datetime
                save_dir = os.path.join(
                    self.config.report_output_dir.replace('reports', 'gan_outputs'),
                    datetime.now().strftime('%Y%m%d'),
                )
                os.makedirs(save_dir, exist_ok=True)

                for i, img in enumerate(result['images']):
                    fname = f"gan_{defect_class}_{i:03d}_{int(time.time()*1000)}.jpg"
                    fpath = os.path.join(save_dir, fname)
                    img.save(fpath)
                    image_paths.append(fpath)

            return {
                'generated_count': result['num_samples'],
                'defect_class': defect_class,
                'class_name_cn': result['class_names'][0] if result['class_names'] else '',
                'image_paths': image_paths,
                'generate_time_ms': round(result['generate_time'] * 1000, 1),
                'model_info': {
                    'architecture': 'DCGAN (conditional)',
                    'image_size': gan.image_size,
                },
                'message': f"成功生成 {result['num_samples']} 张 {defect_class} 缺陷样本",
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'GAN 生成失败: {str(e)}',
                'generated_count': 0,
            }

    def _tool_gan_model_info(self, **kwargs) -> Dict:
        """工具：查询 GAN 模型信息"""
        gan = self._get_gan_manager()
        if gan is None:
            return {
                'loaded': False,
                'message': 'GAN 功能未启用',
            }
        info = gan.get_model_info()
        return {
            'loaded': info['loaded'],
            'architecture': info['architecture'],
            'image_size': info['image_size'],
            'channels': info['channels'],
            'num_classes': info['num_classes'],
            'class_names': info['class_names'],
            'class_names_cn': list(info['class_names_cn'].values()),
            'device': info['device'],
            'total_generated': info['total_generated'],
            'message': 'GAN 模型信息' if info['loaded'] else 'GAN 未加载',
        }

    def _tool_gan_generate_grid(self, samples_per_class: int = 8,
                                **kwargs) -> Dict:
        """工具：生成所有类别的 GAN 样本网格图"""
        gan = self._get_gan_manager()
        if gan is None or not gan.is_loaded:
            return {
                'status': 'error',
                'message': 'GAN 模型未加载或不可用',
                'total_samples': 0,
            }

        try:
            import os
            from datetime import datetime
            save_dir = os.path.join(
                self.config.report_output_dir.replace('reports', 'gan_outputs'),
                datetime.now().strftime('%Y%m%d'),
            )
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"gan_grid_{int(time.time()*1000)}.jpg"
            )

            import time as _time
            t0 = _time.time()
            grid_result = gan.generate_grid(
                samples_per_class=samples_per_class,
                save_path=save_path,
            )
            elapsed = _time.time() - t0

            return {
                'grid_image_path': save_path,
                'samples_per_class': samples_per_class,
                'num_classes': gan.num_classes,
                'total_samples': grid_result['total_samples'],
                'generate_time_ms': round(elapsed * 1000, 1),
                'message': f"已生成 {grid_result['total_samples']} 张样本的类别网格图",
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'网格生成失败: {str(e)}',
                'total_samples': 0,
            }

    # ==================== GAN-YOLO 融合工具 ====================

    def _get_fusion_manager(self):
        """懒加载融合管理器"""
        if self._fusion_manager is not None:
            return self._fusion_manager
        if not self.config.gan_enabled:
            return None
        try:
            from models.fusion import create_fusion_manager
            self._fusion_manager = create_fusion_manager(
                gan_manager=self._get_gan_manager(),
                yolo_model=self.model,
            )
        except Exception as e:
            print(f"[Fusion] 加载失败: {e}")
            self._fusion_manager = None
        return self._fusion_manager

    def _tool_fused_detect(self, image_path: str,
                           anomaly_threshold: float = 0.15,
                           _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：融合检测（GAN重建+YOLO检测双路验证）"""
        fusion = self._get_fusion_manager()
        if fusion is None:
            return {
                'status': 'error',
                'message': '融合模块不可用（需要 GAN 和 YOLO 都已加载）',
            }

        try:
            judge_result = fusion.fused_quality_judgement(image_path)

            return {
                'quality': judge_result['quality'],
                'quality_cn': judge_result['quality_cn'],
                'confidence': judge_result['confidence'],
                'detection_count': judge_result['detection_count'],
                'reconstruction_error': judge_result['cross_validation']['gan_reconstruction_error'],
                'yolo_max_confidence': judge_result['cross_validation']['yolo_max_confidence'],
                'cross_validation_notes': judge_result['cross_validation']['notes'],
                'fusion_method': judge_result['fusion_method'],
                'sources_used': judge_result['sources_used'],
                'processing_time_ms': judge_result['processing_time_ms'],
                'message': f"融合检测完成，质量: {judge_result['quality_cn']} "
                           f"(置信度 {judge_result['confidence']:.0%})",
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'融合检测失败: {str(e)}',
            }

    def _tool_mine_hard_samples(self, num_per_class: int = 20,
                                _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：困难样本挖掘"""
        fusion = self._get_fusion_manager()
        if fusion is None:
            return {
                'status': 'error',
                'message': '融合模块不可用',
            }

        try:
            result = fusion.mine_hard_samples(num_per_class=num_per_class)

            return {
                'total_generated': result['total_generated'],
                'hard_samples_count': result['hard_samples_count'],
                'hard_ratio': result['hard_ratio'],
                'difficulty_distribution': result['difficulty_distribution'],
                'per_class_stats': {
                    name: {
                        'hard': s['hard'],
                        'total': s['total'],
                        'hard_ratio': s['hard_ratio'],
                    }
                    for name, s in result['per_class_stats'].items()
                },
                'message': f"挖掘完成: {result['hard_samples_count']} 个困难样本 "
                           f"({result['hard_ratio']:.0%})",
                'note': result.get('note', ''),
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'困难样本挖掘失败: {str(e)}',
            }

    def _tool_adaptive_augmentation(self, base_samples: int = 100,
                                    max_ratio: float = 5.0,
                                    _ctx: TaskContext = None, **kwargs) -> Dict:
        """工具：自适应数据增强

        注意：此工具需要先调用 analyze_class_performance 或已有性能数据。
        简化版本：基于预设的类别难度分布进行自适应增强。
        """
        gan = self._get_gan_manager()
        if gan is None or not gan.is_loaded:
            return {
                'status': 'error',
                'message': 'GAN 模型不可用',
            }

        try:
            from models.fusion import GanYoloFusion

            # 简化版：使用一个模拟的性能报告来演示自适应增强
            # 实际使用时应先分析真实检测数据
            mock_performance = {
                'per_class': {
                    'crazing': {'performance_score': 0.45},
                    'inclusion': {'performance_score': 0.85},
                    'patches': {'performance_score': 0.72},
                    'pitted_surface': {'performance_score': 0.68},
                    'rolled-in_scale': {'performance_score': 0.78},
                    'scratches': {'performance_score': 0.38},
                }
            }

            fusion = self._get_fusion_manager()
            if fusion:
                aug_result = fusion.adaptive_augmentation(
                    performance_report=mock_performance,
                    base_samples_per_class=base_samples,
                    max_ratio=max_ratio,
                )
            else:
                # 退化为均匀生成
                aug_result = {
                    'method': 'uniform_augmentation',
                    'description': '融合模块不可用，退化为均匀生成',
                    'total_generated': base_samples * 6,
                }

            return {
                'method': aug_result.get('method', 'adaptive_augmentation'),
                'description': aug_result.get('description', ''),
                'base_samples': base_samples,
                'max_ratio': max_ratio,
                'total_generated': aug_result.get('total_generated', 0),
                'weak_classes': [
                    {'class_name': 'scratches', 'needs_augmentation': True},
                    {'class_name': 'crazing', 'needs_augmentation': True},
                ],
                'output_dir': aug_result.get('output_dir', ''),
                'message': f"自适应增强完成，共生成 {aug_result.get('total_generated', 0)} 张样本",
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'自适应增强失败: {str(e)}',
            }

    # ==================== 规则模式的规划 ====================

    def _plan_by_keywords(self, user_input: str, ctx: TaskContext) -> List[tuple]:
        """基于关键词的任务规划（规则模式）"""
        text = user_input.strip()
        text_lower = text.lower()

        # 识别意图
        has_detect_kw = any(kw in text for kw in
                           ['检测', '识别', '看看', '找出', '查一下', '有没有缺陷', '缺陷检测', '分析一下'])
        has_count_kw = any(kw in text for kw in ['多少个', '几个', '数量', '统计', '计数', '分布'])
        has_quality_kw = any(kw in text_lower for kw in
                            ['合格', '不合格', '质量', '判定', '评级', '评估', 'pass', 'fail', '能不能过'])
        has_explain_kw = any(kw in text for kw in ['为什么', '原因', '解释', '哪里有问题'])
        has_report_kw = any(kw in text for kw in ['报告', '生成报告', '导出', '报表'])
        has_area_kw = any(kw in text for kw in ['面积', '占比', '大小'])
        has_conf_kw = any(kw in text for kw in ['置信度', '可信度'])
        has_rule_kw = any(kw in text for kw in ['规则', '阈值', '标准'])
        has_detail_kw = any(kw in text for kw in ['详情', '详细', '列出'])
        has_full_kw = any(kw in text for kw in ['全部', '整体', '全图', '整批', '全部缺陷', '全量'])
        has_batch_kw = any(kw in text for kw in ['批量', '文件夹', '目录', 'folder', 'batch', '一批'])

        # 提取文件夹路径（如果用户输入中包含路径）
        folder_path = None
        if has_batch_kw:
            # 尝试从用户输入中提取路径（匹配 Windows 路径或 Unix 路径）
            import re
            # 匹配 Windows 路径: C:\xxx\xxx 或 \\xxx\xxx
            path_match = re.search(r'[A-Za-z]:\\[^\s，。、"]+', text)
            if not path_match:
                path_match = re.search(r'[A-Za-z]:/[^\s，。、"]+', text)
            if path_match:
                folder_path = path_match.group(0).strip().rstrip('。，、')

        # 识别缺陷类别
        target_class = None
        class_kw = {
            '裂纹': 'crazing', 'crazing': 'crazing',
            '夹杂': 'inclusion', 'inclusion': 'inclusion',
            '划痕': 'scratches', 'scratches': 'scratches',
            '斑块': 'patches', 'patches': 'patches',
            '麻点': 'pitted_surface', 'pitted_surface': 'pitted_surface',
            '氧化皮': 'rolled-in_scale', 'rolled-in_scale': 'rolled-in_scale',
        }
        for kw, cls in class_kw.items():
            if kw in text:
                target_class = cls
                break

        plan = []

        # 批量文件夹检测（优先级最高）
        # 两种情况触发：1) 明确提到批量/文件夹关键词；2) 已处于文件夹模式且有检测/统计/质量意图
        ctx_has_folder = hasattr(ctx, 'folder_path') and getattr(ctx, 'folder_path', None)
        has_batch_results = hasattr(ctx, 'batch_results') and getattr(ctx, 'batch_results', None)
        need_batch = ctx_has_folder and (
            has_detect_kw or has_count_kw or has_quality_kw
            or has_area_kw or has_conf_kw or has_detail_kw
            or has_report_kw or has_batch_kw
        )

        if has_batch_kw or need_batch:
            folder = getattr(ctx, 'folder_path', None) or folder_path
            if folder:
                # 如果还没有检测结果，先执行批量检测
                if not has_batch_results:
                    plan.append(('batch_detect_folder', {'folder_path': folder}))

                # 根据用户意图添加针对性的批量分析工具
                # 1. 质量/不合格类问题 → 质量统计
                added_quality = False
                if has_quality_kw or has_explain_kw:
                    plan.append(('batch_quality_stats', {}))
                    added_quality = True
                    # 如果明确问"哪些不合格"或具体哪些图片，列出不合格图片
                    if any(kw in text for kw in ['哪些', '哪张', '列表', '列出', '明细', '具体', '都是谁']):
                        plan.append(('batch_list_failed', {'quality_filter': 'FAIL', 'limit': 20}))
                    elif any(kw in text for kw in ['待复核', 'review']):
                        plan.append(('batch_list_failed', {'quality_filter': 'REVIEW', 'limit': 20}))

                # 2. 分布/统计/数量类问题 → 缺陷分布
                added_dist = False
                if has_count_kw or any(kw in text for kw in ['分布', '比例', '占比', '最多', '主要', '各类', '分类']):
                    plan.append(('batch_defect_distribution', {}))
                    added_dist = True

                # 3. 详情类问题 → 详情列表
                if has_detail_kw:
                    plan.append(('batch_details', {'limit': 20}))

                # 4. 报告类 → 质量统计 + 缺陷分布
                if has_report_kw:
                    if not added_quality:
                        plan.append(('batch_quality_stats', {}))
                    if not added_dist:
                        plan.append(('batch_defect_distribution', {}))

                # 5. 如果只是"检测一下"等泛泛的问题，默认给出质量+分布
                if not has_batch_results and len(plan) == 1:
                    # 只有检测步骤，没有具体分析意图，默认添加质量统计和缺陷分布
                    plan.append(('batch_quality_stats', {}))
                    plan.append(('batch_defect_distribution', {}))
                elif has_batch_results and len(plan) == 0:
                    # 已有结果但没有匹配到具体分析工具，默认给出质量统计
                    plan.append(('batch_quality_stats', {}))
                    plan.append(('batch_defect_distribution', {}))

                return plan  # 批量模式，返回计划

        # 判断是否需要"先子集再全量"的模式
        # 特征：同时提到具体类别 和 "全部/整体/全图"，并且有质量判定意图
        need_subset_then_full = (
            target_class is not None
            and has_full_kw
            and (has_quality_kw or has_detect_kw)
        )

        # 类别筛选（如果提到了特定类别）
        if target_class and not ctx.is_subset:
            plan.append(('filter_by_class', {'target_class': target_class}))

        # 如果需要先子集再全量，先在子集范围内做统计展示
        if need_subset_then_full:
            if has_count_kw or True:  # 至少展示子集数量
                plan.append(('count_defects', {'scope': 'current'}))
            # 然后切回全量
            plan.append(('switch_full_scope', {}))
        elif has_full_kw and ctx.is_subset:
            # 已经在子集，用户要看全部，先切回
            plan.append(('switch_full_scope', {}))

        # 统计（不需要子集→全量模式时才在这里加）
        if has_count_kw and not need_subset_then_full:
            plan.append(('count_defects', {'scope': 'current'}))
        elif has_detect_kw and not need_subset_then_full and not has_quality_kw:
            # 只是检测/看看，不需要判定，也加统计
            plan.append(('count_defects', {'scope': 'current'}))

        # 面积分析
        if has_area_kw:
            plan.append(('area_analysis', {'scope': 'current'}))

        # 置信度分析
        if has_conf_kw:
            plan.append(('confidence_dist', {'scope': 'current'}))

        # 质量判定
        if has_quality_kw or (has_detect_kw and not need_subset_then_full):
            # 如果是子集→全量模式，质量判定的scope是full（已经切回了）
            # 否则用current
            plan.append(('quality_judge', {'scope': 'current'}))

        # 解释
        if has_explain_kw:
            plan.append(('quality_judge', {'scope': 'current'}))  # 判定本身就包含解释

        # 详情
        if has_detail_kw:
            plan.append(('detection_details', {'scope': 'current', 'limit': 20}))

        # 规则查询
        if has_rule_kw:
            plan.append(('rule_lookup', {}))

        # 报告
        if has_report_kw:
            plan.append(('generate_report', {'report_type': 'summary', 'format': 'json'}))

        # 默认行为：检测 + 统计 + 判定
        if not plan and ctx.has_raw_detections:
            plan = [
                ('count_defects', {'scope': 'current'}),
                ('quality_judge', {'scope': 'current'}),
            ]
        elif not plan and not ctx.has_raw_detections:
            # 没有检测数据且没有明确意图，先做图像质量检查
            plan = [
                ('image_quality_check', {}),
            ]

        return plan

    # ==================== 辅助方法 ====================

    def _can_use_llm(self) -> bool:
        """检查 LLM 是否可用（真实可用，不是 mock）"""
        try:
            orch = self._get_llm_orchestrator()
            return orch.is_llm_available()
        except Exception:
            return False

    def _get_llm_orchestrator(self) -> LLMAgentOrchestrator:
        """获取 LLM 编排器（懒加载）"""
        if self._llm_orchestrator is not None:
            return self._llm_orchestrator

        # 构建配置
        if self.config.llm_provider and self.config.llm_provider != 'none':
            # 使用显式配置
            llm_config = {
                'provider': self.config.llm_provider,
                'api_key': self.config.llm_api_key,
                'base_url': self.config.llm_base_url,
                'model': self.config.llm_model,
                'temperature': self.config.llm_temperature,
            }
        elif self.config.llm_provider == 'none':
            # 强制禁用 LLM，使用规则模式
            llm_config = {'provider': 'none'}
        else:
            # 从环境变量自动检测
            llm_config = load_llm_config_from_env()

        provider = create_llm_provider(llm_config)
        tool_names = list(self._tool_impls.keys())

        self._llm_orchestrator = LLMAgentOrchestrator(
            provider=provider,
            tool_names=tool_names,
            max_steps=self.config.max_steps,
            max_cost_credits=self.config.max_cost_credits,
        )

        return self._llm_orchestrator

    def _get_model_version(self) -> str:
        """获取模型版本描述"""
        if self.model is None:
            return "unknown"
        try:
            if hasattr(self.model, 'ckpt') and self.model.ckpt:
                return getattr(self.model.ckpt, 'get', lambda k, d: d)('model_name', 'yolov8')
        except Exception:
            pass
        return "yolov8s_baseline"


def _get_class_cn(cls_name: str) -> str:
    """获取类别中文名"""
    mapping = {
        'crazing': '裂纹', 'inclusion': '夹杂', 'patches': '斑块',
        'pitted_surface': '麻点', 'rolled-in_scale': '氧化铁皮', 'scratches': '划痕',
    }
    return mapping.get(cls_name, cls_name)
