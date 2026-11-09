# -*- coding: utf-8 -*-
"""
执行轨迹记录 —— Agent 执行过程的可观察性

记录内容：
- 任务ID
- 每一步：工具名、参数摘要、返回摘要、状态、错误、耗时
- 模型版本、规则版本
- 最终结论及依据范围
- 费用统计

注意：不记录模型内部思维链（也没有），只记录可观察的工具调用行为
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from .tool_schema import ToolStatus


@dataclass
class TraceStep:
    """单步执行记录"""
    step_index: int
    tool_name: str
    action_desc: str           # 简短行动说明（人类可读）
    params_summary: str        # 参数摘要（不含敏感信息）
    result_summary: str        # 返回摘要（不含大量数据）
    status: str                # success / error / timeout / validation_error
    error: str                 # 错误信息
    execution_time_ms: float
    model_used: str = ""       # 使用的模型（如果涉及LLM）
    cost_credits: int = 0      # 费用
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'step': self.step_index,
            'tool': self.tool_name,
            'action': self.action_desc,
            'params': self.params_summary,
            'result': self.result_summary,
            'status': self.status,
            'error': self.error,
            'time_ms': round(self.execution_time_ms, 2),
            'model': self.model_used,
            'cost': self.cost_credits,
            'timestamp': self.timestamp,
        }


class ExecutionTrace:
    """
    执行轨迹 —— 记录 Agent 的完整执行过程
    用于界面展示、调试和报告追溯
    """

    def __init__(self, task_id: str, model_version: str = "", rule_version: str = ""):
        self.task_id = task_id
        self.model_version = model_version
        self.rule_version = rule_version
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.steps: List[TraceStep] = []
        self.total_cost = 0
        self.max_steps_reached = False
        self.budget_exceeded = False
        self.llm_unavailable_fallback = False  # 是否因LLM不可用而回退到规则模式
        self.llm_fallback_reason = ''          # 降级原因
        self.llm_calls: List[Dict] = []       # LLM 调用记录
        self.llm_total_prompt_tokens = 0
        self.llm_total_completion_tokens = 0
        self.llm_call_count = 0
        self.final_conclusion: str = ""
        self.final_conclusion_scope: str = ""

    # ==================== 步骤记录 ====================

    def add_step(self, step: TraceStep):
        """添加执行步骤"""
        self.steps.append(step)
        self.total_cost += step.cost_credits

    def add_tool_step(self, step_index: int, tool_name: str, action_desc: str,
                      params_summary: str, result_summary: str, status: ToolStatus,
                      error: str = "", exec_ms: float = 0, model_used: str = "",
                      cost: int = 0):
        """便捷方法：添加工具调用步骤"""
        step = TraceStep(
            step_index=step_index,
            tool_name=tool_name,
            action_desc=action_desc,
            params_summary=params_summary,
            result_summary=result_summary,
            status=status.value,
            error=error,
            execution_time_ms=exec_ms,
            model_used=model_used,
            cost_credits=cost,
        )
        self.add_step(step)

    def mark_finished(self, conclusion: str, scope: str):
        """标记任务完成"""
        self.end_time = datetime.now().isoformat()
        self.final_conclusion = conclusion
        self.final_conclusion_scope = scope

    def add_llm_call(self, call_info: Dict):
        """记录一次 LLM 调用"""
        self.llm_calls.append(call_info)
        self.llm_call_count += 1
        self.llm_total_prompt_tokens += call_info.get('prompt_tokens', 0)
        self.llm_total_completion_tokens += call_info.get('completion_tokens', 0)

    # ==================== 查询 ====================

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_time_ms(self) -> float:
        return sum(s.execution_time_ms for s in self.steps)

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.steps if s.status == 'success')

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.steps if s.status in ('error', 'timeout', 'validation_error'))

    def has_tool(self, tool_name: str) -> bool:
        return any(s.tool_name == tool_name for s in self.steps)

    # ==================== 导出 ====================

    def to_dict(self) -> Dict:
        """导出为字典（用于报告和界面展示）"""
        return {
            'task_id': self.task_id,
            'model_version': self.model_version,
            'rule_version': self.rule_version,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_steps': self.step_count,
            'total_time_ms': round(self.total_time_ms, 2),
            'total_cost': self.total_cost,
            'success_steps': self.success_count,
            'error_steps': self.error_count,
            'max_steps_reached': self.max_steps_reached,
            'budget_exceeded': self.budget_exceeded,
            'llm_fallback': self.llm_unavailable_fallback,
            'llm_fallback_reason': self.llm_fallback_reason,
            'llm_call_count': self.llm_call_count,
            'llm_total_prompt_tokens': self.llm_total_prompt_tokens,
            'llm_total_completion_tokens': self.llm_total_completion_tokens,
            'llm_total_tokens': self.llm_total_prompt_tokens + self.llm_total_completion_tokens,
            'llm_calls': self.llm_calls,
            'final_conclusion': self.final_conclusion,
            'final_scope': self.final_conclusion_scope,
            'steps': [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        """导出为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary_text(self) -> str:
        """生成人类可读的摘要"""
        lines = []
        lines.append(f"任务 {self.task_id}")
        lines.append(f"总步骤: {self.step_count}，成功: {self.success_count}，失败: {self.error_count}")
        lines.append(f"总耗时: {self.total_time_ms:.1f}ms，总费用: {self.total_cost} 积分")
        if self.llm_unavailable_fallback:
            lines.append("⚠️ LLM 不可用，已回退到规则模式")
        if self.max_steps_reached:
            lines.append("⚠️ 已达到最大步骤限制")
        if self.budget_exceeded:
            lines.append("⚠️ 已超出费用预算")
        lines.append("")
        lines.append("执行轨迹:")
        for s in self.steps:
            status_icon = {'success': '✅', 'error': '❌', 'timeout': '⏱️',
                          'validation_error': '⚠️'}.get(s.status, '❓')
            lines.append(f"  {s.step_index}. {status_icon} {s.action_desc} "
                        f"[{s.tool_name}] ({s.execution_time_ms:.0f}ms)")
            if s.error:
                lines.append(f"     错误: {s.error}")
        if self.final_conclusion:
            lines.append("")
            lines.append(f"最终结论: {self.final_conclusion}")
            lines.append(f"结论范围: {self.final_conclusion_scope}")
        return "\n".join(lines)


# ==================== 持久化存储 ====================

class TraceStore:
    """执行轨迹的 SQLite 持久化存储"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS execution_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                model_version TEXT,
                rule_version TEXT,
                start_time TEXT,
                end_time TEXT,
                total_steps INTEGER,
                total_time_ms REAL,
                total_cost INTEGER,
                success_steps INTEGER,
                error_steps INTEGER,
                max_steps_reached INTEGER DEFAULT 0,
                budget_exceeded INTEGER DEFAULT 0,
                llm_fallback INTEGER DEFAULT 0,
                final_conclusion TEXT,
                final_scope TEXT,
                steps_json TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save(self, trace: ExecutionTrace):
        """保存执行轨迹"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        data = trace.to_dict()
        c.execute("""
            INSERT OR REPLACE INTO execution_traces
            (task_id, model_version, rule_version, start_time, end_time,
             total_steps, total_time_ms, total_cost, success_steps, error_steps,
             max_steps_reached, budget_exceeded, llm_fallback,
             final_conclusion, final_scope, steps_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['task_id'], data['model_version'], data['rule_version'],
            data['start_time'], data['end_time'],
            data['total_steps'], data['total_time_ms'], data['total_cost'],
            data['success_steps'], data['error_steps'],
            1 if data['max_steps_reached'] else 0,
            1 if data['budget_exceeded'] else 0,
            1 if data['llm_fallback'] else 0,
            data['final_conclusion'], data['final_scope'],
            json.dumps(data['steps'], ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    def get(self, task_id: str) -> Optional[Dict]:
        """查询执行轨迹"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM execution_traces WHERE task_id = ?", (task_id,))
        row = c.fetchone()
        conn.close()
        if row is None:
            return None
        d = dict(row)
        d['steps'] = json.loads(d['steps_json']) if d['steps_json'] else []
        del d['steps_json']
        return d

    def list_recent(self, limit: int = 20) -> List[Dict]:
        """列出最近的执行轨迹"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, task_id, start_time, total_steps, total_time_ms,
                   final_conclusion, final_scope, success_steps, error_steps
            FROM execution_traces
            ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows
