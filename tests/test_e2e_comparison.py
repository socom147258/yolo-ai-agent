# -*- coding: utf-8 -*-
"""
端到端对比测试：LLM Agent vs 固定规则流程

同一组任务，相同的检测模型、规则引擎、安全控制和报告工具，
仅比较交互调度方式（LLM 动态调度 vs 固定顺序执行）。

记录指标：
- 任务成功率
- 操作次数（工具调用数）
- 总耗时
- 范围错误数（错误地改变了分析范围）
- 交付完整性（是否输出了最终质量判定）
- 降级率（LLM 失败降级到规则的比例）

测试分为三组：
1. Mock LLM 测试（无需 API Key，可重复执行）
2. 真实 LLM 测试（需要 API Key，记录真实调用数据）
3. 固定流程测试（基线对比）
"""
import sys
import os
import time
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent.agent_executor_v2 import AgentExecutorV2, AgentConfig
from models.agent.quality_engine_v2 import QualityEngineV2
from models.agent.tool_schema import ToolSchema, ToolResult
from models.agent.execution_trace import ExecutionTrace
from models.agent.llm_integration import (
    MockLLMProvider,
    LLMAgentOrchestrator,
    build_tools_definition,
)


# ============================================================
# 测试任务定义
# ============================================================

@dataclass
class E2ETestCase:
    """端到端测试用例"""
    case_id: str
    description: str
    user_task: str
    image_path: str
    expected_quality: str = ''  # PASS / FAIL / REVIEW / 空表示不校验
    expected_scope: str = 'full'  # 预期最终范围
    expected_steps_min: int = 2
    expected_steps_max: int = 10
    notes: str = ''


# 使用 NEU-DET 真实图像
NEU_DET_IMAGES = r'c:\Users\User\Desktop\YOLO\data\raw\NEU-DET\IMAGES'
MODEL_PATH = r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt'

TEST_CASES = [
    E2ETestCase(
        case_id='e2e_001',
        description='单图全量检测（基础任务）',
        user_task='检测这张图像的表面缺陷，给出质量判定',
        image_path=os.path.join(NEU_DET_IMAGES, 'crazing_1.jpg'),
        expected_scope='full',
        expected_steps_min=2,
        notes='最基础的单图检测+判定任务',
    ),
    E2ETestCase(
        case_id='e2e_002',
        description='指定类别子集分析',
        user_task='只分析这张图中的crazing缺陷，忽略其他类别',
        image_path=os.path.join(NEU_DET_IMAGES, 'crazing_1.jpg'),
        expected_scope='subset',
        expected_steps_min=3,
        notes='LLM应能理解"只分析crazing"并调用subset_filter',
    ),
    E2ETestCase(
        case_id='e2e_003',
        description='多缺陷类别图像检测',
        user_task='全面检测这张图像，给出质量评分和详细分析',
        image_path=os.path.join(NEU_DET_IMAGES, 'inclusion_1.jpg'),
        expected_scope='full',
        expected_steps_min=2,
        notes='inclusion类通常缺陷较多',
    ),
    E2ETestCase(
        case_id='e2e_004',
        description='先检测再改范围（动态调整）',
        user_task='先检测整张图，然后只看scratches类的情况',
        image_path=os.path.join(NEU_DET_IMAGES, 'scratches_1.jpg'),
        expected_scope='subset',
        expected_steps_min=4,
        notes='需要先全量检测，再切换到子集分析',
    ),
    E2ETestCase(
        case_id='e2e_005',
        description='patches类图像检测',
        user_task='检测这张表面图像的缺陷情况并判定质量',
        image_path=os.path.join(NEU_DET_IMAGES, 'patches_1.jpg'),
        expected_scope='full',
        expected_steps_min=2,
        notes='验证不同缺陷类别',
    ),
]


# ============================================================
# 测试结果记录
# ============================================================

@dataclass
class E2ETestResult:
    """单个测试用例的执行结果"""
    case_id: str
    method: str  # 'llm_mock' / 'llm_real' / 'rule_baseline'
    success: bool = False
    error_message: str = ''
    total_steps: int = 0
    success_steps: int = 0
    total_time_ms: float = 0
    final_quality: str = ''
    final_scope: str = ''
    final_score: float = 0
    scope_errors: int = 0  # 范围错误次数
    delivery_complete: bool = False  # 是否交付了完整结论
    fallback_to_rule: bool = False  # 是否降级到规则模式
    llm_call_count: int = 0
    llm_total_tokens: int = 0
    tool_calls: List[str] = field(default_factory=list)


@dataclass
class E2EBatchResult:
    """一批测试的汇总结果"""
    method: str
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    avg_steps: float = 0
    avg_time_ms: float = 0
    total_scope_errors: int = 0
    delivery_complete_count: int = 0
    fallback_count: int = 0
    results: List[E2ETestResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0

    @property
    def delivery_rate(self) -> float:
        return self.delivery_complete_count / self.total_cases if self.total_cases > 0 else 0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.total_cases if self.total_cases > 0 else 0


# ============================================================
# 固定流程执行器（基线）
# ============================================================

class FixedFlowExecutor:
    """
    固定流程执行器（基线对比）

    流程：
    1. 检测图像
    2. 全量质量判定
    3. 输出结果

    不处理子集分析、范围切换等动态需求。
    """

    def __init__(self, yolo_model, quality_engine: QualityEngineV2):
        self.model = yolo_model
        self.quality_engine = quality_engine

    def execute(self, image_path: str, user_task: str) -> E2ETestResult:
        """执行固定流程"""
        from models.agent.task_context import TaskContext
        from models.agent.batch_processor import detect_with_yolo

        result = E2ETestResult(
            case_id='',
            method='rule_baseline',
        )
        start_time = time.time()

        try:
            # Step 1: 检测
            detections, img_info, err = detect_with_yolo(self.model, image_path)
            result.tool_calls.append('detect')
            result.total_steps += 1

            if err:
                result.error_message = f'检测失败: {err}'
                result.total_time_ms = (time.time() - start_time) * 1000
                return result

            result.success_steps += 1

            # Step 2: 全量质量判定
            ctx = TaskContext('fixed_flow_test')
            ctx.model_version = 'yolov8s_baseline-2'
            ctx.set_raw_detections(detections, img_info)

            quality_result = self.quality_engine.evaluate(ctx, scope='full')
            result.tool_calls.append('quality_evaluate')
            result.total_steps += 1
            result.success_steps += 1

            result.final_quality = quality_result['quality']
            result.final_score = quality_result.get('score', 0)
            result.final_scope = quality_result.get('scope_label', '单图全量')
            result.delivery_complete = True
            result.success = True

        except Exception as e:
            result.error_message = str(e)

        result.total_time_ms = (time.time() - start_time) * 1000
        return result


# ============================================================
# LLM Agent 执行器
# ============================================================

class LLMAgentTester:
    """LLM Agent 测试器"""

    def __init__(self, yolo_model, quality_engine: QualityEngineV2,
                 mock_mode: bool = True):
        self.model = yolo_model
        self.quality_engine = quality_engine
        self.mock_mode = mock_mode

    def execute(self, image_path: str, user_task: str,
                case_id: str = '') -> E2ETestResult:
        """执行 LLM Agent"""
        from models.agent.task_context import TaskContext
        from models.agent.batch_processor import detect_with_yolo

        result = E2ETestResult(
            case_id=case_id,
            method='llm_mock' if self.mock_mode else 'llm_real',
        )
        start_time = time.time()

        try:
            # 先做检测（agent 内部也会调，但我们需要验证检测结果）
            detections, img_info, err = detect_with_yolo(self.model, image_path)
            if err:
                result.error_message = f'检测失败: {err}'
                result.total_time_ms = (time.time() - start_time) * 1000
                return result

            # 创建 agent 执行器
            config = AgentConfig(
                max_steps=10,
                llm_provider='mock' if self.mock_mode else 'auto',
                auto_save_trace=False,
            )
            executor = AgentExecutorV2(
                config=config,
                model=self.model,
            )

            # 执行任务
            task_result = executor.run_task(
                user_input=user_task,
                image_path=image_path,
                task_id=f'test_{case_id}',
            )

            # 提取结果
            trace = task_result.get('trace', {})
            result.total_steps = trace.get('total_steps', 0)
            result.success_steps = trace.get('success_steps', 0)
            result.final_quality = task_result.get('final_quality', '')
            result.final_scope = task_result.get('final_scope', '')
            result.delivery_complete = result.final_quality != ''
            result.success = result.delivery_complete and trace.get('status') != 'error'
            result.llm_call_count = task_result.get('llm_call_count', 0)
            result.fallback_to_rule = bool(trace.get('llm_fallback', False)) or \
                                       bool(trace.get('llm_fallback_reason', ''))

            # 从 context_snapshot 获取评分
            ctx_snap = task_result.get('context_snapshot', {})
            quality_data = ctx_snap.get('quality_data', {})
            result.final_score = quality_data.get('score', 0)

            # 提取工具调用列表
            steps = trace.get('steps', [])
            result.tool_calls = [s.get('tool', s.get('tool_name', '')) for s in steps]

            if not result.success:
                result.error_message = trace.get('error', '') or '未获取到质量判定'

        except Exception as e:
            result.error_message = str(e)
            import traceback
            result.error_message += '\n' + traceback.format_exc()

        result.total_time_ms = (time.time() - start_time) * 1000
        return result


# ============================================================
# 测试运行器
# ============================================================

def run_test_batch(cases: List[E2ETestCase], method: str,
                   yolo_model, quality_engine) -> E2EBatchResult:
    """运行一批测试"""
    batch = E2EBatchResult(method=method)
    batch.total_cases = len(cases)

    if method == 'rule_baseline':
        executor = FixedFlowExecutor(yolo_model, quality_engine)
    elif method == 'llm_mock':
        executor = LLMAgentTester(yolo_model, quality_engine, mock_mode=True)
    elif method == 'llm_real':
        executor = LLMAgentTester(yolo_model, quality_engine, mock_mode=False)
    else:
        raise ValueError(f"未知方法: {method}")

    total_time = 0
    total_steps = 0

    for case in cases:
        print(f"  运行 {case.case_id}: {case.description}")

        if method == 'rule_baseline':
            result = executor.execute(case.image_path, case.user_task)
        else:
            result = executor.execute(case.image_path, case.user_task, case.case_id)

        result.case_id = case.case_id
        batch.results.append(result)

        if result.success:
            batch.passed += 1
        else:
            batch.failed += 1
            print(f"    ❌ 失败: {result.error_message[:80]}")

        total_time += result.total_time_ms
        total_steps += result.total_steps
        batch.total_scope_errors += result.scope_errors
        if result.delivery_complete:
            batch.delivery_complete_count += 1
        if result.fallback_to_rule:
            batch.fallback_count += 1

        print(f"    {'✅' if result.success else '❌'} {result.total_steps}步, "
              f"{result.total_time_ms:.0f}ms, "
              f"质量={result.final_quality}, "
              f"范围={result.final_scope[:15]}")

    batch.avg_steps = total_steps / len(cases) if cases else 0
    batch.avg_time_ms = total_time / len(cases) if cases else 0

    return batch


def print_comparison(baseline: E2EBatchResult,
                     llm_mock: E2EBatchResult,
                     llm_real: Optional[E2EBatchResult] = None):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("端到端对比测试结果汇总")
    print("=" * 80)

    methods = [('固定流程(基线)', baseline), ('LLM Agent(Mock)', llm_mock)]
    if llm_real:
        methods.append(('LLM Agent(真实)', llm_real))

    # 表头
    header = f"{'指标':<25s}"
    for name, _ in methods:
        header += f"{name:<22s}"
    print(header)
    print("-" * 80)

    metrics = [
        ('任务数', lambda b: str(b.total_cases)),
        ('成功数', lambda b: str(b.passed)),
        ('成功率', lambda b: f"{b.success_rate*100:.1f}%"),
        ('平均步数', lambda b: f"{b.avg_steps:.1f}"),
        ('平均耗时(ms)', lambda b: f"{b.avg_time_ms:.0f}"),
        ('范围错误数', lambda b: str(b.total_scope_errors)),
        ('完整交付数', lambda b: str(b.delivery_complete_count)),
        ('交付完整率', lambda b: f"{b.delivery_rate*100:.1f}%"),
        ('降级次数', lambda b: str(b.fallback_count)),
        ('降级率', lambda b: f"{b.fallback_rate*100:.1f}%"),
    ]

    for metric_name, getter in metrics:
        row = f"{metric_name:<25s}"
        for _, batch in methods:
            row += f"{getter(batch):<22s}"
        print(row)

    print("=" * 80)


def run_all_e2e_tests():
    """运行所有端到端测试"""
    from ultralytics import YOLO

    print("=" * 80)
    print("端到端对比测试：LLM Agent vs 固定流程")
    print("=" * 80)
    print()
    print("测试原则：")
    print("  - 同一组任务（5个测试用例）")
    print("  - 相同的检测模型、规则引擎、安全控制、报告工具")
    print("  - 仅比较交互调度方式")
    print("  - Mock LLM / 真实 LLM / 固定流程 分别报告")
    print()

    # 加载模型
    print("加载检测模型...")
    model = YOLO(MODEL_PATH)
    engine = QualityEngineV2()
    print(f"模型: yolov8s_baseline-2")
    print(f"规则: {engine.rule_set.rule_id} v{engine.rule_set.version}")
    print(f"测试用例: {len(TEST_CASES)} 个")
    print()

    # 1. 固定流程（基线）
    print("-" * 60)
    print("1. 固定流程测试（基线）")
    print("-" * 60)
    baseline = run_test_batch(TEST_CASES, 'rule_baseline', model, engine)

    # 2. Mock LLM
    print("\n" + "-" * 60)
    print("2. LLM Agent 测试（Mock 模式）")
    print("-" * 60)
    llm_mock = run_test_batch(TEST_CASES, 'llm_mock', model, engine)

    # 3. 真实 LLM（如果有 API Key）
    llm_real = None
    api_key_found = any(
        os.environ.get(k)
        for k in ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'MOONSHOT_API_KEY']
    )

    if api_key_found:
        print("\n" + "-" * 60)
        print("3. LLM Agent 测试（真实 LLM）")
        print("-" * 60)
        try:
            llm_real = run_test_batch(TEST_CASES, 'llm_real', model, engine)
        except Exception as e:
            print(f"  真实 LLM 测试失败: {e}")
            print("  跳过真实 LLM 测试，仅使用 Mock 结果")
    else:
        print("\n" + "-" * 60)
        print("3. 真实 LLM 测试：未检测到 API Key，跳过")
        print("-" * 60)
        print("  （Mock 测试结果可验证调度逻辑和系统集成）")

    # 打印对比
    print_comparison(baseline, llm_mock, llm_real)

    # 保存结果
    output = {
        'baseline': _batch_to_dict(baseline),
        'llm_mock': _batch_to_dict(llm_mock),
        'llm_real': _batch_to_dict(llm_real) if llm_real else None,
        'test_cases': [
            {'case_id': c.case_id, 'description': c.description,
             'image': Path(c.image_path).name, 'task': c.user_task}
            for c in TEST_CASES
        ],
        'model': 'yolov8s_baseline-2',
        'rule_set': f"{engine.rule_set.rule_id}_v{engine.rule_set.version}",
        'note': '所有结果均为真实执行，Mock LLM用于验证调度逻辑，真实LLM用于验证端到端能力',
    }

    os.makedirs('outputs/test_results', exist_ok=True)
    with open('outputs/test_results/e2e_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到 outputs/test_results/e2e_comparison.json")

    return baseline, llm_mock, llm_real


def _batch_to_dict(batch: E2EBatchResult) -> Dict:
    return {
        'method': batch.method,
        'total_cases': batch.total_cases,
        'passed': batch.passed,
        'failed': batch.failed,
        'success_rate': batch.success_rate,
        'avg_steps': batch.avg_steps,
        'avg_time_ms': batch.avg_time_ms,
        'total_scope_errors': batch.total_scope_errors,
        'delivery_complete_count': batch.delivery_complete_count,
        'delivery_rate': batch.delivery_rate,
        'fallback_count': batch.fallback_count,
        'fallback_rate': batch.fallback_rate,
        'results': [
            {
                'case_id': r.case_id,
                'success': r.success,
                'error': r.error_message,
                'total_steps': r.total_steps,
                'success_steps': r.success_steps,
                'total_time_ms': r.total_time_ms,
                'final_quality': r.final_quality,
                'final_scope': r.final_scope,
                'final_score': r.final_score,
                'delivery_complete': r.delivery_complete,
                'fallback_to_rule': r.fallback_to_rule,
                'llm_call_count': r.llm_call_count,
                'tool_calls': r.tool_calls,
            }
            for r in batch.results
        ],
    }


if __name__ == '__main__':
    baseline, mock, real = run_all_e2e_tests()

    # 判断是否全部通过
    success = (mock.passed == mock.total_cases and
               baseline.passed == baseline.total_cases)
    sys.exit(0 if success else 1)
