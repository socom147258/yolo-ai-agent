# -*- coding: utf-8 -*-
"""
LLM 集成测试
- Mock LLM 模式：验证工具调用循环
- 规则模式降级：验证无 LLM 时正常工作
- 模式切换：验证 LLM 配置检测
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent import (
    AgentExecutorV2,
    AgentConfig,
)
from models.agent.llm_integration import (
    load_llm_config_from_env,
    create_llm_provider,
    build_tools_definition,
    MockLLMProvider,
)


def test_llm_config_detection():
    """测试 LLM 配置自动检测"""
    print("=" * 60)
    print("测试1：LLM 配置自动检测")
    print("=" * 60)

    config = load_llm_config_from_env()
    print(f"  Provider: {config['provider']}")
    print(f"  Model: {config['model']}")
    print(f"  API Key: {'已配置' if config['api_key'] else '未配置'}")
    print(f"  Base URL: {config.get('base_url', 'default') or 'default'}")

    provider = create_llm_provider(config)
    print(f"  Provider 类型: {type(provider).__name__}")
    print(f"  是否可用: {provider.is_available()}")

    # Mock Provider 测试
    mock = MockLLMProvider({'provider': 'mock', 'model': 'mock-test'})
    print(f"  Mock 可用: {mock.is_available()}")

    print("✅ 配置检测测试通过\n")
    return True


def test_mock_llm_tool_calling():
    """测试 Mock LLM 的工具调用能力"""
    print("=" * 60)
    print("测试2：Mock LLM 工具调用循环")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        # 使用 mock 模式
        config = AgentConfig(
            max_steps=10,
            llm_provider='mock',
            llm_model='mock-agent',
        )
        agent = AgentExecutorV2(config, model=model, db_path=db_path)

        # 验证 LLM 可用（mock 总是可用）
        print(f"  _can_use_llm(): {agent._can_use_llm()}")
        orch = agent._get_llm_orchestrator()
        print(f"  Orchestrator 类型: {type(orch).__name__}")
        print(f"  Provider: {type(orch.provider).__name__}")
        print(f"  工具数: {len(orch.tool_names)}")

        # 执行任务
        img_path = r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\crazing_0001.jpg'
        result = agent.run_task(
            user_input='检测一下缺陷，质量合格吗',
            image_path=img_path,
        )

        print(f"\n  执行结果:")
        print(f"    模式: {result['mode']}")
        print(f"    任务ID: {result['task_id']}")
        print(f"    最终范围: {result['final_scope']}")
        print(f"    最终质量: {result.get('final_quality', 'N/A')}")

        trace = result['trace']
        print(f"    总步骤: {trace['total_steps']}")
        print(f"    LLM 调用次数: {trace.get('llm_call_count', 0)}")
        print(f"    LLM 模式: {trace.get('llm_calls', [{}])[0].get('mode', 'N/A') if trace.get('llm_calls') else 'N/A'}")

        print(f"\n  执行轨迹:")
        for step in trace['steps']:
            icon = '✅' if step['status'] == 'success' else '❌'
            print(f"    {step['step']}. {icon} {step['tool']} - {step['action'][:40]}")

        # 验证：Mock LLM 模式应该有工具调用
        assert result['mode'] == 'llm', f"模式应为 llm，实际为 {result['mode']}"
        assert trace['total_steps'] > 0, "应该有执行步骤"
        assert trace.get('llm_call_count', 0) > 0, "应该有 LLM 调用记录"

        print("\n✅ Mock LLM 工具调用测试通过\n")
        return True


def test_rule_fallback():
    """测试规则模式降级（none provider）"""
    print("=" * 60)
    print("测试3：规则模式降级")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        # 使用 none 模式（规则模式）
        config = AgentConfig(max_steps=10, llm_provider='none')
        agent = AgentExecutorV2(config, model=model, db_path=db_path)

        print(f"  _can_use_llm(): {agent._can_use_llm()}")

        img_path = r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\patches_0001.jpg'
        result = agent.run_task(
            user_input='检测一下缺陷数量',
            image_path=img_path,
        )

        print(f"  模式: {result['mode']}")
        print(f"  总步骤: {result['trace']['total_steps']}")
        print(f"  LLM 降级: {result['trace'].get('llm_fallback', False)}")

        # 验证：规则模式
        assert 'rule' in result['mode'], f"模式应包含 rule，实际为 {result['mode']}"

        print("✅ 规则模式降级测试通过\n")
        return True


def test_tools_definition():
    """测试工具定义转换"""
    print("=" * 60)
    print("测试4：工具 Schema → OpenAI 格式转换")
    print("=" * 60)

    tools = build_tools_definition([
        'detect_image', 'quality_judge', 'count_defects',
        'filter_by_class', 'area_analysis',
    ])

    print(f"  转换了 {len(tools)} 个工具定义:")
    for t in tools:
        func = t['function']
        params = func['parameters']
        req_count = len(params.get('required', []))
        prop_count = len(params.get('properties', {}))
        print(f"    • {func['name']}: {prop_count} 参数, {req_count} 必填")

    # 验证结构
    for t in tools:
        assert t['type'] == 'function'
        assert 'name' in t['function']
        assert 'description' in t['function']
        assert 'parameters' in t['function']
        assert t['function']['parameters']['type'] == 'object'

    print("\n✅ 工具定义转换测试通过\n")
    return True


def run_all():
    tests = [
        test_llm_config_detection,
        test_tools_definition,
        test_mock_llm_tool_calling,
        test_rule_fallback,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"💥 {test.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"LLM 集成测试: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
