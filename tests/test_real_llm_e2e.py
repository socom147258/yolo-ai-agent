# -*- coding: utf-8 -*-
"""
真实 LLM 端到端测试 - 使用 DeepSeek API 验证工具调用能力

运行前请确保已配置 .env 文件中的 DEEPSEEK_API_KEY
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent.llm_integration import (
    load_llm_config_from_env,
    create_llm_provider,
    LLMCallResult,
)


def test_real_llm_connection():
    """测试1：真实 LLM 连接验证"""
    print("=" * 60)
    print("测试1：真实 LLM 连接验证")
    print("=" * 60)

    config = load_llm_config_from_env()
    print(f"  Provider: {config['provider']}")
    print(f"  Model: {config['model']}")
    key = config.get('api_key', '')
    print(f"  API Key: {'已配置 (' + key[:8] + '...' + key[-4:] + ')' if key else '未配置'}")
    print(f"  Base URL: {config.get('base_url', '') or 'default'}")

    assert config['provider'] != 'none', "LLM Provider 未配置"
    assert config['provider'] != 'mock', "当前为 Mock 模式，请配置真实 API Key"
    assert config['api_key'], "API Key 为空"

    provider = create_llm_provider(config)
    assert provider.is_available(), "LLM Provider 不可用"

    print(f"  Provider 类型: {type(provider).__name__}")
    print("  ✅ 连接配置验证通过\n")
    return True, provider


def test_real_llm_simple_chat():
    """测试2：真实 LLM 简单对话"""
    print("=" * 60)
    print("测试2：真实 LLM 简单对话")
    print("=" * 60)

    config = load_llm_config_from_env()
    provider = create_llm_provider(config)

    messages = [
        {"role": "user", "content": "你好，请回复'测试通过'三个字。"}
    ]

    print("  发送: 你好，请回复'测试通过'三个字。")
    result = provider.chat_with_tools(messages, [])

    assert isinstance(result, LLMCallResult), "返回类型错误"
    assert result.response_text, "响应内容为空"
    assert not result.error, f"LLM 调用出错: {result.error}"

    print(f"  响应: {result.response_text}")
    print(f"  Token: prompt={result.prompt_tokens}, completion={result.completion_tokens}, total={result.total_tokens}")
    print(f"  耗时: {result.time_ms:.0f}ms")

    assert "测试通过" in result.response_text, f"响应中未包含'测试通过': {result.response_text}"

    print("  ✅ 简单对话测试通过\n")
    return True


def test_real_llm_tool_calling():
    """测试3：真实 LLM 工具调用能力"""
    print("=" * 60)
    print("测试3：真实 LLM 工具调用能力")
    print("=" * 60)

    config = load_llm_config_from_env()
    provider = create_llm_provider(config)

    # 定义一个简单的加法工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "计算两个数字的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "第一个数字"},
                        "b": {"type": "number", "description": "第二个数字"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "请计算 123 + 456 等于多少？使用 add_numbers 工具来计算。"}
    ]

    print("  发送: 请计算 123 + 456 等于多少？使用 add_numbers 工具来计算。")
    result = provider.chat_with_tools(messages, tools)

    assert isinstance(result, LLMCallResult), "返回类型错误"
    assert not result.error, f"LLM 调用出错: {result.error}"

    print(f"  响应类型: {'tool_calls' if result.tool_calls else 'text'}")
    print(f"  工具调用数: {len(result.tool_calls) if result.tool_calls else 0}")

    # LLM 应该调用工具
    assert result.tool_calls and len(result.tool_calls) > 0, f"LLM 未调用工具，直接回复: {result.content}"

    tool_call = result.tool_calls[0]
    print(f"  工具名: {tool_call['name']}")
    print(f"  参数: {tool_call['arguments']}")

    assert tool_call['name'] == 'add_numbers', f"工具名错误: {tool_call['name']}"

    # 解析参数
    args = tool_call['arguments']
    if isinstance(args, str):
        args = json.loads(args)
    assert args['a'] == 123, f"参数 a 错误: {args['a']}"
    assert args['b'] == 456, f"参数 b 错误: {args['b']}"

    # 模拟工具返回结果，继续对话
    tool_result = 123 + 456
    messages.append({
        "role": "assistant",
        "content": result.response_text,
        "tool_calls": [
            {
                "id": tool_call.get('id', 'call_1'),
                "type": "function",
                "function": {
                    "name": tool_call['name'],
                    "arguments": json.dumps(args) if isinstance(args, dict) else args,
                }
            }
        ]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.get('id', 'call_1'),
        "content": str(tool_result),
    })

    print(f"  工具返回: {tool_result}")
    print("  发送工具结果，等待最终回答...")

    result2 = provider.chat_with_tools(messages, tools)
    assert not result2.error, f"第二次 LLM 调用出错: {result2.error}"

    print(f"  最终回答: {result2.response_text}")
    assert "579" in result2.response_text, f"回答中未包含正确结果 579: {result2.response_text}"

    print("  ✅ 工具调用测试通过\n")
    return True


def test_real_llm_agent_tool_calling():
    """测试4：真实 LLM 调用 Agent 质检工具"""
    print("=" * 60)
    print("测试4：真实 LLM 调用 Agent 质检工具（Schema 验证）")
    print("=" * 60)

    from models.agent.llm_integration import build_tools_definition

    config = load_llm_config_from_env()
    provider = create_llm_provider(config)

    # 只取 detect_image 和 quality_judge 两个工具测试
    test_tool_names = ['detect_image', 'quality_judge']
    test_tools = build_tools_definition(test_tool_names)
    print(f"  测试工具数: {len(test_tools)}")
    for t in test_tools:
        print(f"    - {t['function']['name']}: {t['function']['description'][:50]}...")

    messages = [
        {"role": "system", "content": "你是一个质检助手，可以调用检测工具来分析图像缺陷。"},
        {"role": "user", "content": "请分析图像 'test.jpg' 的质量，先检测再判定。"}
    ]

    result = provider.chat_with_tools(messages, test_tools)
    assert not result.error, f"LLM 调用出错: {result.error}"

    print(f"  响应类型: {'tool_calls' if result.tool_calls else 'text'}")

    # LLM 应该调用 detect_image 工具
    if result.tool_calls:
        print(f"  工具调用数: {len(result.tool_calls)}")
        for tc in result.tool_calls:
            print(f"    - {tc['name']}: {str(tc['arguments'])[:80]}")

        # 验证至少调用了一个有效工具
        valid_tools = set(test_tool_names)
        called_tools = {tc['name'] for tc in result.tool_calls}
        assert called_tools.issubset(valid_tools), f"调用了未注册的工具: {called_tools - valid_tools}"
        print("  ✅ 工具调用在白名单内")
    else:
        print(f"  文本回复: {result.content[:100]}")
        print("  ⚠️  LLM 未调用工具（可能直接回复了文本）")

    print(f"  Token: prompt={result.prompt_tokens}, completion={result.completion_tokens}, total={result.total_tokens}")
    print("  ✅ Agent 工具 Schema 测试通过\n")
    return True


def main():
    tests = [
        test_real_llm_connection,
        test_real_llm_simple_chat,
        test_real_llm_tool_calling,
        test_real_llm_agent_tool_calling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} 失败: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"真实LLM端到端测试: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 60)

    if failed == 0:
        print("🎉 真实 LLM 端到端测试全部通过！")
    else:
        print(f"⚠️  有 {failed} 项测试失败")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
