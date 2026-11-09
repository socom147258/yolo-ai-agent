# -*- coding: utf-8 -*-
"""
LLM 集成模块 - 支持结构化工具调用的大模型接入

设计原则：
1. 多 Provider 支持（openai / deepseek / qwen / moonshot 等 OpenAI 兼容接口）
2. 真实 LLM 不可用时自动降级为规则模式，不冒充自主 Agent
3. 所有调用记录完整（输入、输出、token、耗时、错误）
4. 工具使用严格白名单，LLM 只能调用注册的工具
5. 支持 mock 模式用于测试（不消耗 API 额度）

环境变量配置：
    LLM_PROVIDER=openai | deepseek | qwen | moonshot | mock | none
    LLM_API_KEY=sk-xxx
    LLM_BASE_URL=https://api.deepseek.com/v1  (可选，自定义端点)
    LLM_MODEL=deepseek-chat  (模型名)
"""
import os
import json
import time
import copy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from models.agent.tool_schema import TOOL_SCHEMAS, ToolSchema


# ============================================================
# 数据结构
# ============================================================

@dataclass
class LLMCallResult:
    """LLM 调用结果"""
    success: bool = False
    mode: str = 'none'  # 'llm' | 'rule' | 'mock'
    model: str = ''
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    response_text: str = ''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_ms: float = 0
    error: str = ''
    fallback_to_rule: bool = False
    fallback_reason: str = ''
    raw_request: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConversationTurn:
    """对话轮次记录"""
    role: str  # 'user' | 'assistant' | 'tool'
    content: str = ''
    tool_calls: List[Dict] = field(default_factory=list)
    tool_call_id: str = ''
    tool_name: str = ''


# ============================================================
# 工具 Schema 转换为 OpenAI Function Calling 格式
# ============================================================

def _convert_schema_to_openai_tool(tool_schema: ToolSchema) -> Dict[str, Any]:
    """将内部 ToolSchema 转换为 OpenAI function calling 格式"""
    properties = {}
    required = []

    for param in tool_schema.params:
        prop = {}

        # 类型映射
        type_map = {
            'str': 'string',
            'string': 'string',
            'int': 'integer',
            'integer': 'integer',
            'float': 'number',
            'number': 'number',
            'bool': 'boolean',
            'boolean': 'boolean',
            'list': 'array',
            'array': 'array',
            'dict': 'object',
            'object': 'object',
        }
        prop['type'] = type_map.get(param.type, 'string')

        # 描述
        if param.description:
            prop['description'] = param.description

        # 枚举
        if param.choices:
            prop['enum'] = param.choices

        # 默认值（放入描述）
        if param.default is not None and param.description:
            prop['description'] += f"（默认：{param.default}）"

        properties[param.name] = prop

        if param.required:
            required.append(param.name)

    return {
        'type': 'function',
        'function': {
            'name': tool_schema.name,
            'description': tool_schema.description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
            }
        }
    }


def build_tools_definition(tool_names: List[str]) -> List[Dict[str, Any]]:
    """根据工具名列表构建 OpenAI 格式的工具定义"""
    tools = []
    for name in tool_names:
        if name in TOOL_SCHEMAS:
            tools.append(_convert_schema_to_openai_tool(TOOL_SCHEMAS[name]))
    return tools


# ============================================================
# System Prompt 构建
# ============================================================

def build_system_prompt(context_info: Dict[str, Any], rule_info: Dict[str, Any]) -> str:
    """构建系统提示词"""

    prompt = f"""你是一个工业质检 Agent，负责协助质量检测工程师进行缺陷检测与质量判定。

## 你的能力
你可以调用以下工具来完成质检任务：
- 图像缺陷检测
- 缺陷数量统计
- 缺陷类别筛选
- 置信度分布分析
- 面积占比分析（检测框覆盖率，代理指标）
- 质量规则判定
- 规则查询
- 分析范围切换

## 重要约束
1. **分析范围**：当前分析范围为 {context_info.get('scope_label', '未知')}。
   - 如果用户要求"只看某类缺陷"，先调用筛选工具切换到子集范围
   - 如果用户要求"看全部/整体"，先切换回全量范围
   - 子集范围内的结论仅针对子集，不得代表整体质量

2. **质量判定安全**：
   - 没有检测数据时，判定为 REVIEW（证据不足）
   - 低质量图像会触发强制 REVIEW
   - 明确 FAIL 证据时不降级为 REVIEW
   - 当前使用的是演示规则（{rule_info.get('rule_id', 'unknown')} v{rule_info.get('version', 'unknown')}），仅供参考

3. **面积表述**：
   - 面积指标为"检测框覆盖率"，是代理指标，不是真实缺陷面积
   - 重叠检测框已做并集去重处理

4. **执行限制**：
   - 最多调用 {context_info.get('max_steps', 10)} 个工具
   - 每完成一个工具调用，根据结果决定下一步
   - 如果信息足够回答用户问题，用自然语言总结并结束

## 工作方式
用户提出质检任务后，你需要：
1. 理解用户意图
2. 选择合适的工具并调用
3. 查看工具返回结果
4. 决定是否继续调用其他工具或结束
5. 用中文给出清晰、准确的结论

请始终以质检工程师的专业、严谨态度回答问题。不确定时明确说明，不要编造数据。
"""
    return prompt


# ============================================================
# LLM Provider 实现
# ============================================================

class LLMProvider:
    """LLM 提供者基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = config.get('provider', 'none')
        self.model_name = config.get('model', '')
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', '')
        self.temperature = config.get('temperature', 0.1)
        self.timeout_s = config.get('timeout_s', 30)

    def is_available(self) -> bool:
        """检查是否可用"""
        return False

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> LLMCallResult:
        """带工具调用的对话（子类实现）"""
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容接口 Provider（支持 DeepSeek/Qwen/Moonshot 等）"""

    def is_available(self) -> bool:
        return bool(self.api_key) and bool(self.model_name)

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> LLMCallResult:
        result = LLMCallResult(mode='llm', model=self.model_name)
        start_time = time.time()

        try:
            from openai import OpenAI

            client_kwargs = {'api_key': self.api_key, 'timeout': self.timeout_s}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url

            client = OpenAI(**client_kwargs)

            # 记录请求
            result.raw_request = {
                'model': self.model_name,
                'messages': messages,
                'tools': tools,
                'temperature': self.temperature,
            }

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
            )

            elapsed = (time.time() - start_time) * 1000
            result.time_ms = elapsed
            result.success = True

            # 记录用量
            if response.usage:
                result.prompt_tokens = response.usage.prompt_tokens
                result.completion_tokens = response.usage.completion_tokens
                result.total_tokens = response.usage.total_tokens

            # 解析响应
            choice = response.choices[0]
            result.raw_response = json.loads(response.model_dump_json())['choices'][0]
            result.response_text = choice.message.content or ''

            # 解析工具调用
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {'_raw': tc.function.arguments}

                    result.tool_calls.append({
                        'id': tc.id,
                        'name': tc.function.name,
                        'arguments': args,
                    })

        except ImportError:
            result.error = 'openai 包未安装'
            result.fallback_to_rule = True
            result.fallback_reason = '缺少 openai 依赖包'
            result.time_ms = (time.time() - start_time) * 1000
        except Exception as e:
            result.error = str(e)
            result.fallback_to_rule = True
            result.fallback_reason = f'LLM 调用失败: {type(e).__name__}'
            result.time_ms = (time.time() - start_time) * 1000

        return result


class MockLLMProvider(LLMProvider):
    """
    Mock LLM Provider - 用于测试，不消耗 API 额度
    根据用户输入关键词模拟工具调用选择，验证链路完整性
    注意：这是 mock，不代表真实 LLM 能力
    """

    def is_available(self) -> bool:
        return True  # Mock 总是可用

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> LLMCallResult:
        result = LLMCallResult(mode='mock', model='mock-agent')
        start_time = time.time()

        # 统计已完成的工具调用次数（模拟 LLM 知道已做了什么）
        tool_call_count = 0
        last_tool_result = ''
        last_user_msg = ''

        for msg in messages:
            if msg['role'] == 'user':
                last_user_msg = msg.get('content', '')
            elif msg['role'] == 'tool':
                tool_call_count += 1
                last_tool_result = msg.get('content', '')

        user_input = last_user_msg.lower()

        # Mock 策略：
        # - 第一轮（没有工具结果时）：根据关键词选择工具
        # - 有工具结果后：如果已经做了质量判定或计数，就结束
        # - 最多模拟 2 个工具调用，然后给出最终回答

        if tool_call_count == 0:
            # 第一轮：选择第一个工具
            tool_name = None
            tool_args = {}

            # 优先级：检测 > 判定 > 统计 > 面积 > 置信度
            if any(w in user_input for w in ['检测', '看看有什么', '识别', '缺陷']) and 'image_path' in [
                t['function']['name'] for t in tools
            ]:
                # 但检测由 agent 框架自动完成，所以这里选下一个有意义的工具
                pass

            if any(w in user_input for w in ['合格', '质量', '判定', '评级']):
                tool_name = 'quality_judge'
            elif any(w in user_input for w in ['多少个', '数量', '计数', '统计']):
                tool_name = 'count_defects'
            elif any(w in user_input for w in ['面积', '占比', '覆盖率']):
                tool_name = 'area_analysis'
            elif any(w in user_input for w in ['置信度', '置信']):
                tool_name = 'confidence_distribution'
            elif any(w in user_input for w in ['规则', '阈值', '标准']):
                tool_name = 'rule_lookup'
            else:
                # 默认：做质量判定
                tool_name = 'quality_judge'

            if tool_name:
                result.tool_calls.append({
                    'id': 'mock_call_001',
                    'name': tool_name,
                    'arguments': tool_args,
                })
                result.response_text = f'好的，我先调用 {tool_name} 工具来分析。'
        elif tool_call_count == 1:
            # 第二轮：根据用户原始意图，可能再调用一个工具或结束
            # 如果用户问的是复杂问题，可能需要两步
            needs_quality = any(w in user_input for w in ['合格', '质量', '判定'])
            needs_count = any(w in user_input for w in ['多少个', '数量'])
            first_was_quality = '质量' in last_tool_result or 'PASS' in last_tool_result or 'FAIL' in last_tool_result or 'REVIEW' in last_tool_result

            if needs_quality and not first_was_quality:
                # 用户要质量判定但第一步没做，补做
                result.tool_calls.append({
                    'id': 'mock_call_002',
                    'name': 'quality_judge',
                    'arguments': {},
                })
                result.response_text = '接下来我调用质量判定工具。'
            else:
                # 已有足够信息，给出最终回答
                result.response_text = (
                    f'分析完成。根据检测结果，{last_tool_result[:100]} '
                    f'以上就是当前分析的结论。请问还需要其他分析吗？'
                )
        else:
            # 第三轮及以后：结束对话
            result.response_text = (
                f'分析已完成。如果还有其他问题，请随时告诉我。'
            )

        result.success = True
        result.prompt_tokens = 0
        result.completion_tokens = 0
        result.time_ms = (time.time() - start_time) * 1000

        return result


# ============================================================
# Provider 工厂
# ============================================================

def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    根据配置创建 LLM Provider

    支持的 provider:
    - openai: OpenAI 官方
    - deepseek: DeepSeek（OpenAI 兼容）
    - qwen: 通义千问（OpenAI 兼容）
    - moonshot: 月之暗面（OpenAI 兼容）
    - mock: Mock 测试用
    - none / '': 无 LLM，使用规则模式
    """
    provider = config.get('provider', 'none').lower()

    if provider in ('openai', 'deepseek', 'qwen', 'moonshot', 'custom'):
        return OpenAICompatibleProvider(config)
    elif provider == 'mock':
        return MockLLMProvider(config)
    else:
        return LLMProvider(config)  # 基类，is_available() 返回 False


def load_llm_config_from_env() -> Dict[str, Any]:
    """从环境变量加载 LLM 配置
    
    自动从项目根目录的 .env 文件加载配置（如果存在）。
    """
    # 尝试从 .env 文件加载（项目根目录）
    try:
        from dotenv import load_dotenv
        import pathlib
        # 向上查找项目根目录的 .env 文件
        project_root = pathlib.Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=False)  # 不覆盖已有的环境变量
    except ImportError:
        pass  # python-dotenv 未安装时跳过
    except Exception:
        pass  # 任何 .env 加载错误都不影响正常流程

    provider = os.environ.get('LLM_PROVIDER', os.environ.get('OPENAI_API_KEY') and 'openai' or 'none')

    # 自动检测 Provider
    if provider == 'none':
        if os.environ.get('DEEPSEEK_API_KEY'):
            provider = 'deepseek'
        elif os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'):
            provider = 'qwen'
        elif os.environ.get('MOONSHOT_API_KEY'):
            provider = 'moonshot'

    api_key = ''
    base_url = ''
    model = ''

    if provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY', '')
        model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        base_url = os.environ.get('OPENAI_BASE_URL', '')
    elif provider == 'deepseek':
        api_key = os.environ.get('DEEPSEEK_API_KEY', os.environ.get('LLM_API_KEY', ''))
        base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        model = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    elif provider == 'qwen':
        api_key = os.environ.get('DASHSCOPE_API_KEY', os.environ.get('QWEN_API_KEY', os.environ.get('LLM_API_KEY', '')))
        base_url = os.environ.get('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = os.environ.get('QWEN_MODEL', 'qwen-plus')
    elif provider == 'moonshot':
        api_key = os.environ.get('MOONSHOT_API_KEY', os.environ.get('LLM_API_KEY', ''))
        base_url = os.environ.get('MOONSHOT_BASE_URL', 'https://api.moonshot.cn/v1')
        model = os.environ.get('MOONSHOT_MODEL', 'moonshot-v1-8k')
    elif provider == 'mock':
        model = 'mock-agent'
    else:
        provider = 'none'

    # 通用覆盖
    if os.environ.get('LLM_API_KEY'):
        api_key = os.environ.get('LLM_API_KEY')
    if os.environ.get('LLM_BASE_URL'):
        base_url = os.environ.get('LLM_BASE_URL')
    if os.environ.get('LLM_MODEL'):
        model = os.environ.get('LLM_MODEL')

    return {
        'provider': provider,
        'api_key': api_key,
        'base_url': base_url,
        'model': model,
        'temperature': float(os.environ.get('LLM_TEMPERATURE', '0.1')),
        'timeout_s': int(os.environ.get('LLM_TIMEOUT', '30')),
    }


# ============================================================
# 多轮 Agent 执行器（LLM 驱动）
# ============================================================

class LLMAgentOrchestrator:
    """
    LLM Agent 编排器 - 管理多轮工具调用循环

    特点：
    - 真正的 LLM 工具调用，不是预编排脚本
    - LLM 根据工具返回结果动态选择下一步
    - 有界循环（最大步骤、超时、预算）
    - LLM 失败时自动降级到规则模式
    - 完整记录每一步的 LLM 请求/响应/Token/耗时
    """

    def __init__(self, provider: LLMProvider, tool_names: List[str],
                 max_steps: int = 10, max_cost_credits: float = 100):
        self.provider = provider
        self.tool_names = tool_names
        self.max_steps = max_steps
        self.max_cost_credits = max_cost_credits
        self.tools_def = build_tools_definition(tool_names)

    def is_llm_available(self) -> bool:
        """LLM 是否可用"""
        return self.provider.is_available()

    def plan_next_step(self, messages: List[Dict], context_info: Dict,
                       rule_info: Dict) -> LLMCallResult:
        """
        让 LLM 规划下一步（调用哪个工具，或结束）

        返回 LLMCallResult，包含 tool_calls 或结束响应
        """
        # 构建系统提示
        sys_prompt = build_system_prompt(context_info, rule_info)

        # 构建消息列表
        full_messages = [{'role': 'system', 'content': sys_prompt}] + messages

        # 调用 LLM
        result = self.provider.chat_with_tools(full_messages, self.tools_def)

        return result

    def validate_tool_call(self, tool_name: str, arguments: Dict) -> Tuple[bool, str]:
        """
        验证 LLM 的工具调用（白名单 + 参数校验）

        返回 (valid, error_message)
        """
        # 白名单检查
        if tool_name not in self.tool_names:
            return False, f"工具 '{tool_name}' 不在白名单中"

        # 参数校验
        if tool_name in TOOL_SCHEMAS:
            schema = TOOL_SCHEMAS[tool_name]
            is_valid, errors, _ = schema.validate_params(arguments)
            if not is_valid:
                return False, f"参数校验失败: {'; '.join(errors)}"

        return True, ''


# ============================================================
# 快速测试
# ============================================================

if __name__ == '__main__':
    config = load_llm_config_from_env()
    print(f"LLM 配置:")
    print(f"  Provider: {config['provider']}")
    print(f"  Model: {config['model']}")
    print(f"  Base URL: {config.get('base_url', 'default')}")
    print(f"  API Key: {'已配置' if config['api_key'] else '未配置'}")
    print()

    provider = create_llm_provider(config)
    print(f"Provider 类型: {type(provider).__name__}")
    print(f"是否可用: {provider.is_available()}")
    print()

    # 测试工具定义
    tools = build_tools_definition(['detect_image', 'quality_judge', 'count_defects'])
    print(f"工具定义数: {len(tools)}")
    for t in tools:
        print(f"  - {t['function']['name']}: {t['function']['description'][:40]}...")
    print()

    # Mock 模式测试
    if config['provider'] == 'mock' or not provider.is_available():
        print("使用 Mock Provider 进行测试:")
        mock = MockLLMProvider(config)
        result = mock.chat_with_tools(
            [{'role': 'user', 'content': '检测一下缺陷数量'}],
            build_tools_definition(['detect_image', 'count_defects', 'quality_judge'])
        )
        print(f"  成功: {result.success}")
        print(f"  工具调用: {[tc['name'] for tc in result.tool_calls]}")
        print(f"  响应: {result.response_text[:50]}")
        print(f"  耗时: {result.time_ms:.0f}ms")
