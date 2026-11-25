# -*- coding: utf-8 -*-
"""
Agent v2 页面组件 —— 升级后的质检 Agent 界面

新增功能：
- LLM 配置选项（Mock / DeepSeek / OpenAI / 自动 / 规则模式）
- 支持范围切换（全量 / 按类别筛选）
- 执行追踪展示（每一步工具调用、耗时、状态）
- 质量判定结果 + 免责声明
- 一键生成 Word 报告（v2 版 8 大章节）
- 自动创建复核任务（FAIL/REVIEW 时）
- 检测结果可视化
"""

import streamlit as st
import os
import json
import io
import tempfile
from html import escape
from pathlib import Path
from shutil import copyfile
from PIL import Image, ImageDraw
from datetime import datetime
from typing import Dict, List, Optional

from models.agent import (
    AgentExecutorV2,
    AgentConfig,
    ReviewWorkflow,
    QualityEngineV2,
)


# 类别中文名映射
CLASS_NAMES_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化皮',
    'scratches': '划痕',
}


def render_agent_v2_page(agent_v2: AgentExecutorV2, db_path: str,
                         review_workflow: ReviewWorkflow):
    """渲染 Agent v2 页面（升级增强版）"""

    st.markdown('<div class="main-header">智检 Agent · 钢材表面质检</div>', unsafe_allow_html=True)

    st.caption("上传钢材图像，交给 Agent 检测、判断并生成质检报告。每一步可在执行轨迹查看。")

    st.button("一键演示质检", type="primary", on_click=_start_demo,
              help="检测两张钢材样例、一张模拟低质量图和一个损坏文件。")

    # 顶部状态栏
    _render_status_bar(agent_v2)

    st.markdown("---")

    # LLM 配置面板（折叠）
    _render_llm_config_panel()

    # 范围切换面板
    _render_scope_panel(agent_v2)

    st.markdown("---")

    # 三栏布局：左侧上传/输入，中间对话，右侧执行轨迹
    col_input, col_chat, col_trace = st.columns([2, 4, 3])

    with col_input:
        _render_input_panel(agent_v2)

    with col_chat:
        _render_chat_panel(agent_v2, review_workflow)

    with col_trace:
        _render_trace_panel()

    # 底部：质量判定结果 + 报告生成 + 复核
    _render_result_actions(agent_v2, review_workflow, db_path)

    # 底部安全提示
    _render_safety_footer()


def _render_status_bar(agent: AgentExecutorV2):
    """顶部状态栏"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        provider = agent.config.llm_provider
        if provider == 'none':
            mode_label = "规则模式"
            mode_help = "规则模式：关键词匹配+固定流程"
        elif provider == 'mock':
            mode_label = "Mock 模式"
            mode_help = "Mock 模式：模拟 LLM 行为，不消耗 API 额度"
        else:
            mode_label = f"LLM: {provider}"
            mode_help = f"LLM 模式：{provider} 驱动的智能工具调用"
        st.metric("执行模式", mode_label, help=mode_help)

    with col2:
        st.metric("最大步骤", agent.config.max_steps,
                  help="Agent 执行的最大工具调用次数，防止无限循环")

    with col3:
        rule_info = agent.quality_engine.rule_set
        badge = "⚠️ 演示规则" if rule_info.is_demo else "✅ 正式规则"
        st.metric("规则版本", f"{rule_info.rule_id} v{rule_info.version}",
                  delta=badge, delta_color="off")

    with col4:
        st.metric("工具数量", len(agent.get_available_tools()),
                  help="Agent 可调用的工具白名单")


def _render_llm_config_panel():
    """LLM 配置面板（折叠式）"""
    with st.expander("⚙️ LLM 配置", expanded=False):
        st.caption("选择 LLM 提供方，影响 Agent 的智能规划能力")

        llm_options = {
            "auto": "🤖 自动检测（从 .env / 环境变量）",
            "mock": "🎭 Mock 模式（模拟，无 API 消耗）",
            "deepseek": "🔵 DeepSeek",
            "openai": "🟢 OpenAI",
            "none": "⚙️ 纯规则模式",
        }

        current = st.session_state.get('llm_provider', 'auto')
        selected = st.selectbox(
            "LLM 提供方",
            options=list(llm_options.keys()),
            format_func=lambda k: llm_options[k],
            index=list(llm_options.keys()).index(current) if current in llm_options else 0,
            key="llm_provider_select",
        )

        if selected != st.session_state.get('llm_provider', 'auto'):
            st.session_state['llm_provider'] = selected
            # 清除 Agent 缓存以便重新加载
            if 'load_agent_v2' in st.session_state.get('_cache_resource_keys', set()):
                try:
                    from streamlit.runtime.caching import cache_resource
                    cache_resource.clear()
                except Exception:
                    pass
            st.success(f"已切换到 {llm_options[selected]}，刷新后生效")
            st.info("💡 点击侧边栏其他模块再切回来，或刷新页面以应用新配置")

        st.caption("配置说明：")
        st.caption("• 自动检测：从项目根目录 .env 文件读取 LLM_PROVIDER / API_KEY 等配置")
        st.caption("• Mock 模式：模拟 LLM 响应，用于界面测试和演示")
        st.caption("• DeepSeek / OpenAI：需要对应 API Key，从环境变量读取")
        st.caption("• 纯规则模式：关键词匹配 + 固定流程，无需网络")


def _render_scope_panel(agent: AgentExecutorV2):
    """范围切换面板"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        scope = st.session_state.get('v2_scope', 'full')
        scope_label = "📊 全量分析" if scope == 'full' else "🔍 子集分析"
        st.markdown(f"**当前范围：** {scope_label}")

    with col2:
        # 类别筛选
        class_options = ["全量"] + list(CLASS_NAMES_CN.keys())
        class_labels = ["全量（所有缺陷类别）"] + [
            f"{k} ({v})" for k, v in CLASS_NAMES_CN.items()
        ]
        current_class = st.session_state.get('v2_filter_class', '全量')
        idx = class_options.index(current_class) if current_class in class_options else 0

        selected_class = st.selectbox(
            "按类别筛选",
            options=class_options,
            format_func=lambda x: class_labels[class_options.index(x)],
            index=idx,
            key="v2_class_filter",
        )

        if selected_class != current_class:
            st.session_state['v2_filter_class'] = selected_class
            if selected_class == '全量':
                st.session_state['v2_scope'] = 'full'
            else:
                st.session_state['v2_scope'] = 'subset'
                st.session_state['v2_subset_class'] = selected_class

    with col3:
        st.caption("")
        if st.button("🔄 切换范围", use_container_width=True, key="v2_switch_scope"):
            target_class = st.session_state.get('v2_filter_class', '全量')
            if target_class != '全量':
                # 触发子集切换指令
                st.session_state['v2_pending_cmd'] = f"只看{CLASS_NAMES_CN.get(target_class, target_class)}缺陷"
            else:
                st.session_state['v2_pending_cmd'] = "切回全量再判定"
            st.rerun()


def _start_demo():
    directory = Path(tempfile.mkdtemp(prefix='steel_demo_'))
    samples = Path(__file__).resolve().parents[1] / 'sample_images'
    for name in ('crazing_0001.jpg', 'scratches_0001.jpg'):
        copyfile(samples / name, directory / name)
    Image.new('RGB', (200, 200), (128, 128, 128)).save(directory / '模拟低质量图.jpg')
    (directory / '模拟损坏文件.jpg').write_bytes(b'Invalid image demo')
    st.session_state['v2_demo_folder'] = str(directory)
    st.session_state['v2_input_mode'] = '文件夹批量'
    st.session_state['v2_batch_input_type'] = '答辩演示'
    _set_inspection_input(folder_path=str(directory))
    st.session_state['v2_pending_cmd'] = '批量检测，列出待复核图片并说明判定原因，生成批次报告'


def _set_inspection_input(image_path=None, folder_path=None):
    """仅在检测对象发生变化时清理已有证据。"""
    state = st.session_state
    if (state.get('v2_image_path'), state.get('v2_folder_path')) == (image_path, folder_path):
        return
    for key in ('v2_image_evidence', 'v2_batch_results', 'v2_batch_by_class',
                'v2_batch_total_detections', 'v2_last_result', 'v2_last_trace',
                'v2_context', 'v2_last_review_id', 'v2_report_requested', 'v2_report_download'):
        state.pop(key, None)
    state['v2_image_path'] = image_path
    state['v2_folder_path'] = folder_path


def _render_input_panel(agent: AgentExecutorV2):
    """左侧：上传和输入面板"""
    st.subheader("📁 输入")

    # 输入模式切换
    input_mode = st.radio(
        "输入方式",
        options=["单图上传", "文件夹批量"],
        index=0 if st.session_state.get('input_mode', 'single') == 'single' else 1,
        key="v2_input_mode",
        horizontal=True,
    )
    st.session_state['input_mode'] = 'single' if input_mode == "单图上传" else 'batch'

    if input_mode == "单图上传":
        # 上传图片
        uploaded = st.file_uploader(
            "上传检测图像",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="v2_upload"
        )

        if uploaded is not None:
            if st.session_state.get('v2_single_upload_id') != uploaded.file_id:
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
                    image_file.write(uploaded.getbuffer())
                st.session_state['v2_single_upload_id'] = uploaded.file_id
                st.session_state['v2_uploaded_image_path'] = image_file.name
            _set_inspection_input(image_path=st.session_state['v2_uploaded_image_path'])
            st.session_state['v2_image_name'] = Path(uploaded.name).name

            st.image(uploaded, caption=uploaded.name, use_column_width=True)
            st.success("✅ 图像已上传")
        else:
            _set_inspection_input()
            st.session_state.pop('v2_single_upload_id', None)

    else:
        # 批量模式：三种输入方式
        batch_modes = ["📤 上传多张图片", "🎯 使用内置示例图片"]
        if st.session_state.get('v2_demo_folder'):
            batch_modes.append('答辩演示')
        if os.environ.get('ALLOW_LOCAL_FOLDER') == '1':
            batch_modes.append("📂 本地文件夹路径")
        batch_input_type = st.radio(
            "批量输入方式",
            options=batch_modes,
            index=0,
            key="v2_batch_input_type",
            horizontal=True,
        )

        if batch_input_type == '答辩演示':
            _set_inspection_input(folder_path=st.session_state['v2_demo_folder'])
            st.caption("4 个输入，包括两张样例、一张模拟低质量图和一个损坏文件。")
        elif batch_input_type == "📤 上传多张图片":
            uploaded_files = st.file_uploader(
                "上传多张图片（按住 Ctrl 多选）",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                key="v2_batch_upload",
                help="云端部署时推荐使用此方式，可直接上传图片进行批量检测"
            )

            if uploaded_files and len(uploaded_files) > 0:
                upload_ids = tuple(uf.file_id for uf in uploaded_files)
                if st.session_state.get('v2_batch_upload_ids') != upload_ids:
                    directory = Path(tempfile.mkdtemp(prefix='steel_batch_'))
                    for index, upload in enumerate(uploaded_files):
                        filename = Path(upload.name.replace(chr(92), '/')).name
                        (directory / f'{index + 1:03}_{filename}').write_bytes(upload.getbuffer())
                    st.session_state['v2_batch_upload_ids'] = upload_ids
                    st.session_state['v2_uploaded_batch_path'] = str(directory)
                _set_inspection_input(folder_path=st.session_state['v2_uploaded_batch_path'])
                saved_count = len(uploaded_files)

                if saved_count <= 6:
                    cols = st.columns(min(saved_count, 3))
                    for i, uf in enumerate(uploaded_files[:6]):
                        cols[i % 3].image(uf, caption=uf.name, width=100)
                else:
                    st.caption(f"共 {saved_count} 张图片，在对话框输入「批量检测」开始检测")

            else:
                _set_inspection_input()
                st.session_state.pop('v2_batch_upload_ids', None)
                st.info("上传多张图片后即可发起批量检测。")

        elif batch_input_type == "🎯 使用内置示例图片":
            sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_images")
            if os.path.isdir(sample_dir):
                exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
                sample_files = [f for f in os.listdir(sample_dir) if os.path.splitext(f)[1] in exts]
                count = len(sample_files)

                # 按类型统计
                types = {}
                for f in sample_files:
                    prefix = f.split('_')[0]
                    types[prefix] = types.get(prefix, 0) + 1

                st.info(f"📊 内置示例数据集：共 **{count} 张** 图片，{len(types)} 类缺陷")

                # 显示各类数量
                type_cols = st.columns(3)
                type_names = {
                    "crazing": "裂纹", "inclusion": "夹杂物", "patches": "斑块",
                    "pitted": "麻点", "rolled-in": "氧化皮", "scratches": "划痕",
                }
                for i, (t, c) in enumerate(sorted(types.items())):
                    name = type_names.get(t, t)
                    type_cols[i % 3].metric(name, f"{c} 张")

                if st.button("🚀 使用此数据集开始检测", use_container_width=True, type="primary"):
                    _set_inspection_input(folder_path=sample_dir)
                    st.session_state['v2_pending_cmd'] = '批量检测'
                    st.rerun()

                if st.session_state.get('v2_folder_path') == sample_dir:
                    st.success("✅ 当前已选择内置示例数据集")
            else:
                st.warning("⚠️ 内置示例图片目录不存在")

        else:
            # 本地文件夹路径（仅本地部署时有效）
            st.warning("⚠️ 此方式仅在本地部署时可用，云端部署无法读取本地文件夹")
            folder_path = st.text_input(
                "📂 本地文件夹路径",
                value=st.session_state.get('v2_folder_path') or '',
                placeholder="例如：C:\\Users\\xxx\\data\\test_images",
                key="v2_folder_input",
                help="仅本地部署可用，云端部署请使用「上传多张图片」或「内置示例图片」"
            )

            if folder_path.strip():
                _set_inspection_input(folder_path=folder_path.strip())
                # 验证文件夹是否存在
                if os.path.isdir(folder_path.strip()):
                    # 统计图片数量
                    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
                    count = sum(1 for f in os.listdir(folder_path.strip())
                               if os.path.splitext(f)[1] in exts)
                    st.success(f"✅ 文件夹有效，找到 {count} 张图片")
                    st.caption(f"路径：{folder_path.strip()}")
                else:
                    st.error("❌ 文件夹不存在，请检查路径是否正确")
            else:
                _set_inspection_input()
                st.info("💡 输入文件夹路径后，在对话框输入「批量检测」即可开始")

    st.markdown("---")

    # 快捷指令
    st.caption("💡 快捷指令")
    if st.session_state.get('input_mode', 'single') == 'single':
        quick_cmds = [
            "检测一下有什么缺陷",
            "裂纹有多少个",
            "质量合格吗",
            "看看面积占比",
            "置信度分布如何",
            "当前规则是什么",
            "切回全量再判定",
            "生成检测报告",
        ]
    else:
        quick_cmds = [
            "只看裂纹，再判断整批",
            "批量检测文件夹",
            "统计合格率",
            "哪些不合格",
            "各类缺陷分布",
            "生成批次报告",
        ]

    for cmd in quick_cmds:
        if st.button(cmd, key=f"quick_{cmd}", use_container_width=True):
            st.session_state['v2_pending_cmd'] = cmd
            st.rerun()

    st.markdown("---")

    # 配置选项
    with st.expander("⚙️ 执行配置"):
        st.number_input("最大步骤数", min_value=1, max_value=30,
                       value=st.session_state.get('v2_max_steps', 10),
                       key='v2_max_steps')
        st.text_input("规则集 ID", value="demo_default",
                     help="当前使用的规则集", disabled=True)
        st.caption("修改规则请前往「⚙️ 质量规则」页面")


def _render_chat_panel(agent: AgentExecutorV2, review_workflow: ReviewWorkflow):
    """中间：对话面板"""
    st.subheader("💬 质检对话")

    # 初始化对话历史
    if 'v2_chat_history' not in st.session_state:
        st.session_state['v2_chat_history'] = [
            {
                'role': 'agent',
                'content': '您好！我是质检 Agent v2 👋\n\n'
                           '我可以帮您做缺陷检测、质量判定、统计分析等。\n'
                           '每次操作都会记录执行轨迹，确保过程可追溯。',
            }
        ]

    # 显示对话
    for msg_idx, msg in enumerate(st.session_state['v2_chat_history']):
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-user">👤 {escape(msg["content"])}</div>', unsafe_allow_html=True)
        else:
            content = msg.get('content', '')
            st.markdown(f'<div class="chat-agent">🤖 {escape(content).replace(chr(10), "<br>")}</div>',
                       unsafe_allow_html=True)

            # 显示范围标签
            scope = msg.get('scope_label', '')
            if scope:
                scope_badge = '🔍 子集' if '子集' in scope else '📊 全量'
                st.caption(f"{scope_badge} 分析范围：{scope}")

            # 显示质量标签
            quality = msg.get('quality', '')
            if quality:
                badges = {
                    'PASS': ('✅ PASS 合格', 'quality-pass'),
                    'REVIEW': ('⚠️ REVIEW 待复核', 'quality-review'),
                    'FAIL': ('❌ FAIL 不合格', 'quality-fail'),
                    'INCOMPLETE': ('INCOMPLETE 未完成', 'quality-pending'),
                }
                label, cls = badges.get(quality, ('❓ 未知', 'quality-pending'))
                st.markdown(
                    f'<span class="quality-badge {cls}">{label}</span>',
                    unsafe_allow_html=True
                )

    # 输入框
    user_input = st.text_input("输入质检指令：", key="v2_input",
                              placeholder="例如：检测一下缺陷，看看合格吗")

    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send_clicked = st.button("发送", key="v2_send", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("清空", key="v2_clear", use_container_width=True)

    # 处理发送
    pending_cmd = st.session_state.get('v2_pending_cmd', '')
    if send_clicked and user_input.strip():
        _handle_user_input(agent, user_input.strip(), review_workflow)
    elif pending_cmd:
        st.session_state['v2_pending_cmd'] = ''
        _handle_user_input(agent, pending_cmd, review_workflow)

    if clear_clicked:
        st.session_state['v2_chat_history'] = [
            {'role': 'agent', 'content': '对话已清空。您好！我是质检 Agent v2 👋'}
        ]
        if 'v2_last_trace' in st.session_state:
            del st.session_state['v2_last_trace']
        if 'v2_last_result' in st.session_state:
            del st.session_state['v2_last_result']
        if 'v2_batch_results' in st.session_state:
            del st.session_state['v2_batch_results']
        if 'v2_batch_by_class' in st.session_state:
            del st.session_state['v2_batch_by_class']
        if 'v2_batch_total_detections' in st.session_state:
            del st.session_state['v2_batch_total_detections']
        st.session_state['v2_scope'] = 'full'
        st.session_state['v2_filter_class'] = '全量'
        st.rerun()


def _handle_user_input(agent: AgentExecutorV2, user_input: str,
                       review_workflow: ReviewWorkflow):
    """处理用户输入"""
    # 添加用户消息
    st.session_state['v2_chat_history'].append({
        'role': 'user',
        'content': user_input
    })

    # 执行 Agent 任务
    image_path = st.session_state.get('v2_image_path')
    folder_path = st.session_state.get('v2_folder_path')
    # 恢复已有批量检测结果（避免重复检测）
    prev_batch_results = st.session_state.get('v2_batch_results')
    prev_batch_by_class = st.session_state.get('v2_batch_by_class')
    prev_batch_total = st.session_state.get('v2_batch_total_detections')

    with st.spinner('Agent 执行中...'):
        try:
            result = agent.run_task(
                user_input=user_input,
                image_path=image_path,
                folder_path=folder_path,
                batch_results=prev_batch_results,
                batch_by_class=prev_batch_by_class,
                batch_total_detections=prev_batch_total,
                image_evidence=st.session_state.get('v2_image_evidence'),
            )
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            st.error(f"执行出错：{str(e)}")
            with st.expander("🔍 详细错误信息"):
                st.code(error_detail)
            st.session_state['v2_chat_history'].append({
                'role': 'agent',
                'content': f'❌ 执行出错：{str(e)}',
            })
            st.rerun()
            return

    if result.get('image_evidence'):
        st.session_state['v2_image_evidence'] = result['image_evidence']

    # 保存轨迹和结果
    st.session_state['v2_last_trace'] = result['trace']
    st.session_state['v2_context'] = result['context_snapshot']
    st.session_state['v2_last_result'] = result
    st.session_state.pop('v2_report_download', None)
    st.session_state['v2_report_requested'] = any(
        keyword in user_input for keyword in ('报告', '报表', '导出')
    )

    # 保存批量检测结果（用于多轮对话复用）
    if 'batch_results' in result and result['batch_results']:
        st.session_state['v2_batch_results'] = result['batch_results']
        if 'batch_by_class' in result:
            st.session_state['v2_batch_by_class'] = result['batch_by_class']
        if 'batch_total_detections' in result:
            st.session_state['v2_batch_total_detections'] = result['batch_total_detections']

    # 生成回复
    trace = result['trace']
    steps = trace.get('steps', [])

    # 提取质量判定
    quality = result.get('final_quality', '')
    for step in reversed(steps):
        if step.get('tool_name') == 'quality_judge' and step.get('status') == 'success':
            result_text = step.get('result_summary', '')
            if 'FAIL' in result_text:
                quality = 'FAIL'
            elif 'REVIEW' in result_text:
                quality = 'REVIEW'
            elif 'PASS' in result_text:
                quality = 'PASS'
            break

    # 回复内容
    final = result.get('final_conclusion', '执行完成')
    scope_label = result.get('final_scope', '')

    # 补充方法备注
    method_notes = ""
    ctx = result.get('context_snapshot', {})
    if ctx.get('is_subset'):
        method_notes = "⚠️ 注意：当前结论基于子集范围，不代表整体质量"

    response_text = final
    if method_notes:
        response_text += "\n\n" + method_notes

    # 自动创建复核任务（FAIL / REVIEW 时）
    review_id = st.session_state.get('v2_last_review_id')
    if quality in ('REVIEW', 'FAIL') and image_path and not review_id:
        try:
            # 提取检测结果和评估结果
            detections = ctx.get('current_detections', [])
            eval_result = result.get('quality_evaluation') or _extract_eval_result(steps)

            review_id = review_workflow.create_task(
                task_id=result.get('task_id', ''),
                record_id=0,  # 临时 ID
                image_name=st.session_state.get('v2_image_name', 'unknown'),
                image_path=image_path,
                original_quality=quality,
                original_score=eval_result.get('score', 0),
                original_scope=scope_label,
                reason=eval_result.get('explanation', '')[:200],
                quality_issue=quality,
                priority='high' if quality == 'FAIL' else 'normal',
                model_version=ctx.get('model_version', ''),
                rule_version=ctx.get('rule_version', ''),
                original_detections=detections,
                original_eval=eval_result,
            )
            st.session_state['v2_last_review_id'] = review_id
        except Exception as e:
            st.session_state['v2_last_review_error'] = str(e)

    st.session_state['v2_chat_history'].append({
        'role': 'agent',
        'content': response_text,
        'quality': quality,
        'scope_label': scope_label,
        'trace_id': trace.get('task_id', ''),
        'review_id': review_id,
    })

    st.rerun()


def _extract_eval_result(steps):
    """从执行步骤中提取质量评估结果"""
    for step in reversed(steps):
        if step.get('tool_name') == 'quality_judge' and step.get('status') == 'success':
            # 尝试从 result_summary 解析
            summary = step.get('result_summary', '')
            import re
            score_match = re.search(r'(\d+)分', summary)
            score = int(score_match.group(1)) if score_match else 0

            quality = ''
            if 'FAIL' in summary:
                quality = 'FAIL'
            elif 'REVIEW' in summary:
                quality = 'REVIEW'
            elif 'PASS' in summary:
                quality = 'PASS'

            return {
                'quality': quality,
                'score': score,
                'explanation': summary,
            }
    return {'quality': '', 'score': 0, 'explanation': ''}


def _render_trace_panel():
    """右侧：执行轨迹面板"""
    st.subheader("📋 执行轨迹")

    trace = st.session_state.get('v2_last_trace')

    if not trace:
        st.info("暂无执行记录\n\n发送指令后，这里会显示每一步的工具调用过程。")
        return

    # 轨迹概览
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总步骤", trace.get('total_steps', 0))
    with col2:
        st.metric("总耗时", f"{trace.get('total_time_ms', 0):.0f}ms")

    _render_detection_evidence(result)

    # 执行模式
    mode = trace.get('mode', 'unknown')
    mode_map = {
        'llm': ('🟢 LLM 模式', 'success'),
        'rule_based': ('⚙️ 规则模式', 'info'),
        'llm_fallback_to_rule': ('🟡 LLM降级', 'warning'),
    }
    mode_label, _ = mode_map.get(mode, (mode, 'info'))
    st.caption(f"执行模式：{mode_label}")

    # LLM 回退提示
    if trace.get('llm_unavailable_fallback'):
        st.warning("⚠️ LLM 不可用，当前为规则模式")

    st.markdown("---")

    # 步骤列表
    steps = trace.get('steps', [])
    for step in steps:
        status = step.get('status', '')
        status_icon = {
            'success': '✅', 'error': '❌',
            'timeout': '⏱️', 'validation_error': '⚠️'
        }.get(status, '❓')

        step_num = step.get('step_index', step.get('step', '?'))
        tool_name = step.get('tool_name', step.get('tool', ''))
        action = step.get('action_desc', tool_name)

        with st.expander(f"{status_icon} 步骤{step_num}: {action}"):
            st.caption(f"工具：`{tool_name}`")
            st.caption(f"耗时：{step.get('exec_ms', step.get('time_ms', 0)):.0f}ms")

            if step.get('params_summary'):
                st.text("参数：")
                st.code(step.get('params_summary', ''), language='json')

            if step.get('result_summary'):
                st.text("结果：")
                st.write(step.get('result_summary', ''))

            if step.get('error'):
                st.error(step['error'])

    st.markdown("---")

    # 下载轨迹
    trace_json = json.dumps(trace, ensure_ascii=False, indent=2, default=str)
    st.download_button(
        "📥 下载轨迹 JSON",
        data=trace_json,
        file_name=f"trace_{trace.get('task_id', 'unknown')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # 上下文快照
    ctx = st.session_state.get('v2_context')
    if ctx:
        with st.expander("📊 上下文快照"):
            st.json(ctx)


def _render_result_actions(agent: AgentExecutorV2, review_workflow: ReviewWorkflow, db_path: str):
    """底部：质量判定结果 + 报告生成 + 复核操作"""
    result = st.session_state.get('v2_last_result')
    trace = st.session_state.get('v2_last_trace')

    if not result or not trace:
        return

    st.markdown("---")
    st.subheader("📊 检测结果与操作")

    quality = result.get('final_quality', '')
    scope_label = result.get('final_scope', '')

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.markdown("**质量判定**")
        badges = {
            'PASS': '<span class="quality-badge quality-pass">✅ PASS 合格</span>',
            'REVIEW': '<span class="quality-badge quality-review">⚠️ REVIEW 待复核</span>',
            'FAIL': '<span class="quality-badge quality-fail">❌ FAIL 不合格</span>',
        }
        badge_html = badges.get(quality, f'<span class="quality-badge quality-pending">{escape(quality)}</span>')
        st.markdown(badge_html, unsafe_allow_html=True)
        st.caption(f"范围：{scope_label}")

        # 免责声明
        st.warning(
            "⚠️ **免责声明**\n\n"
            "本判定基于演示规则，仅供参考，不构成工业质量判定依据。"
            "面积指标为检测框覆盖率代理指标。"
        )

    with col2:
        st.markdown("**操作**")

        has_evidence = bool(result.get('image_evidence') or result.get('batch_results'))
        generate_clicked = st.button(
            "生成 Word 报告", use_container_width=True, type="primary", disabled=not has_evidence
        )
        if (generate_clicked or st.session_state.pop('v2_report_requested', False)) and has_evidence:
            _generate_word_report(agent, result, trace)
        download = st.session_state.get('v2_report_download')
        if download:
            st.download_button(
                "下载 Word 报告", data=download['data'], file_name=download['filename'],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                on_click='ignore', use_container_width=True,
            )

        # 查看复核任务
        review_id = st.session_state.get('v2_last_review_id')
        if review_id:
            st.success(f"✅ 已创建复核任务：{review_id}")
            st.caption("请前往「📝 复核管理」页面处理")
        elif quality in ('REVIEW', 'FAIL'):
            if st.button("📝 手动创建复核任务", use_container_width=True):
                _create_manual_review(review_workflow, result)
                st.rerun()

    with col3:
        st.markdown("**检测摘要**")
        ctx = result.get('context_snapshot', {})
        batch = result.get('batch_results', [])
        det_count = (sum(row['detection_count'] for row in batch) if batch
                     else ctx.get('detection_count', 0))
        st.metric("缺陷数量", det_count)
        st.metric("执行步数", trace.get('total_steps', 0))
        st.metric("总耗时", f"{trace.get('total_time_ms', 0):.0f}ms")

    _render_detection_evidence(result)


def _render_detection_evidence(result):
    batch = result.get('batch_results')
    single = result.get('image_evidence')
    if not batch and not single:
        return

    st.subheader("检测证据与判定依据")
    if batch:
        categories = ['全部类别', *CLASS_NAMES_CN]
        focus = st.selectbox("检测框范围", categories, key='v2_evidence_class',
                             format_func=lambda item: CLASS_NAMES_CN.get(item, item))
        view = batch
        if focus != '全部类别':
            view = [row for row in batch if any(
                item['cls_name'] == focus for item in row.get('detections', [])
            )]
            count = sum(item['cls_name'] == focus for row in view for item in row['detections'])
            st.caption(f"子集：{CLASS_NAMES_CN[focus]}，{len(view)} 张图、{count} 个检测框。"
                       f"整批判定范围：{len(batch)} 个输入。")
        if not view:
            st.info("当前类别没有可查看的检测框。")
            return
        selected = st.selectbox(
            "查看图片与判定依据", list(range(len(view))),
            format_func=lambda idx: f"{view[idx]['image_name']} · {view[idx]['quality']}",
        )
        evidence = view[selected]
        if evidence.get('quality') == 'ERROR':
            st.warning("该文件读取或推理失败，已计入未完成。")
            st.code(evidence.get('error', '检测失败'))
            return
        evaluation = evidence.get('quality_evaluation', {})
        detections = [item for item in evidence.get('detections', [])
                      if focus == '全部类别' or item['cls_name'] == focus]
    else:
        evidence = single['image']
        evaluation = result.get('quality_evaluation', {})
        detections = result.get('context_snapshot', {}).get('current_detections', single['detections'])

    if evaluation:
        st.caption(f"判定范围：{evaluation.get('scope_label', '单图全量')} · "
                   f"规则：{evaluation['rule_id']} v{evaluation['rule_version']}")
        st.text(evaluation['explanation'])
    else:
        st.caption("当前记录没有保存判定依据，请重新检测。")

    path = evidence.get('image_path')
    if path and Path(path).is_file():
        with Image.open(path) as source:
            annotated = source.convert('RGB')
        draw = ImageDraw.Draw(annotated)
        for item in detections:
            box = (item['x1'], item['y1'], item['x2'], item['y2'])
            draw.rectangle(box, outline='#ff9f43', width=2)
            draw.text(box[:2], f"{item['cls_name']} {item['confidence']:.2f}", fill='#ffffff', stroke_width=1)
        st.image(annotated, caption=f"{evidence['image_name']} · {len(detections)} 个检测框", width=360)


def _generate_word_report(agent: AgentExecutorV2, result: Dict, trace: Dict):
    """生成 Word 报告（使用 v2 报告生成器，支持单图和批量模式）"""
    try:
        from models.agent.report_generator_v2 import QualityReportGenerator

        ctx = result.get('context_snapshot', {})
        batch_results = result.get('batch_results', [])
        batch_by_class = result.get('batch_by_class', {})
        final_scope = result.get('final_scope', '')

        # 轨迹信息
        trace_info = {
            'task_id': result.get('task_id', ''),
            'model_version': ctx.get('model_version', 'unknown'),
            'rule_version': ctx.get('rule_version', 'unknown'),
            'total_steps': trace.get('total_steps', 0),
            'success_steps': trace.get('success_count', 0),
            'total_time_ms': trace.get('total_time_ms', 0),
        }

        gen = QualityReportGenerator()

        # ===== 批量模式：生成批次报告 =====
        if batch_results:
            total_images = len(batch_results)
            from models.agent.quality_semantics import batch_quality_from_individuals

            qualities = [r.get('quality', '') for r in batch_results]
            pass_count = qualities.count('PASS')
            review_count = qualities.count('REVIEW')
            fail_count = qualities.count('FAIL')
            completed = pass_count + review_count + fail_count
            total_detections = sum(r.get('detection_count', 0) for r in batch_results)
            batch_quality, batch_reason = batch_quality_from_individuals(
                qualities, total_images, completed
            )

            # 构建图像详情列表（适配报告生成器的字段）
            images_for_report = []
            for r in batch_results:
                images_for_report.append({
                    'image_name': r.get('image_name', ''),
                    'status': 'failed' if r.get('quality') == 'ERROR' else 'success',
                    'detection_count': r.get('detection_count', 0),
                    'quality': r.get('quality', '-'),
                    'quality_score': r.get('score', 0),
                    'is_forced': False,
                })

            # 构建批次结果（适配 generate_batch_report 的格式）
            batch_data = {
                'batch_id': result.get('task_id', ''),
                'start_time': trace.get('start_time', ''),
                'end_time': trace.get('end_time', ''),
                'batch_quality': batch_quality,
                'batch_quality_reason': batch_reason,
                'model_version': ctx.get('model_version', 'unknown'),
                'rule_version': ctx.get('rule_version', 'unknown'),
                'rule_is_demo': ctx.get('rule_is_demo', True),
                'summary': {
                    'planned': total_images,
                    'success': completed,
                    'failed': total_images - completed,
                    'skipped': 0,
                    'pass': pass_count,
                    'fail': fail_count,
                    'review': review_count,
                    'total_detections': total_detections,
                    'by_class': batch_by_class,
                },
                'images': images_for_report,
            }

            gen.generate_batch_report(batch_data, trace_info)
            file_name = f"批次质检报告_{total_images}张_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

        # ===== 单图模式：生成单图报告 =====
        else:
            quality_result = result.get('quality_evaluation') or _extract_full_eval_result(trace)
            image_name = ctx.get('image_name', st.session_state.get('v2_image_name', 'unknown'))
            is_subset = ctx.get('is_subset', False)
            subset_class = ctx.get('subset_class', '')

            # 检测摘要
            detections = ctx.get('current_detections', [])
            by_class = {}
            total_conf = 0
            for d in detections:
                cls_name = d.get('cls_name', '') if isinstance(d, dict) else d[0]
                conf = d.get('confidence', 0) if isinstance(d, dict) else d[1]
                by_class[cls_name] = by_class.get(cls_name, 0) + 1
                total_conf += conf
            avg_conf = total_conf / len(detections) if detections else 0

            detection_summary = {
                'total_count': len(detections),
                'by_class': by_class,
                'avg_confidence': avg_conf,
                'area_ratio': quality_result.get('total_box_area_pct', 0),
            }

            gen.generate_single_image_report(
                image_name=image_name,
                quality_result=quality_result,
                detection_summary=detection_summary,
                trace_info=trace_info,
                scope_label=final_scope or '单图全量',
                is_subset=is_subset,
                subset_class=subset_class,
            )
            file_name = f"质检报告_{image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

        # 保存到内存并提供下载
        buf = io.BytesIO()
        gen.doc.save(buf)
        buf.seek(0)

        st.session_state['v2_report_download'] = {'data': buf.getvalue(), 'filename': file_name}

    except ImportError as e:
        st.error(f"报告生成失败：缺少依赖 - {str(e)}")
        st.caption("请确保已安装 python-docx")
    except Exception as e:
        st.error(f"报告生成失败：{str(e)}")


def _extract_full_eval_result(trace):
    """从 trace 中提取完整的质量评估结果"""
    steps = trace.get('steps', [])
    for step in reversed(steps):
        if step.get('tool_name') == 'quality_judge' and step.get('status') == 'success':
            # 尝试从 step 数据中提取完整结果
            # 这里用摘要信息构建一个基础结果
            summary = step.get('result_summary', '')
            import re
            score_match = re.search(r'(\d+)分', summary)
            score = int(score_match.group(1)) if score_match else 0

            quality = ''
            if 'FAIL' in summary:
                quality = 'FAIL'
            elif 'REVIEW' in summary:
                quality = 'REVIEW'
            elif 'PASS' in summary:
                quality = 'PASS'

            return {
                'quality': quality,
                'score': score,
                'explanation': summary,
                'scope_label': trace.get('final_scope', ''),
                'is_demo_rule': True,
                'rule_id': 'demo_default',
                'rule_version': '1.0.0',
                'rule_source': '演示规则',
                'total_box_area_pct': 0,
                'is_forced': False,
                'forced_reason': '',
                'method_notes': [
                    '⚠️ 演示规则：本判定基于实验演示规则，不构成工业质量判定依据',
                    '面积指标为「检测框覆盖率」，是真实缺陷面积的代理指标，非物理测量值',
                ],
            }
    return {
        'quality': '', 'score': 0, 'explanation': '',
        'scope_label': '', 'is_demo_rule': True,
        'rule_id': 'demo_default', 'rule_version': '1.0.0',
        'rule_source': '演示规则', 'total_box_area_pct': 0,
        'is_forced': False, 'forced_reason': '',
        'method_notes': [],
    }


def _create_manual_review(review_workflow: ReviewWorkflow, result: Dict):
    """手动创建复核任务"""
    try:
        ctx = result.get('context_snapshot', {})
        image_path = st.session_state.get('v2_image_path', '')
        image_name = st.session_state.get('v2_image_name', 'unknown')
        quality = result.get('final_quality', 'REVIEW')

        review_id = review_workflow.create_task(
            task_id=result.get('task_id', ''),
            record_id=0,
            image_name=image_name,
            image_path=image_path,
            original_quality=quality,
            original_score=0,
            original_scope=result.get('final_scope', ''),
            reason='手动创建复核任务',
            quality_issue=quality,
            priority='normal',
            model_version=ctx.get('model_version', ''),
            rule_version=ctx.get('rule_version', ''),
            original_detections=[],
            original_eval={},
        )
        st.session_state['v2_last_review_id'] = review_id
        st.success(f"✅ 复核任务已创建：{review_id}")
    except Exception as e:
        st.error(f"创建复核任务失败：{str(e)}")


def _render_safety_footer():
    """底部安全提示"""
    st.markdown("---")
    st.caption(
        "🛡️ **安全说明**\n\n"
        "• 本系统使用演示规则，判定结果仅供参考，不构成工业质量判定依据。\n"
        "• 面积指标为「检测框覆盖率」，是真实缺陷面积的代理指标，非物理测量值。\n"
        "• 未检测到缺陷仅表示模型未识别出已知缺陷类别，不等于产品合格。\n"
        "• 子集范围内的判定结论仅针对筛选后的缺陷，不代表整体质量。\n"
        "• 所有判定过程可通过执行轨迹追溯。"
    )
