# -*- coding: utf-8 -*-
"""
基于 GAN-YOLO 与 AI Agent 的工业表面缺陷智能检测及质量分析系统
================================================================
功能模块：
1. 质检Agent v2 —— 可追溯版质检 Agent（LLM/规则双模式）
2. AI Agent 对话 —— 自然语言交互 + 多工具调用
3. 智能检测 —— 单张/批量图像检测 + 三级质量判定
4. 数据分析 —— 缺陷统计与趋势分析
5. 质量规则 —— 三级判定规则配置
6. 报告中心 —— v2 版 8 大章节 Word 报告
7. 复核管理 —— 复核任务列表、RBAC、操作日志
"""

import streamlit as st
import os
import sys
import json
import tempfile
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 项目根目录（基于 app.py 所在位置，避免硬编码绝对路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 如果 venv 存在，将 venv 的 site-packages 加入路径（解决 Anaconda 环境缺少依赖的问题）
_venv_site_packages = os.path.join(BASE_DIR, 'venv', 'Lib', 'site-packages')
if os.path.isdir(_venv_site_packages):
    sys.path.insert(0, _venv_site_packages)

# 页面配置
st.set_page_config(
    page_title="AI Agent 智能缺陷检测系统",
    page_icon="🔍",
    layout="wide",
)

# ============================================================
# 自定义样式（统一质量徽章 + 响应式优化）
# ============================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    /* 统一质量徽章样式 */
    .quality-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        color: white;
    }
    .quality-pass { background: #28a745; }
    .quality-review { background: #ffc107; color: #333; }
    .quality-fail { background: #dc3545; }
    .quality-pending { background: #6c757d; }
    .quality-confirmed { background: #17a2b8; }
    .quality-rejected { background: #6f42c1; }
    .quality-corrected { background: #fd7e14; }

    /* 顶部免责声明提示条 */
    .disclaimer-bar {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 13px;
        color: #856404;
    }
    .disclaimer-bar strong { color: #856404; }

    /* 对话气泡 */
    .chat-user {
        background: #e7f3ff;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 8px 0;
        margin-right: 20%;
    }
    .chat-agent {
        background: #f0f0f0;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 8px 0;
        margin-left: 20%;
    }
    .tool-call {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin: 4px 0;
    }

    /* LLM 状态指示器 */
    .llm-status-connected { color: #28a745; font-weight: bold; }
    .llm-status-disconnected { color: #dc3545; font-weight: bold; }
    .llm-status-mock { color: #ffc107; font-weight: bold; }

    /* 卡片容器 */
    .info-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #1f77b4;
    }

    /* 响应式：小屏幕调整侧边栏 */
    @media (max-width: 768px) {
        .main-header { font-size: 20px; }
    }

    /* 表格行高亮 */
    .review-row-pending { background-color: #fff3cd !important; }
    .review-row-confirmed { background-color: #d4edda !important; }
    .review-row-rejected { background-color: #f8d7da !important; }
    .review-row-corrected { background-color: #ffeeba !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 模型加载（缓存）
# ============================================================

def find_model_path():
    """
    按优先级查找可用的检测模型
    返回: (model_path, model_description)
    """
    candidates = [
        # 优先：部署专用模型目录
        (os.path.join(BASE_DIR, "models", "weights", "best.pt"),
         "YOLOv8s + cDCGAN 增强模型"),
        # 备选：GAN 增强模型
        (os.path.join(BASE_DIR, "outputs", "phase2", "yolov8s_gan", "weights", "best.pt"),
         "YOLOv8s + cDCGAN 增强模型"),
        (os.path.join(BASE_DIR, "outputs", "phase2", "yolo_gan", "weights", "best.pt"),
         "YOLOv8s + cDCGAN 增强模型"),
        # 次选：基线模型
        (os.path.join(BASE_DIR, "outputs", "baseline", "yolov8s_baseline-2", "weights", "best.pt"),
         "YOLOv8s 基线模型（纯真实数据）"),
        (os.path.join(BASE_DIR, "outputs", "baseline", "yolov8s_baseline", "weights", "best.pt"),
         "YOLOv8 基线模型"),
        # 兜底：根目录预训练权重
        (os.path.join(BASE_DIR, "yolov8s.pt"),
         "YOLOv8s 官方预训练权重（未微调）"),
    ]
    for path, desc in candidates:
        if os.path.exists(path):
            return path, desc
    return None, "未找到模型"


@st.cache_resource
def load_agent():
    from models.agent.detection_agent import AIAgent

    model_path, model_desc = find_model_path()
    if model_path is None:
        st.error("❌ 未找到任何模型权重文件，请先训练模型或放入预训练权重。")
        st.stop()

    db_path = os.path.join(BASE_DIR, "outputs", "agent", "app_records.db")
    agent = AIAgent(model_path, db_path)
    # 附加模型信息，供界面显示
    agent.model_path = model_path
    agent.model_description = model_desc
    return agent


@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO

    model_path, model_desc = find_model_path()
    if model_path is None:
        st.error("❌ 未找到任何模型权重文件。")
        st.stop()

    model = YOLO(model_path)
    model.model_path = model_path
    model.model_description = model_desc
    return model


@st.cache_resource
def load_agent_v2(llm_provider_override=None):
    """加载 Agent v2 执行器（可追溯版），支持 LLM Provider 覆盖"""
    from models.agent import AgentExecutorV2, AgentConfig

    model = load_yolo_model()
    db_path = os.path.join(BASE_DIR, "outputs", "agent", "app_records.db")

    # 确定 LLM 配置
    if llm_provider_override and llm_provider_override != 'auto':
        provider = llm_provider_override
        llm_model = ''
        llm_api_key = ''
        llm_base_url = ''

        if provider == 'deepseek':
            llm_model = 'deepseek-chat'
            llm_base_url = 'https://api.deepseek.com/v1'
            # 尝试从环境变量获取 key
            llm_api_key = os.environ.get('DEEPSEEK_API_KEY', os.environ.get('LLM_API_KEY', ''))
        elif provider == 'openai':
            llm_model = 'gpt-4o-mini'
            llm_api_key = os.environ.get('OPENAI_API_KEY', '')
        elif provider == 'mock':
            llm_model = 'mock-agent'
        else:  # none / rule
            provider = 'none'
    else:
        # 自动检测
        from models.agent.llm_integration import load_llm_config_from_env
        env_config = load_llm_config_from_env()
        provider = env_config['provider']
        llm_model = env_config.get('model', '')
        llm_api_key = env_config.get('api_key', '')
        llm_base_url = env_config.get('base_url', '')

    config = AgentConfig(
        max_steps=10,
        max_cost_credits=100,
        llm_provider=provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        rule_id='demo_default',
        auto_save_trace=True,
    )

    agent = AgentExecutorV2(config, model=model, db_path=db_path)
    agent.model_description = getattr(model, 'model_description', 'unknown')
    return agent


@st.cache_resource
def load_review_workflow():
    """加载复核工作流（SQLite）"""
    from models.agent import ReviewWorkflow

    review_db_path = os.path.join(BASE_DIR, "outputs", "agent", "review.db")
    os.makedirs(os.path.dirname(review_db_path), exist_ok=True)
    return ReviewWorkflow(review_db_path)


@st.cache_resource
def load_report_generator():
    """加载报告生成器 v2"""
    from models.agent.report_generator_v2 import QualityReportGenerator
    return QualityReportGenerator


# 类别颜色
CLASS_COLORS = {
    'crazing': (255, 0, 0),
    'inclusion': (255, 165, 0),
    'patches': (255, 255, 0),
    'pitted_surface': (0, 255, 0),
    'rolled-in_scale': (0, 0, 255),
    'scratches': (128, 0, 128),
}

CLASS_NAMES_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化皮',
    'scratches': '划痕',
}


def draw_detections(image, detections):
    """在图像上绘制检测框"""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        cls_name, conf, x1, y1, x2, y2 = det
        color = CLASS_COLORS.get(cls_name, (255, 0, 0))
        label = f"{CLASS_NAMES_CN.get(cls_name, cls_name)} {conf:.2f}"

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 画标签背景
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_h - 2), label, fill="white", font=font)

    return img


def get_quality_badge(quality):
    """获取质量等级标签HTML（统一徽章样式）"""
    badges = {
        'PASS': '<span class="quality-badge quality-pass">✅ PASS 合格</span>',
        'REVIEW': '<span class="quality-badge quality-review">⚠️ REVIEW 待复核</span>',
        'FAIL': '<span class="quality-badge quality-fail">❌ FAIL 不合格</span>',
    }
    return badges.get(quality, f'<span class="quality-badge quality-pending">{quality}</span>')


def get_review_status_badge(status):
    """获取复核状态标签HTML"""
    status_map = {
        'pending': ('⏳ 待处理', 'quality-review'),
        'in_progress': ('🔄 处理中', 'quality-pending'),
        'confirmed': ('✅ 已确认', 'quality-confirmed'),
        'rejected': ('❌ 已驳回', 'quality-rejected'),
        'corrected': ('✏️ 已修正', 'quality-corrected'),
        'escalated': ('⬆️ 已升级', 'quality-fail'),
    }
    label, cls = status_map.get(status, (status, 'quality-pending'))
    return f'<span class="quality-badge {cls}">{label}</span>'


# ============================================================
# 顶部免责声明提示条
# ============================================================
def render_disclaimer_bar():
    """渲染顶部免责声明提示条"""
    st.markdown(
        """
        <div class="disclaimer-bar">
            <strong>⚠️ 免责声明：</strong>
            本系统使用演示规则进行质量判定，结果仅供参考，不构成工业质量判定依据。
            面积指标为「检测框覆盖率」代理指标，非物理测量值。
            所有检测与判定过程可通过执行轨迹追溯。
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LLM 状态获取
# ============================================================
def get_llm_status_info():
    """获取 LLM 状态信息，用于侧边栏显示"""
    provider = st.session_state.get('llm_provider', 'auto')

    if provider == 'none' or provider == 'rule':
        return {
            'status': 'rule',
            'label': '规则模式',
            'badge_class': 'llm-status-disconnected',
            'icon': '⚙️',
        }
    elif provider == 'mock':
        return {
            'status': 'mock',
            'label': 'Mock 模式',
            'badge_class': 'llm-status-mock',
            'icon': '🎭',
        }
    else:
        # 尝试检测是否真的可用
        try:
            from models.agent.llm_integration import load_llm_config_from_env
            env_cfg = load_llm_config_from_env()
            if env_cfg.get('api_key'):
                return {
                    'status': 'connected',
                    'label': f'已连接 ({env_cfg.get("provider", "?")})',
                    'badge_class': 'llm-status-connected',
                    'icon': '🟢',
                    'model': env_cfg.get('model', ''),
                }
            else:
                return {
                    'status': 'disconnected',
                    'label': '未连接',
                    'badge_class': 'llm-status-disconnected',
                    'icon': '🔴',
                }
        except Exception:
            return {
                'status': 'unknown',
                'label': '未知',
                'badge_class': 'llm-status-disconnected',
                'icon': '❓',
            }


# ============================================================
# 登录验证（参赛作品展示用，评委通过账号密码登录体验）
# ============================================================
def check_login():
    """简单的登录验证，返回是否已登录"""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    return st.session_state['logged_in']

def login_page():
    """显示登录页面"""
    st.markdown('<div style="text-align: center; padding: 60px 20px;">', unsafe_allow_html=True)
    st.markdown('# 🔍 AI Agent 智能缺陷检测系统')
    st.markdown('### 工业表面缺陷智能检测及质量分析平台')
    st.markdown('---')
    st.markdown('#### 评委体验登录')

    # 预设的评委账号（可根据需要添加多个）
    JUDGE_ACCOUNTS = {
        'judge': 'Judge@2026',
        'pingwei': 'PW@ai2026',
        'demo': 'demo123456',
    }

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input('👤 账号', key='login_username')
        password = st.text_input('🔒 密码', type='password', key='login_password')

        if st.button('登 录', use_container_width=True, type='primary'):
            if username in JUDGE_ACCOUNTS and JUDGE_ACCOUNTS[username] == password:
                st.session_state['logged_in'] = True
                st.session_state['login_user'] = username
                st.success('✅ 登录成功！正在进入系统...')
                st.rerun()
            else:
                st.error('❌ 账号或密码错误，请重试')

        st.markdown('---')
        st.caption('💡 提示：本系统为参赛作品演示版，仅供评委体验使用')
        st.caption('📧 如有问题请联系参赛团队')

    st.markdown('</div>', unsafe_allow_html=True)

# 如果未登录，显示登录页并停止后续渲染
if not check_login():
    login_page()
    st.stop()

# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.title("🔍 功能导航")
# 显示当前登录用户
st.sidebar.caption(f"👤 当前用户：{st.session_state.get('login_user', 'guest')}")
if st.sidebar.button("🚪 退出登录", key="logout_btn"):
    st.session_state['logged_in'] = False
    st.rerun()
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "选择模块",
    ["🔬 质检Agent v2", "🤖 AI Agent 对话", "🔍 智能检测",
     "📊 数据分析", "⚙️ 质量规则", "🎨 GAN数据增强", "📋 报告中心", "📝 复核管理"],
    index=0,
)

st.sidebar.markdown("---")

# LLM 状态显示
llm_info = get_llm_status_info()
st.sidebar.markdown(
    f"### {llm_info['icon']} LLM 状态\n"
    f"<span class='{llm_info['badge_class']}'>{llm_info['label']}</span>",
    unsafe_allow_html=True,
)
if llm_info.get('model'):
    st.sidebar.caption(f"模型: {llm_info['model']}")

st.sidebar.markdown("---")

# 模型信息
try:
    _agent_for_info = load_agent()
    model_desc = getattr(_agent_for_info, 'model_description', '未知')
    model_badge = "🟢" if "GAN" in model_desc else "🔵"
    st.sidebar.markdown(f"{model_badge} **当前模型**\n\n{model_desc}")
except Exception:
    st.sidebar.warning("⚠️ 模型加载失败")

st.sidebar.markdown("---")

# 📥 测试图片下载（按缺陷类型分组）
sample_dir = os.path.join(BASE_DIR, "sample_images")
if os.path.isdir(sample_dir):
    st.sidebar.markdown("📥 **测试图片下载**")
    st.sidebar.caption("100 张示例图片，6 类缺陷，点击展开下载")
    
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # 按缺陷类型分组
    defect_groups = {}
    defect_names = {
        "crazing": ("裂纹", "Crazing"),
        "inclusion": ("夹杂物", "Inclusion"),
        "patches": ("斑块", "Patches"),
        "pitted": ("麻点", "Pitted Surface"),
        "rolled-in": ("氧化皮", "Rolled-in Scale"),
        "scratches": ("划痕", "Scratches"),
    }
    
    for fname in sample_files:
        # 提取缺陷类型前缀（第一个下划线前的部分）
        prefix = fname.split('_')[0]
        if prefix not in defect_groups:
            defect_groups[prefix] = []
        defect_groups[prefix].append(fname)
    
    # 生成打包 ZIP（如果不存在）
    import zipfile
    import io
    
    zip_path = os.path.join(sample_dir, "all_samples.zip")
    
    for prefix in sorted(defect_groups.keys()):
        files = defect_groups[prefix]
        cn_name, en_name = defect_names.get(prefix, (prefix, prefix))
        count = len(files)
        
        with st.sidebar.expander(f"📁 {cn_name} ({en_name}) - {count} 张"):
            # 打包下载按钮
            zip_bytes = io.BytesIO()
            with zipfile.ZipFile(zip_bytes, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname in files:
                    fpath = os.path.join(sample_dir, fname)
                    zf.write(fpath, arcname=fname)
            zip_bytes.seek(0)
            
            st.download_button(
                label=f"📦 全部下载 ({count}张)",
                data=zip_bytes.getvalue(),
                file_name=f"{prefix}_{count}张.zip",
                mime="application/zip",
                key=f"zip_{prefix}",
                use_container_width=True,
            )
            
            st.markdown("---")
            
            # 前 5 张单独下载
            for fname in files[:5]:
                fpath = os.path.join(sample_dir, fname)
                try:
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {fname}",
                            data=f.read(),
                            file_name=fname,
                            mime="image/jpeg",
                            key=f"sample_{fname}",
                            use_container_width=True,
                        )
                except Exception:
                    pass
            
            if count > 5:
                st.caption(f"... 还有 {count - 5} 张，点击上方全部下载")

st.sidebar.markdown("---")
st.sidebar.info(
    "**系统说明**\n\n"
    "本系统基于 cDCGAN + YOLOv8 + AI Agent 构建，"
    "支持自然语言交互、多工具协调、三级质量判定。\n\n"
    "规则参数为实验演示设定，非生产标准。"
)


# ============================================================
# 渲染顶部免责声明（所有页面）
# ============================================================
render_disclaimer_bar()


# ============================================================
# 模块0：质检 Agent v2（可追溯版）
# ============================================================
if page == "🔬 质检Agent v2":
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'pages'))

    # 获取当前 LLM Provider 选择
    llm_provider_sel = st.session_state.get('llm_provider', 'auto')
    agent_v2 = load_agent_v2(llm_provider_override=llm_provider_sel)

    db_path = os.path.join(BASE_DIR, "outputs", "agent", "app_records.db")
    review_workflow = load_review_workflow()

    from pages.agent_v2_page import render_agent_v2_page
    render_agent_v2_page(agent_v2, db_path, review_workflow)


# ============================================================
# 模块1：AI Agent 对话
# ============================================================
elif page == "🤖 AI Agent 对话":
    st.markdown('<div class="main-header">🤖 AI Agent 智能对话</div>', unsafe_allow_html=True)
    st.markdown("用自然语言下达检测任务，Agent 会自动理解意图、选择工具并执行。")

    agent = load_agent()

    # 输入模式选择
    input_mode = st.radio(
        "📂 输入方式",
        ["单张图片", "文件夹批量"],
        horizontal=True,
        key="agent_input_mode"
    )

    if input_mode == "单张图片":
        # 上传单张图片
        uploaded_file = st.file_uploader(
            "📁 上传检测图片（可选，支持继续对话）",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="agent_upload"
        )

        if uploaded_file is not None:
            # 保存临时文件
            tmp_dir = os.path.join(BASE_DIR, "outputs", "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"agent_upload_{datetime.now().strftime('%H%M%S')}.png")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['agent_image'] = tmp_path
            st.session_state['agent_image_name'] = uploaded_file.name
            st.session_state['agent_folder'] = None  # 清除文件夹模式

            # 显示缩略图
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(uploaded_file, caption="已上传图片", use_column_width=True)
            with col2:
                st.success(f"✅ 已上传：{uploaded_file.name}")
                st.caption("现在可以在下方输入自然语言指令，例如：")
                st.caption("• 检测一下这张图有什么缺陷")
                st.caption("• 找出裂纹缺陷有多少个")
                st.caption("• 质量合格吗？为什么？")
                st.caption("• 生成报告")

    else:
        # 文件夹批量模式
        st.info("💡 选择要检测的图片文件夹，然后用自然语言下达指令")

        col1, col2 = st.columns([3, 1])
        with col1:
            folder_path = st.text_input(
                "📂 文件夹路径",
                key="agent_folder_path",
                placeholder="例如：C:\\Users\\User\\Desktop\\images 或 D:\\dataset\\test"
            )
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📁 选择文件夹", key="pick_folder_btn"):
                # 用 streamlit 原生方式，我们这里只是提示用户粘贴路径
                st.info("请将文件夹路径粘贴到上方输入框中")

        if folder_path and os.path.isdir(folder_path):
            # 统计图片数量
            image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [f for f in os.listdir(folder_path)
                          if f.lower().endswith(image_exts)]

            if image_files:
                st.session_state['agent_folder'] = folder_path
                st.session_state['agent_image'] = None  # 清除单图模式
                st.success(f"✅ 已加载文件夹：{folder_path}")
                st.metric("待检测图片数", len(image_files))

                # 显示前几张预览
                with st.expander(f"📸 预览图片（共 {len(image_files)} 张，显示前 6 张）"):
                    preview_imgs = sorted(image_files)[:6]
                    cols = st.columns(3)
                    for i, img_name in enumerate(preview_imgs):
                        with cols[i % 3]:
                            try:
                                st.image(
                                    os.path.join(folder_path, img_name),
                                    caption=img_name,
                                    use_column_width=True
                                )
                            except Exception:
                                st.caption(img_name)

                st.caption("示例指令：")
                st.caption("• 批量检测这个文件夹里的图片")
                st.caption("• 合格率是多少？")
                st.caption("• 哪些不合格？")
                st.caption("• 统计一下缺陷分布")
                st.caption("• 生成批次报告")
            else:
                st.warning("⚠️ 该文件夹中没有找到图片文件（支持 jpg/jpeg/png/bmp）")
                st.session_state['agent_folder'] = None
        elif folder_path:
            st.error(f"❌ 文件夹不存在：{folder_path}")
            st.session_state['agent_folder'] = None

    # 初始化对话历史
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
            {
                'role': 'agent',
                'content': '您好！我是 AI 质量检测助手 👋\n\n'
                           '您可以上传图片，然后用自然语言告诉我要做什么。\n'
                           '例如："检测一下缺陷"、"统计裂纹数量"、"合格吗"、"为什么不合格"等。'
            }
        ]

    # 显示对话历史
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Agent 回复可能包含工具调用信息
            content = msg.get('content', '')
            tool_info = msg.get('tool_info', '')
            st.markdown(f'<div class="chat-agent">🤖 {content.replace(chr(10), "<br>")}</div>',
                       unsafe_allow_html=True)
            if tool_info:
                st.markdown(f'<div class="tool-call">🔧 调用工具：{tool_info}</div>',
                           unsafe_allow_html=True)

    # 输入框
    user_input = st.text_input("💬 输入您的指令：", key="agent_input", placeholder="例如：检测一下这张图有什么缺陷")

    if st.button("发送", key="send_btn", use_container_width=True):
        if not user_input.strip():
            st.warning("请输入指令")
        else:
            # 添加用户消息
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': user_input
            })

            # 执行 Agent 任务
            image_path = st.session_state.get('agent_image')
            folder_path = st.session_state.get('agent_folder')
            result = agent.execute_task(user_input, image=image_path, folder=folder_path)

            # 添加 Agent 回复
            tool_calls = " → ".join(result['tool_calls'])
            st.session_state['chat_history'].append({
                'role': 'agent',
                'content': result['final_response'],
                'tool_info': f"意图识别: {result['intent']} (置信度 {result['confidence']}) | 工具调用: {tool_calls}"
            })

            # 如果有单图检测结果，显示图片
            if result['detections'] and image_path:
                with st.expander("🔍 查看检测可视化结果"):
                    img = Image.open(image_path)
                    annotated = draw_detections(img, result['detections'])
                    st.image(annotated, caption="检测结果可视化", use_column_width=True)

            # 如果有批量检测结果，显示汇总
            batch_results = result.get('batch_results')
            if batch_results:
                with st.expander(f"📊 批次检测汇总（共 {len(batch_results)} 张）"):
                    # 统计
                    pass_count = sum(1 for r in batch_results if r.get('quality') == 'PASS')
                    fail_count = sum(1 for r in batch_results if r.get('quality') == 'FAIL')
                    review_count = sum(1 for r in batch_results if r.get('quality') == 'REVIEW')
                    total_det = sum(r.get('detection_count', 0) for r in batch_results)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("总图片数", len(batch_results))
                    col2.metric("合格 PASS", pass_count)
                    col3.metric("不合格 FAIL", fail_count)
                    col4.metric("缺陷总数", total_det)

                    # 列表
                    st.dataframe(
                        [{'图片名': r.get('image_name', ''),
                          '质量等级': r.get('quality', '-'),
                          '缺陷数': r.get('detection_count', 0),
                          '评分': r.get('score', 0)}
                         for r in batch_results],
                        use_container_width=True,
                        height=300
                    )

            st.rerun()

    # 清空对话
    if st.button("🗑️ 清空对话", key="clear_chat"):
        st.session_state['chat_history'] = [
            {
                'role': 'agent',
                'content': '对话已清空。您好！我是 AI 质量检测助手 👋\n\n'
                           '您可以上传图片，然后用自然语言告诉我要做什么。'
            }
        ]
        st.rerun()


# ============================================================
# 模块2：智能检测
# ============================================================
elif page == "🔍 智能检测":
    st.markdown('<div class="main-header">🔍 智能缺陷检测</div>', unsafe_allow_html=True)

    agent = load_agent()
    model = load_yolo_model()

    # 模式选择
    mode = st.radio("检测模式", ["单张检测", "批量检测"], horizontal=True)

    # 置信度阈值
    conf_threshold = st.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)

    if mode == "单张检测":
        uploaded_file = st.file_uploader(
            "上传检测图片",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="single_upload"
        )

        if uploaded_file is not None:
            # 保存临时文件
            tmp_dir = os.path.join(BASE_DIR, "outputs", "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"single_{datetime.now().strftime('%H%M%S')}.png")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 运行检测
            with st.spinner("检测中..."):
                import time
                t0 = time.time()
                results = model.predict(source=tmp_path, conf=conf_threshold, verbose=False)
                inference_time = (time.time() - t0) * 1000

                detections = []
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls_idx = int(box.cls[0].item())
                            cls_name = model.names[cls_idx] if cls_idx < len(model.names) else str(cls_idx)
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append([cls_name, round(conf, 4),
                                             round(x1, 1), round(y1, 1),
                                             round(x2, 1), round(y2, 1)])

            # 显示结果
            col1, col2 = st.columns([2, 1])

            with col1:
                img = Image.open(tmp_path)
                annotated = draw_detections(img, detections)
                st.image(annotated, caption=f"检测结果（{len(detections)}个缺陷）",
                        use_column_width=True)

            with col2:
                st.subheader("检测结果")
                st.metric("缺陷数量", len(detections))
                st.metric("推理耗时", f"{inference_time:.1f} ms")

                # 质量判定
                eval_result = agent.rule_engine.evaluate(detections)
                st.markdown(f"### 质量判定：{get_quality_badge(eval_result['quality'])}",
                           unsafe_allow_html=True)
                st.metric("综合评分", f"{eval_result['score']} 分")

                # 保存按钮
                if st.button("💾 保存记录", use_container_width=True):
                    record_id = agent._save_record(
                        image_name=uploaded_file.name,
                        detections=detections,
                        eval_result=eval_result,
                    )
                    st.success(f"✅ 记录已保存 (ID: {record_id})")

            # 详细检测列表
            if detections:
                st.markdown("---")
                st.subheader("📋 检测详情")
                df = pd.DataFrame(detections, columns=['类别', '置信度', 'x1', 'y1', 'x2', 'y2'])
                df['类别'] = df['类别'].map(lambda x: f"{x} ({CLASS_NAMES_CN.get(x, x)})")
                st.dataframe(df, use_container_width=True)

                # 判定详情
                with st.expander("📖 查看质量判定详情"):
                    st.write(eval_result['explanation'])
                    if eval_result['reasons_fail']:
                        st.write("**不合格原因：**")
                        for r in eval_result['reasons_fail']:
                            st.write(f"  • {r}")
                    if eval_result['reasons_review']:
                        st.write("**复核建议：**")
                        for r in eval_result['reasons_review']:
                            st.write(f"  • {r}")

    else:  # 批量检测
        uploaded_files = st.file_uploader(
            "上传多张检测图片",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files:
            st.info(f"已上传 {len(uploaded_files)} 张图片")

            if st.button("🚀 开始批量检测", type="primary"):
                progress_bar = st.progress(0)
                results_list = []

                for i, f in enumerate(uploaded_files):
                    # 保存临时文件
                    tmp_dir = os.path.join(BASE_DIR, "outputs", "temp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    tmp_path = os.path.join(tmp_dir, f"batch_{i}_{datetime.now().strftime('%H%M%S')}.png")
                    with open(tmp_path, "wb") as fobj:
                        fobj.write(f.getbuffer())

                    # 检测
                    results = model.predict(source=tmp_path, conf=conf_threshold, verbose=False)
                    detections = []
                    if results and len(results) > 0:
                        r = results[0]
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls_idx = int(box.cls[0].item())
                                cls_name = model.names[cls_idx] if cls_idx < len(model.names) else str(cls_idx)
                                conf = float(box.conf[0].item())
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                detections.append([cls_name, round(conf, 4),
                                                 round(x1, 1), round(y1, 1),
                                                 round(x2, 1), round(y2, 1)])

                    eval_result = agent.rule_engine.evaluate(detections)

                    results_list.append({
                        '文件名': f.name,
                        '缺陷数量': len(detections),
                        '质量等级': eval_result['quality'],
                        '评分': eval_result['score'],
                    })

                    progress_bar.progress((i + 1) / len(uploaded_files))

                # 显示汇总
                st.success(f"✅ 批量检测完成！共 {len(uploaded_files)} 张图片")

                df = pd.DataFrame(results_list)
                st.dataframe(df, use_container_width=True)

                # 统计
                st.markdown("### 📊 质量分布")
                quality_counts = df['质量等级'].value_counts()
                for q, c in quality_counts.items():
                    st.markdown(f"- {get_quality_badge(q)}：{c} 张", unsafe_allow_html=True)


# ============================================================
# 模块3：数据分析
# ============================================================
elif page == "📊 数据分析":
    st.markdown('<div class="main-header">📊 缺陷数据分析</div>', unsafe_allow_html=True)

    agent = load_agent()
    stats = agent.get_statistics()

    # 概览卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总检测数", stats['total_records'])
    col2.metric("合格数", stats['pass_count'])
    col3.metric("不合格数", stats['fail_count'])
    col4.metric("待复核数", stats['review_count'])

    col1, col2, col3 = st.columns(3)
    col1.metric("合格率", f"{stats['pass_rate']}%")
    col2.metric("平均评分", f"{stats['avg_score']} 分")

    # 质量分布饼图数据
    st.markdown("---")
    st.subheader("质量等级分布")

    q_data = stats['quality_distribution']
    if q_data:
        chart_data = pd.DataFrame({
            '等级': list(q_data.keys()),
            '数量': list(q_data.values())
        })
        st.bar_chart(chart_data.set_index('等级'))

    # 最近记录
    st.markdown("---")
    st.subheader("📋 最近检测记录")
    records = agent.query_records(limit=20)
    if records:
        df = pd.DataFrame(records)
        df = df[['id', 'image_name', 'timestamp', 'total_defects', 'quality', 'quality_score']]
        df.columns = ['ID', '图片名', '检测时间', '缺陷数', '质量等级', '评分']
        st.dataframe(df, use_container_width=True)
    else:
        st.info("暂无检测记录，请先在检测模块进行检测")


# ============================================================
# 模块4：质量规则配置
# ============================================================
elif page == "⚙️ 质量规则":
    st.markdown('<div class="main-header">⚙️ 三级质量判定规则</div>', unsafe_allow_html=True)

    agent = load_agent()
    config = agent.rule_engine.config

    st.info("📌 本系统采用三级质量判定体系：**PASS（合格）→ REVIEW（待复核）→ FAIL（不合格）**")
    st.caption("以下参数为实验演示设定，非生产质量标准。可根据实际需求灵活调整。")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 FAIL 判定规则")
        max_minor = st.number_input("最大轻度缺陷数（超过则扣分）",
                                    min_value=1, max_value=50, value=config['max_minor_defects'])
        max_critical = st.number_input("最大严重缺陷数（超过则扣分）",
                                       min_value=0, max_value=20, value=config['max_critical_defects'])
        severity_thr = st.slider("严重等级阈值（≥此值视为严重缺陷）",
                                 min_value=1, max_value=5, value=config['fail_severity_threshold'])
        max_total_area = st.number_input("最大总缺陷面积占比 (%)",
                                         min_value=1.0, max_value=50.0, value=config['max_total_area_pct'], step=0.5)
        max_single_area = st.number_input("最大单缺陷面积占比 (%)",
                                          min_value=1.0, max_value=20.0, value=config['max_single_area_pct'], step=0.5)

    with col2:
        st.subheader("🟡 REVIEW 判定规则")
        review_conf = st.slider("低置信度阈值（低于则触发复核）",
                                min_value=0.1, max_value=0.9, value=config['review_low_confidence'], step=0.05)
        review_low_count = st.number_input("低置信度检测数阈值（≥则复核）",
                                           min_value=1, max_value=10, value=config['review_low_conf_count'])
        review_borderline = st.slider("临界评分区间（60~此值之间建议复核）",
                                      min_value=60, max_value=90, value=config['review_borderline_score'])

    # 缺陷严重等级说明
    st.markdown("---")
    st.subheader("📋 缺陷严重等级表")
    severity_data = [
        ['裂纹 (crazing)', 5, '结构性缺陷，最严重'],
        ['夹杂 (inclusion)', 4, '异物嵌入，严重'],
        ['划痕 (scratches)', 3, '表面损伤，中等'],
        ['斑块 (patches)', 3, '表面变色，中等'],
        ['麻点 (pitted_surface)', 2, '表面凹点，较轻'],
        ['氧化皮 (rolled-in_scale)', 2, '表面氧化，较轻'],
    ]
    df = pd.DataFrame(severity_data, columns=['缺陷类别', '严重等级(1-5)', '说明'])
    st.table(df)

    # 应用按钮
    if st.button("✅ 应用新规则", type="primary"):
        new_config = {
            'max_minor_defects': max_minor,
            'max_critical_defects': max_critical,
            'fail_severity_threshold': severity_thr,
            'review_low_confidence': review_conf,
            'review_low_conf_count': review_low_count,
            'review_borderline_score': review_borderline,
            'max_total_area_pct': max_total_area,
            'max_single_area_pct': max_single_area,
        }
        agent.rule_engine.config.update(new_config)
        st.success("✅ 规则已更新，即时生效！")
        st.balloons()


# ============================================================
# 模块：GAN 数据增强
# ============================================================
elif page == "🎨 GAN数据增强":
    st.markdown('<div class="main-header">🎨 GAN 缺陷数据增强</div>', unsafe_allow_html=True)

    st.info("💡 **条件 DCGAN** 生成 6 类工业表面缺陷样本，用于数据增强和模型训练。基于 NEU-DET 数据集训练，支持按类别条件生成。")

    # 加载 GAN 管理器
    @st.cache_resource(show_spinner="加载 GAN 模型中...")
    def load_gan_manager():
        from models.gan import create_gan_manager
        return create_gan_manager()

    gan = load_gan_manager()

    if not gan or not gan.is_loaded:
        st.error("❌ GAN 模型加载失败，请检查检查点路径")
    else:
        info = gan.get_model_info()

        # 模型信息卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("架构", info["architecture"].split(" ")[0])
        col2.metric("图像尺寸", f'{info["image_size"]}×{info["image_size"]}')
        col3.metric("类别数", info["num_classes"])
        col4.metric("生成总数", info["total_generated"])

        st.markdown("---")

        # 生成操作区
        st.subheader("🔧 图像生成")

        gen_tab1, gen_tab2 = st.tabs(["按类别生成", "全类别网格"])

        with gen_tab1:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                defect_class = st.selectbox(
                    "缺陷类别",
                    options=["crazing", "inclusion", "patches",
                             "pitted_surface", "rolled-in_scale", "scratches"],
                    format_func=lambda x: {
                        "crazing": "裂纹 (crazing)",
                        "inclusion": "夹杂 (inclusion)",
                        "patches": "斑块 (patches)",
                        "pitted_surface": "麻点 (pitted_surface)",
                        "rolled-in_scale": "氧化皮 (rolled-in_scale)",
                        "scratches": "划痕 (scratches)",
                    }[x],
                )
            with col_b:
                num_samples = st.slider("生成数量", min_value=1, max_value=32, value=8)
            with col_c:
                st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
                if st.button("🚀 生成样本", type="primary", use_container_width=True):
                    with st.spinner(f"正在生成 {num_samples} 张 {defect_class} 样本..."):
                        result = gan.generate(num_samples=num_samples, class_name=defect_class)
                        st.session_state.gan_result = result
                        st.session_state.gan_result_class = defect_class

            # 显示生成结果
            if 'gan_result' in st.session_state and st.session_state.get('gan_result_class') == defect_class:
                result = st.session_state.gan_result
                st.success(f"✅ 成功生成 {result['num_samples']} 张样本，耗时 {result['generate_time']*1000:.0f}ms")

                # 显示图像网格
                images = result["images"]
                cols = st.columns(min(8, len(images)))
                for i, img in enumerate(images):
                    with cols[i % len(cols)]:
                        st.image(img, caption=f"样本 {i+1}", use_container_width=True)

        with gen_tab2:
            col_d, col_e = st.columns(2)
            with col_d:
                spc = st.slider("每类样本数", min_value=2, max_value=16, value=6, key="spc_slider")
            with col_e:
                st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
                if st.button("🎨 生成网格图", type="primary", use_container_width=True, key="grid_btn"):
                    with st.spinner(f"生成 {spc}×6 类别网格图..."):
                        import tempfile
                        grid_result = gan.generate_grid(samples_per_class=spc)
                        st.session_state.gan_grid_img = grid_result["grid_image"]
                        st.session_state.gan_grid_count = grid_result["total_samples"]

            if 'gan_grid_img' in st.session_state:
                st.success(f"✅ 网格图生成完成，共 {st.session_state.gan_grid_count} 张样本")
                st.image(
                    st.session_state.gan_grid_img,
                    caption="6 类缺陷 GAN 生成样本（从上到下：裂纹、夹杂、斑块、麻点、氧化皮、划痕）",
                    use_container_width=True,
                )

        st.markdown("---")

        # GAN-YOLO 融合功能
        st.subheader("🔗 GAN-YOLO 融合增强")

        st.info("🔄 **4 种深度融合机制**：交叉验证检测、困难样本挖掘、自适应数据增强、融合质量判定。GAN 和 YOLO 不再是简单拼接，而是形成互相促进的闭环。")

        fusion_tab1, fusion_tab2, fusion_tab3 = st.tabs([
            "🔍 融合检测",
            "⛏️ 困难样本挖掘",
            "📈 自适应增强"
        ])

        with fusion_tab1:
            st.markdown("**GAN重建 + YOLO检测 双路验证**")
            st.caption("双模型交叉验证，降低漏检和误检率。YOLO检出+GAN异常=高可信缺陷；YOLO检出+GAN正常=可能误检；YOLO未检出+GAN异常=可能漏检")

            col_fd1, col_fd2 = st.columns(2)
            with col_fd1:
                fd_image = st.file_uploader(
                    "上传检测图像",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="fusion_detect_upload"
                )
                fd_threshold = st.slider(
                    "GAN异常阈值",
                    min_value=0.05, max_value=0.4, value=0.15, step=0.01,
                    help="重建误差超过阈值则判定为异常"
                )
            with col_fd2:
                st.markdown('<div style="height:60px"></div>', unsafe_allow_html=True)
                if st.button("🔍 开始融合检测", type="primary", use_container_width=True,
                             key="fusion_detect_btn"):
                    if fd_image is not None:
                        from models.fusion import create_fusion_manager
                        from models.gan import create_gan_manager as cgm

                        @st.cache_resource(show_spinner="加载融合模型...")
                        def load_fusion():
                            g = cgm()
                            from ultralytics import YOLO
                            y = YOLO(st.session_state.get('model_path',
                                        'outputs/baseline/yolov8s_baseline-2/weights/best.pt'))
                            return create_fusion_manager(gan_manager=g, yolo_model=y)

                        fusion = load_fusion()
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                            f.write(fd_image.read())
                            tmp_path = f.name

                        with st.spinner("执行融合检测..."):
                            result = fusion.fused_quality_judgement(tmp_path)
                            st.session_state.fusion_result = result
                            st.session_state.fusion_img_path = tmp_path

            if 'fusion_result' in st.session_state:
                r = st.session_state.fusion_result
                quality_color = {
                    "PASS": "#28A745",
                    "REVIEW": "#FFC107",
                    "FAIL": "#DC3545",
                }.get(r["quality"], "#6C757D")

                # 结果卡片
                col_q, col_c = st.columns(2)
                col_q.metric("融合质量", r["quality_cn"], delta=None)
                col_c.metric("融合置信度", f"{r['confidence']:.0%}")

                st.info(f"**判定说明**：{' / '.join(r['cross_validation']['notes'])}")

                # 详细数据
                with st.expander("📊 详细数据", expanded=True):
                    col_d1, col_d2, col_d3 = st.columns(3)
                    col_d1.metric("YOLO 检测数", r["detection_count"])
                    col_d2.metric("YOLO 置信度", f"{r['cross_validation']['yolo_max_confidence']:.3f}")
                    col_d3.metric("GAN 重建误差", f"{r['cross_validation']['gan_reconstruction_error']:.4f}")

                    st.caption(f"融合方式: {r['fusion_method']} | "
                               f"信息源: {', '.join(r['sources_used'])} | "
                               f"耗时: {r['processing_time_ms']}ms")

        with fusion_tab2:
            st.markdown("**GAN生成 + YOLO检测 = 困难样本挖掘**")
            st.caption("自动筛选出低置信度、误检、漏检的困难样本，用于针对性模型训练提升")

            col_hs1, col_hs2 = st.columns(2)
            with col_hs1:
                hs_num = st.slider("每类生成数量", min_value=5, max_value=50, value=15, key="hs_num")
            with col_hs2:
                st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
                if st.button("⛏️ 开始挖掘", type="primary", use_container_width=True, key="hs_btn"):
                    from models.fusion import create_fusion_manager
                    from models.gan import create_gan_manager as cgm

                    @st.cache_resource(show_spinner="加载融合模型...")
                    def load_fusion_hs():
                        g = cgm()
                        from ultralytics import YOLO
                        y = YOLO('outputs/baseline/yolov8s_baseline-2/weights/best.pt')
                        return create_fusion_manager(gan_manager=g, yolo_model=y)

                    fusion = load_fusion_hs()

                    with st.spinner(f"挖掘中: 生成 {hs_num*6} 张样本并检测..."):
                        result = fusion.mine_hard_samples(num_per_class=hs_num)
                        st.session_state.hs_result = result

            if 'hs_result' in st.session_state:
                r = st.session_state.hs_result
                st.success(f"✅ 挖掘完成: {r['hard_samples_count']} 个困难样本 ({r['hard_ratio']:.1%})")

                # 难度分布
                st.markdown("**难度分布**")
                diff_data = r["difficulty_distribution"]
                st.bar_chart(diff_data)

                # 各类别困难比例
                st.markdown("**各类别困难比例**")
                class_hard_data = {
                    name: s["hard_ratio"]
                    for name, s in r["per_class_stats"].items()
                }
                st.bar_chart(class_hard_data)

                # 统计表
                st.dataframe({
                    "类别": list(r["per_class_stats"].keys()),
                    "总数": [s["total"] for s in r["per_class_stats"].values()],
                    "困难数": [s["hard"] for s in r["per_class_stats"].values()],
                    "困难率": [f"{s['hard_ratio']:.0%}" for s in r["per_class_stats"].values()],
                }, use_container_width=True, hide_index=True)

        with fusion_tab3:
            st.markdown("**分析类别表现 → 薄弱类别定向增强**")
            st.caption("表现差的类别生成更多样本，实现智能数据增强，最高可达 5 倍基础数量")

            col_aa1, col_aa2, col_aa3 = st.columns(3)
            with col_aa1:
                aa_base = st.number_input("基础生成量", min_value=10, max_value=500, value=50, step=10)
            with col_aa2:
                aa_ratio = st.slider("最大倍率", min_value=1.5, max_value=10.0, value=5.0, step=0.5)
            with col_aa3:
                st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
                if st.button("📈 自适应生成", type="primary", use_container_width=True, key="aa_btn"):
                    from models.fusion import create_fusion_manager, GanYoloFusion
                    from models.gan import create_gan_manager as cgm

                    @st.cache_resource(show_spinner="加载融合模型...")
                    def load_fusion_aa():
                        g = cgm()
                        from ultralytics import YOLO
                        y = YOLO('outputs/baseline/yolov8s_baseline-2/weights/best.pt')
                        return create_fusion_manager(gan_manager=g, yolo_model=y)

                    fusion = load_fusion_aa()

                    # 演示用模拟性能报告
                    mock_perf = {
                        'per_class': {
                            'crazing': {'performance_score': 0.45},
                            'inclusion': {'performance_score': 0.85},
                            'patches': {'performance_score': 0.72},
                            'pitted_surface': {'performance_score': 0.68},
                            'rolled-in_scale': {'performance_score': 0.78},
                            'scratches': {'performance_score': 0.38},
                        }
                    }

                    with st.spinner("执行自适应增强..."):
                        result = fusion.adaptive_augmentation(
                            performance_report=mock_perf,
                            base_samples_per_class=aa_base,
                            max_ratio=aa_ratio,
                        )
                        st.session_state.aa_result = result
                        st.session_state.aa_perf = mock_perf

            if 'aa_result' in st.session_state:
                r = st.session_state.aa_result
                perf = st.session_state.aa_perf
                st.success(f"✅ 自适应增强完成，共生成 {r['total_generated']} 张样本")

                # 性能评分 vs 增强数量对比
                aug_data = {}
                for cls, plan in r.get('results', {}).items():
                    perf_score = perf['per_class'].get(cls, {}).get('performance_score', 0)
                    aug_data[cls] = {
                        '性能评分': perf_score,
                        '增强数量': plan.get('actual_generated', plan.get('planned', 0)),
                    }

                # 显示增强计划
                st.markdown("**各类别增强计划**")
                plan_df = {
                    "类别": list(r.get('plan', {}).keys()),
                    "性能评分": [round(v.get('performance_score', 0), 3)
                               for v in r.get('plan', {}).values()],
                    "增强倍率": [v.get('augmentation_ratio', 0)
                               for v in r.get('plan', {}).values()],
                    "计划生成": [v.get('planned_samples', 0)
                               for v in r.get('plan', {}).values()],
                }
                st.dataframe(plan_df, use_container_width=True, hide_index=True)

                st.caption(f"方法: {r.get('method', '')} | 输出目录: {r.get('output_dir', '')}")

        st.markdown("---")

        # 融合架构图
        st.subheader("🏗️ 融合架构")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────┐
        │              GAN 生成模型                        │
        │  (条件生成 / 重建分析 / 多样性增强)              │
        └─────────┬───────────────────────┬───────────────┘
                  │                       │
         ┌────────▼────────┐    ┌─────────▼────────┐
         │ ① 定向数据增强  │    │ ② 交叉验证检测    │
         │  (薄弱类别)     │    │  (双路验证)       │
         └────────┬────────┘    └─────────┬────────┘
                  │                       │
         ┌────────▼────────┐    ┌─────────▼────────┐
         │ ③ 困难样本挖掘  │    │ ④ 融合质量判定    │
         │  (难例筛选)     │    │  (多源融合)       │
         └────────┬────────┘    └─────────┬────────┘
                  │                       │
        ┌─────────▼───────────────────────▼───────────────┐
        │              YOLO 检测模型                        │
        │   (训练增强 / 检测验证 / 性能提升)                │
        └─────────────────────────────────────────────────┘
        ```
        """)

        st.markdown("---")

        # 检查点信息
        st.subheader("📁 检查点信息")
        from models.gan import list_available_checkpoints
        ckpts = list_available_checkpoints()
        ckpt_df = {"名称": [c["name"] for c in ckpts], "大小 (MB)": [c["size_mb"] for c in ckpts]}
        st.dataframe(ckpt_df, use_container_width=True, hide_index=True)

        st.caption("💡 提示：GAN 生成的缺陷样本可用于扩充训练数据集，提升检测模型的泛化能力和鲁棒性。")


# ============================================================
# 模块5：报告中心（v2 版）
# ============================================================
elif page == "📋 报告中心":
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'pages'))

    agent = load_agent()
    report_gen = load_report_generator()
    review_workflow = load_review_workflow()

    from pages.report_page import render_report_page
    render_report_page(agent, report_gen, review_workflow, BASE_DIR)


# ============================================================
# 模块6：复核管理
# ============================================================
elif page == "📝 复核管理":
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'pages'))

    review_workflow = load_review_workflow()

    from pages.review_page import render_review_page
    render_review_page(review_workflow, BASE_DIR)


# 页脚
st.markdown("---")
st.caption("🔍 基于 GAN-YOLO 与 AI Agent 的工业表面缺陷智能检测及质量分析系统 | 演示版本 v2")
