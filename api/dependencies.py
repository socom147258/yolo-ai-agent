# -*- coding: utf-8 -*-
"""
依赖注入模块

负责模型加载、引擎初始化、数据库连接等共享资源的管理。
使用 FastAPI 的依赖注入机制，确保资源在应用启动时初始化，
请求处理时复用。
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.agent import (
    QualityEngineV2,
    QualityRuleSet,
    ReviewWorkflow,
    AgentExecutorV2,
    AgentConfig,
    ExecutionTrace,
    CLASS_NAMES_CN,
)
from models.agent.batch_processor import BatchProcessor
from models.agent.report_generator_v2 import QualityReportGenerator
from models.agent.llm_integration import load_llm_config_from_env


# ============================================================
# 全局单例（应用启动时初始化）
# ============================================================

_yolo_model = None
_quality_engine: Optional[QualityEngineV2] = None
_review_workflow: Optional[ReviewWorkflow] = None
_agent_executor: Optional[AgentExecutorV2] = None
_batch_processor: Optional[BatchProcessor] = None
_gan_manager: Optional[Any] = None
_llm_config: Optional[Dict[str, Any]] = None
_model_version: str = "unknown"
_config_loaded: bool = False


# ============================================================
# 配置加载
# ============================================================

def _load_env_config() -> Dict[str, Any]:
    """
    从 .env 文件和环境变量加载配置

    Returns:
        配置字典
    """
    global _config_loaded, _llm_config

    # 加载 .env 文件
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass
    except Exception:
        pass

    # 加载 LLM 配置
    _llm_config = load_llm_config_from_env()

    _config_loaded = True
    return get_config()


def get_config() -> Dict[str, Any]:
    """
    获取当前系统配置

    Returns:
        配置字典
    """
    if not _config_loaded:
        _load_env_config()

    model_path = os.environ.get(
        'YOLO_MODEL_PATH',
        str(PROJECT_ROOT / 'outputs' / 'baseline' / 'yolov8s_baseline-2' / 'weights' / 'best.pt')
    )
    data_path = os.environ.get(
        'DATA_PATH',
        str(PROJECT_ROOT / 'data' / 'raw' / 'NEU-DET' / 'IMAGES')
    )
    output_dir = os.environ.get(
        'OUTPUT_DIR',
        str(PROJECT_ROOT / 'outputs' / 'api')
    )
    review_db_path = os.environ.get(
        'REVIEW_DB_PATH',
        str(PROJECT_ROOT / 'outputs' / 'api' / 'review.db')
    )
    trace_db_path = os.environ.get(
        'TRACE_DB_PATH',
        str(PROJECT_ROOT / 'outputs' / 'api' / 'trace.db')
    )
    rule_id = os.environ.get('RULE_ID', 'demo_default')
    rule_version = os.environ.get('RULE_VERSION', '1.0.0')
    gan_enabled = os.environ.get('GAN_ENABLED', 'true').lower() in ('true', '1', 'yes')
    gan_checkpoint = os.environ.get(
        'GAN_CHECKPOINT',
        str(PROJECT_ROOT / 'outputs' / 'checkpoints' / 'dcgan' / 'best_model.pth')
    )

    llm_provider = _llm_config.get('provider', 'none') if _llm_config else 'none'
    llm_model = _llm_config.get('model', '') if _llm_config else ''
    llm_available = llm_provider not in ('none', '')

    return {
        'model_path': model_path,
        'data_path': data_path,
        'output_dir': output_dir,
        'review_db_path': review_db_path,
        'trace_db_path': trace_db_path,
        'rule_id': rule_id,
        'rule_version': rule_version,
        'llm_provider': llm_provider,
        'llm_model': llm_model,
        'llm_available': llm_available,
        'llm_config': _llm_config or {},
        'gan_enabled': gan_enabled,
        'gan_checkpoint': gan_checkpoint,
    }


# ============================================================
# YOLO 模型
# ============================================================

def get_yolo_model():
    """
    获取 YOLO 模型实例（懒加载单例）

    Returns:
        YOLO 模型实例
    """
    global _yolo_model, _model_version

    if _yolo_model is None:
        config = get_config()
        model_path = config['model_path']

        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(model_path)

            # 提取模型版本信息
            model_name = Path(model_path).stem
            parent_dir = Path(model_path).parent.parent.name
            _model_version = f"{parent_dir}/{model_name}"
        except Exception as e:
            raise RuntimeError(f"加载 YOLO 模型失败: {e}")

    return _yolo_model


def get_model_version() -> str:
    """获取模型版本标识"""
    global _model_version
    if _model_version == "unknown":
        try:
            get_yolo_model()  # 触发加载
        except Exception:
            pass
    return _model_version


# ============================================================
# 质量判定引擎
# ============================================================

def get_quality_engine() -> QualityEngineV2:
    """
    获取质量判定引擎实例（单例）

    Returns:
        QualityEngineV2 实例
    """
    global _quality_engine

    if _quality_engine is None:
        config = get_config()
        rule_set = QualityRuleSet.default_demo()
        rule_set.rule_id = config['rule_id']
        rule_set.version = config['rule_version']
        _quality_engine = QualityEngineV2(rule_set)

    return _quality_engine


# ============================================================
# 复核工作流
# ============================================================

def get_review_workflow() -> ReviewWorkflow:
    """
    获取复核工作流实例（单例）

    Returns:
        ReviewWorkflow 实例
    """
    global _review_workflow

    if _review_workflow is None:
        config = get_config()
        db_path = config['review_db_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _review_workflow = ReviewWorkflow(db_path)

    return _review_workflow


# ============================================================
# Agent 执行器
# ============================================================

def get_agent_executor() -> AgentExecutorV2:
    """
    获取 Agent 执行器实例（单例）

    Returns:
        AgentExecutorV2 实例
    """
    global _agent_executor

    if _agent_executor is None:
        config = get_config()
        llm_cfg = config.get('llm_config', {})

        agent_config = AgentConfig(
            llm_provider=config.get('llm_provider', 'none'),
            llm_model=config.get('llm_model', ''),
            llm_api_key=llm_cfg.get('api_key', ''),
            llm_base_url=llm_cfg.get('base_url', ''),
            rule_id=config['rule_id'],
            rule_version=config['rule_version'],
            report_output_dir=str(Path(config['output_dir']) / 'reports'),
            gan_enabled=config.get('gan_enabled', True),
            gan_checkpoint_path=config.get('gan_checkpoint', ''),
        )

        try:
            model = get_yolo_model()
        except Exception:
            model = None

        _agent_executor = AgentExecutorV2(
            config=agent_config,
            model=model,
            db_path=config['trace_db_path'],
        )

    return _agent_executor


# ============================================================
# 批次处理器
# ============================================================

def get_batch_processor() -> BatchProcessor:
    """
    获取批次处理器实例（单例）

    Returns:
        BatchProcessor 实例
    """
    global _batch_processor

    if _batch_processor is None:
        quality_engine = get_quality_engine()
        review_workflow = get_review_workflow()
        model_version = get_model_version()

        _batch_processor = BatchProcessor(
            quality_engine=quality_engine,
            review_workflow=review_workflow,
            model_version=model_version,
        )

    return _batch_processor


# ============================================================
# 报告生成器
# ============================================================

def get_report_generator() -> QualityReportGenerator:
    """
    获取报告生成器（每次新建，因为一个生成器只能生成一份报告）

    Returns:
        新的 QualityReportGenerator 实例
    """
    return QualityReportGenerator()


# ============================================================
# GAN 管理器
# ============================================================

def get_gan_manager():
    """
    获取 GAN 管理器实例（懒加载单例）

    Returns:
        GANManager 实例，如果未启用则返回 None
    """
    global _gan_manager

    config = get_config()
    if not config.get('gan_enabled', True):
        return None

    if _gan_manager is None:
        try:
            from models.gan import create_gan_manager
            ckpt = config.get('gan_checkpoint', '') or None
            _gan_manager = create_gan_manager(checkpoint_path=ckpt)
        except Exception as e:
            print(f"[GAN] 加载失败: {e}")
            _gan_manager = None

    return _gan_manager


# ============================================================
# 类别中文名映射
# ============================================================

def get_class_name_cn(cls_name: str) -> str:
    """获取缺陷类别的中文名"""
    return CLASS_NAMES_CN.get(cls_name, cls_name)


# ============================================================
# 初始化所有资源（应用启动时调用）
# ============================================================

def init_all_resources():
    """
    初始化所有资源（在应用启动事件中调用）
    提前加载模型和引擎，避免首次请求时的延迟
    """
    config = get_config()

    # 确保输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'reports'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'uploads'), exist_ok=True)

    # 初始化质量引擎
    get_quality_engine()

    # 初始化复核工作流（创建数据库表）
    get_review_workflow()

    # 尝试加载 YOLO 模型（失败不影响启动，只是首次检测会慢）
    try:
        get_yolo_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    # 初始化批次处理器
    try:
        get_batch_processor()
    except Exception:
        pass

    # 初始化 Agent 执行器
    try:
        get_agent_executor()
    except Exception:
        pass

    return {
        'model_loaded': model_loaded,
        'llm_available': config['llm_available'],
        'model_version': get_model_version(),
    }
