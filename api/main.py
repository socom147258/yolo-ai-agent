# -*- coding: utf-8 -*-
"""
YOLO 缺陷质检系统 - FastAPI Web API 主应用

提供完整的 RESTful API 接口，包括：
- 图像检测与质量判定
- 批次处理
- 复核工作流
- Agent 智能对话
- 报告生成与下载
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.requests import Request

from api.schemas import (
    QualityJudgeRequest,
    ReviewConfirmRequest,
    ReviewRejectRequest,
    AgentChatRequest,
)
from api.dependencies import (
    get_config,
    get_yolo_model,
    get_quality_engine,
    get_review_workflow,
    get_agent_executor,
    get_batch_processor,
    get_report_generator,
    get_model_version,
    get_gan_manager,
    init_all_resources,
)
from api.utils import (
    success_response,
    error_response,
    save_upload_file,
    save_upload_files,
    detections_to_list,
    image_info_to_dict,
    quality_result_to_dict,
    batch_result_to_dict,
    review_task_to_dict,
    build_task_context_from_detections,
    generate_report_id,
    get_report_path,
)

from models.agent import (
    TaskContext,
    DetectionItem,
    ImageInfo,
    CLASS_NAMES_CN,
)
from models.agent.batch_processor import BatchProcessor, detect_with_yolo


# ============================================================
# FastAPI 应用实例
# ============================================================

app = FastAPI(
    title="YOLO 缺陷质检系统 API",
    description="基于 YOLOv8 的表面缺陷智能质检系统 RESTful API",
    version="1.0.0",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 上传文件大小限制（50MB）
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


# ============================================================
# 应用生命周期事件
# ============================================================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化资源"""
    try:
        result = init_all_resources()
        print(f"[API] 服务启动成功")
        print(f"[API] 模型加载: {'成功' if result['model_loaded'] else '失败'}")
        print(f"[API] LLM 可用: {'是' if result['llm_available'] else '否'}")
        print(f"[API] 模型版本: {result['model_version']}")
    except Exception as e:
        print(f"[API] 启动警告: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("[API] 服务关闭")


# ============================================================
# 中间件：请求大小限制
# ============================================================

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """限制请求体大小"""
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content=error_response(
                    code=413,
                    message=f"上传文件过大，最大允许 {MAX_UPLOAD_SIZE // (1024*1024)}MB"
                ),
            )
    response = await call_next(request)
    return response


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health", summary="健康检查")
async def health_check():
    """
    检查服务运行状态、模型加载状态和 LLM 可用性
    """
    try:
        config = get_config()
        model_loaded = False
        try:
            get_yolo_model()
            model_loaded = True
        except Exception:
            model_loaded = False

        data = {
            "status": "ok",
            "version": "1.0.0",
            "model_loaded": model_loaded,
            "llm_available": config["llm_available"],
        }
        return success_response(data)
    except Exception as e:
        return error_response(code=500, message=f"健康检查失败: {str(e)}")


# ============================================================
# 系统配置
# ============================================================

@app.get("/api/config", summary="获取系统配置")
async def get_system_config():
    """
    获取当前系统配置信息，包括模型版本、规则版本、LLM 状态等
    """
    try:
        config = get_config()
        quality_engine = get_quality_engine()
        rule_set = quality_engine.rule_set

        data = {
            "model_version": get_model_version(),
            "model_path": config["model_path"],
            "rule_id": rule_set.rule_id,
            "rule_version": rule_set.version,
            "rule_source": rule_set.source,
            "is_demo_rule": rule_set.is_demo,
            "llm_provider": config["llm_provider"],
            "llm_model": config["llm_model"],
            "llm_available": config["llm_available"],
            "data_path": config["data_path"],
            "output_dir": config["output_dir"],
            "review_db_path": config["review_db_path"],
            "trace_db_path": config["trace_db_path"],
            "class_names": CLASS_NAMES_CN,
        }
        return success_response(data)
    except Exception as e:
        return error_response(code=500, message=f"获取配置失败: {str(e)}")


# ============================================================
# 单图检测
# ============================================================

@app.post("/api/detect", summary="单图检测")
async def detect_single_image(
    image: UploadFile = File(..., description="待检测的图像文件"),
    with_quality: bool = Query(True, description="是否同时进行质量判定"),
):
    """
    上传单张图像进行缺陷检测，并返回检测结果和质量判定

    - **image**: 待检测的图像文件
    - **with_quality**: 是否同时执行质量判定（默认 true）
    """
    try:
        config = get_config()
        output_dir = os.path.join(config["output_dir"], "uploads")

        # 保存上传文件
        image_path, original_name = save_upload_file(image, output_dir)

        # 加载模型
        model = get_yolo_model()
        quality_engine = get_quality_engine()
        review_workflow = get_review_workflow()

        # 执行检测
        t0 = time.time()
        detections, image_info, error_msg = detect_with_yolo(model, image_path)
        detection_time_ms = (time.time() - t0) * 1000

        if error_msg:
            return error_response(code=500, message=f"检测失败: {error_msg}")

        # 质量判定
        quality_time_ms = 0.0
        quality_result_dict = {}
        review_task_id = ""

        if with_quality:
            t1 = time.time()
            ctx = TaskContext(task_id=Path(image_path).stem)
            ctx.model_version = get_model_version()
            ctx.set_raw_detections(detections, image_info)
            quality_result = quality_engine.evaluate(ctx, scope="full")
            quality_time_ms = (time.time() - t1) * 1000
            quality_result_dict = quality_result_to_dict(quality_result)

            # 如果是 FAIL 或 REVIEW，自动创建复核任务
            if quality_result["quality"] in ("FAIL", "REVIEW"):
                try:
                    review_task_id = review_workflow.create_task(
                        task_id=f"single_{Path(original_name).stem}",
                        record_id=abs(hash(original_name)) % (2**31),
                        image_name=original_name,
                        image_path=image_path,
                        original_quality=quality_result["quality"],
                        original_score=quality_result.get("score", 0),
                        original_scope=quality_result.get("scope_label", "单图全量"),
                        reason=quality_result.get("forced_reason", f"自动检测结果为{quality_result['quality']}"),
                        quality_issue=f"{quality_result['quality']}（需人工确认）",
                        priority="high" if quality_result["quality"] == "FAIL" else "medium",
                        model_version=get_model_version(),
                        rule_version=f"{quality_engine.rule_set.rule_id}_v{quality_engine.rule_set.version}",
                        original_detections=[d.to_tuple() for d in detections],
                        original_eval=quality_result,
                    )
                except Exception:
                    pass  # 复核创建失败不影响检测结果返回

        data = {
            "image_name": original_name,
            "image_info": image_info_to_dict(image_info),
            "detections": detections_to_list(detections),
            "detection_count": len(detections),
            "quality_result": quality_result_dict,
            "detection_time_ms": round(detection_time_ms, 2),
            "quality_time_ms": round(quality_time_ms, 2),
            "review_id": review_task_id,
        }
        return success_response(data)

    except Exception as e:
        return error_response(code=500, message=f"检测失败: {str(e)}")


# ============================================================
# 质量判定（给定检测结果）
# ============================================================

@app.post("/api/quality/judge", summary="质量判定")
async def quality_judge(request: QualityJudgeRequest):
    """
    给定检测结果，执行质量判定

    - **detections**: 检测结果列表
    - **image_width**: 图像宽度
    - **image_height**: 图像高度
    - **image_quality_score**: 图像质量评分（可选）
    """
    try:
        quality_engine = get_quality_engine()

        # 构建任务上下文
        # Pydantic 模型转为字典列表
        det_dicts = [d.model_dump() for d in request.detections]
        ctx = build_task_context_from_detections(
            detections=det_dicts,
            image_width=request.image_width,
            image_height=request.image_height,
            image_quality_score=request.image_quality_score,
        )

        # 执行判定
        quality_result = quality_engine.evaluate(ctx, scope="full")
        result = quality_result_to_dict(quality_result)

        # 覆盖 scope 标签（如果请求中指定了）
        if request.scope_label:
            result["scope_label"] = request.scope_label
        if request.scope:
            result["scope"] = request.scope

        return success_response(result)

    except Exception as e:
        return error_response(code=500, message=f"质量判定失败: {str(e)}")


# ============================================================
# 批次检测
# ============================================================

@app.post("/api/batch/detect", summary="批次检测")
async def detect_batch(
    images: List[UploadFile] = File(..., description="待检测的图像文件列表"),
):
    """
    上传多张图像进行批量检测和质量判定

    - **images**: 图像文件列表（支持多文件上传）
    """
    try:
        config = get_config()
        output_dir = os.path.join(config["output_dir"], "uploads")

        if not images:
            return error_response(code=400, message="请至少上传一张图像")

        # 保存所有上传文件
        image_paths = []
        for upload_file in images:
            path, _ = save_upload_file(upload_file, output_dir)
            image_paths.append(path)

        # 获取批次处理器
        batch_processor = get_batch_processor()
        model = get_yolo_model()

        # 定义检测函数（闭包捕获 model）
        def detect_func(img_path):
            return detect_with_yolo(model, img_path)

        # 执行批次处理
        batch_result = batch_processor.process_batch(image_paths, detect_func)

        # 转换为响应格式
        result = batch_result_to_dict(batch_result)

        # 提升 summary 字段到顶级，方便前端使用
        summary = result.get("summary", {})
        result["total_planned"] = summary.get("planned", 0)
        result["total_success"] = summary.get("success", 0)
        result["total_failed"] = summary.get("failed", 0)
        result["pass_count"] = summary.get("pass_count", 0)
        result["fail_count"] = summary.get("fail_count", 0)
        result["review_count"] = summary.get("review_count", 0)
        result["image_results"] = result.get("images", [])

        return success_response(result)

    except Exception as e:
        return error_response(code=500, message=f"批次检测失败: {str(e)}")


# ============================================================
# 生成批次报告
# ============================================================

@app.post("/api/batch/{batch_id}/report", summary="生成批次报告")
async def generate_batch_report(
    batch_id: str,
):
    """
    为指定批次生成 Word 格式的质检报告

    - **batch_id**: 批次ID
    """
    try:
        config = get_config()

        # 注意：实际使用中应该从数据库或缓存中获取批次结果
        # 这里提供一个简化的实现，演示报告生成流程
        # 生产环境应从持久化存储中加载批次结果

        report_gen = get_report_generator()
        report_id = generate_report_id("BATCH")
        report_path = get_report_path(report_id, config["output_dir"])

        # 构建模拟的批次结果（实际应从数据库读取）
        # 这里使用一个简化的批次结果作为示例
        batch_result = {
            "batch_id": batch_id,
            "summary": {
                "planned": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "pass": 0,
                "fail": 0,
                "review": 0,
            },
            "batch_quality": "INCOMPLETE",
            "batch_quality_reason": "报告功能需配合批次结果持久化使用",
            "model_version": get_model_version(),
            "rule_version": f"{get_quality_engine().rule_set.rule_id}_v{get_quality_engine().rule_set.version}",
            "rule_is_demo": get_quality_engine().rule_set.is_demo,
            "start_time": "",
            "end_time": "",
            "images": [],
        }

        trace_info = {
            "task_id": batch_id,
            "model_version": get_model_version(),
            "rule_version": batch_result["rule_version"],
            "total_steps": 0,
            "success_steps": 0,
            "total_time_ms": 0,
        }

        # 生成报告
        report_gen.generate_batch_report(batch_result, trace_info)
        report_gen.save(report_path)

        file_size = os.path.getsize(report_path) if os.path.exists(report_path) else 0

        data = {
            "report_id": report_id,
            "report_path": report_path,
            "report_type": "batch",
            "file_size": file_size,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return success_response(data)

    except Exception as e:
        return error_response(code=500, message=f"生成报告失败: {str(e)}")


# ============================================================
# 复核任务列表
# ============================================================

@app.get("/api/review/tasks", summary="复核任务列表")
async def list_review_tasks(
    status: Optional[str] = Query(None, description="按状态筛选: pending/confirmed/rejected/corrected/escalated"),
    priority: Optional[str] = Query(None, description="按优先级筛选: low/normal/high/urgent"),
    limit: int = Query(50, description="返回数量限制", ge=1, le=200),
):
    """
    获取复核任务列表，支持按状态和优先级筛选

    - **status**: 任务状态筛选
    - **priority**: 优先级筛选
    - **limit**: 返回数量限制
    """
    try:
        review_workflow = get_review_workflow()

        tasks = review_workflow.list_tasks(
            status=status,
            priority=priority,
            limit=limit,
        )

        # 转换格式
        task_list = [review_task_to_dict(t) for t in tasks]

        data = {
            "tasks": task_list,
            "total": len(task_list),
            "limit": limit,
        }
        return success_response(data)

    except Exception as e:
        return error_response(code=500, message=f"获取复核任务列表失败: {str(e)}")


# ============================================================
# 复核任务详情
# ============================================================

@app.get("/api/review/tasks/{review_id}", summary="复核任务详情")
async def get_review_task(review_id: str):
    """
    获取单个复核任务的详细信息

    - **review_id**: 复核任务ID
    """
    try:
        review_workflow = get_review_workflow()

        task = review_workflow.get_task(review_id)
        if not task:
            return error_response(code=404, message=f"复核任务不存在: {review_id}")

        # 获取操作历史
        action_history = review_workflow.get_action_history(review_id)

        data = {
            "task": review_task_to_dict(task),
            "action_history": action_history,
        }
        return success_response(data)

    except Exception as e:
        return error_response(code=500, message=f"获取复核任务详情失败: {str(e)}")


# ============================================================
# 确认复核
# ============================================================

@app.post("/api/review/tasks/{review_id}/confirm", summary="确认复核")
async def confirm_review(review_id: str, request: ReviewConfirmRequest):
    """
    确认复核结果（维持原判定）

    - **review_id**: 复核任务ID
    - **actor**: 操作人
    - **role**: 角色（需要 can_confirm 权限）
    - **note**: 备注
    """
    try:
        review_workflow = get_review_workflow()

        # 检查任务是否存在
        task = review_workflow.get_task(review_id)
        if not task:
            return error_response(code=404, message=f"复核任务不存在: {review_id}")

        # 执行确认
        success = review_workflow.confirm(
            review_id=review_id,
            actor=request.actor,
            role=request.role,
            note=request.note,
        )

        if success:
            # 获取更新后的任务
            updated_task = review_workflow.get_task(review_id)
            return success_response({
                "review_id": review_id,
                "status": "confirmed",
                "task": review_task_to_dict(updated_task) if updated_task else None,
            })
        else:
            return error_response(code=500, message="确认失败")

    except Exception as e:
        return error_response(code=500, message=f"确认复核失败: {str(e)}")


# ============================================================
# 驳回复核
# ============================================================

@app.post("/api/review/tasks/{review_id}/reject", summary="驳回复核")
async def reject_review(review_id: str, request: ReviewRejectRequest):
    """
    驳回复核结果（修正质量判定）

    - **review_id**: 复核任务ID
    - **actor**: 操作人
    - **role**: 角色（需要 can_reject 权限，仅 inspector/supervisor）
    - **reason**: 驳回原因
    - **corrected_quality**: 修正后的质量等级
    """
    try:
        review_workflow = get_review_workflow()

        # 检查任务是否存在
        task = review_workflow.get_task(review_id)
        if not task:
            return error_response(code=404, message=f"复核任务不存在: {review_id}")

        # 执行驳回
        success = review_workflow.reject(
            review_id=review_id,
            actor=request.actor,
            reason=request.reason,
            corrected_quality=request.corrected_quality,
            role=request.role,
        )

        if success:
            # 获取更新后的任务
            updated_task = review_workflow.get_task(review_id)
            return success_response({
                "review_id": review_id,
                "status": "rejected",
                "corrected_quality": request.corrected_quality,
                "task": review_task_to_dict(updated_task) if updated_task else None,
            })
        else:
            return error_response(code=500, message="驳回失败")

    except Exception as e:
        return error_response(code=500, message=f"驳回复核失败: {str(e)}")


# ============================================================
# Agent 对话
# ============================================================

@app.post("/api/agent/chat", summary="Agent 对话")
async def agent_chat(
    message: str = Form(..., description="用户自然语言消息"),
    task_id: Optional[str] = Form(None, description="可选的任务ID（用于多轮对话）"),
    image: Optional[UploadFile] = File(None, description="可选的图像文件"),
):
    """
    与质检 Agent 进行自然语言对话，支持上传图像进行智能分析

    - **message**: 用户自然语言消息
    - **task_id**: 可选的任务ID（用于多轮对话）
    - **image**: 可选的图像文件
    """
    try:
        config = get_config()
        agent_executor = get_agent_executor()

        image_path = None
        if image:
            output_dir = os.path.join(config["output_dir"], "uploads")
            image_path, _ = save_upload_file(image, output_dir)

        # 执行 Agent 任务
        result = agent_executor.run_task(
            user_input=message,
            image_path=image_path,
            task_id=task_id,
        )

        # 提取结果
        trace = result.get("trace", {})
        final_quality = result.get("final_quality", "")
        final_scope = result.get("final_scope", "")

        # 生成回复文本
        reply = result.get("reply", "")
        if not reply and trace:
            reply = trace.get("final_conclusion", "分析完成")

        data = {
            "task_id": result.get("task_id", ""),
            "reply": reply,
            "final_quality": final_quality,
            "final_scope": final_scope,
            "total_steps": trace.get("total_steps", 0) if trace else 0,
            "total_time_ms": trace.get("total_time_ms", 0) if trace else 0,
            "llm_fallback": trace.get("llm_fallback", False) if trace else True,
            "llm_fallback_reason": trace.get("llm_fallback_reason", "") if trace else "Agent 未初始化",
            "steps": trace.get("steps", []) if trace else [],
        }
        return success_response(data)

    except Exception as e:
        return error_response(code=500, message=f"Agent 对话失败: {str(e)}")


# ============================================================
# 下载报告
# ============================================================

@app.get("/api/report/{report_id}/download", summary="下载报告")
async def download_report(report_id: str):
    """
    下载指定的质检报告（Word 格式）

    - **report_id**: 报告ID
    """
    try:
        config = get_config()
        report_path = get_report_path(report_id, config["output_dir"])

        if not os.path.exists(report_path):
            return error_response(code=404, message=f"报告不存在: {report_id}")

        filename = f"{report_id}.docx"
        return FileResponse(
            path=report_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    except Exception as e:
        return error_response(code=500, message=f"下载报告失败: {str(e)}")


# ============================================================
# GAN 生成相关接口
# ============================================================

@app.get("/api/gan/info", summary="GAN模型信息")
async def gan_info():
    """
    获取当前 GAN 模型的基本信息
    """
    try:
        gan = get_gan_manager()
        if gan is None:
            return success_response(data={
                "enabled": False,
                "loaded": False,
                "message": "GAN 功能未启用"
            })
        info = gan.get_model_info()
        return success_response(data=info)
    except Exception as e:
        return error_response(code=500, message=f"获取 GAN 信息失败: {str(e)}")


@app.post("/api/gan/generate", summary="GAN生成缺陷图像")
async def gan_generate(
    defect_class: str = Form(..., description="缺陷类别"),
    num_samples: int = Form(4, ge=1, le=64, description="生成数量"),
):
    """
    使用条件 GAN 生成指定类别的缺陷样本图像

    - **defect_class**: 缺陷类别 (crazing/inclusion/patches/pitted_surface/rolled-in_scale/scratches)
    - **num_samples**: 生成数量 (1-64)
    """
    try:
        gan = get_gan_manager()
        if gan is None or not gan.is_loaded:
            return error_response(code=503, message="GAN 模型未加载或不可用")

        result = gan.generate(
            num_samples=num_samples,
            class_name=defect_class,
        )

        # 保存图像
        config = get_config()
        gan_output_dir = os.path.join(config["output_dir"], "gan_outputs")
        os.makedirs(gan_output_dir, exist_ok=True)

        image_urls = []
        import time as _time
        timestamp = int(_time.time() * 1000)
        for i, img in enumerate(result["images"]):
            fname = f"gan_{defect_class}_{i:03d}_{timestamp}.jpg"
            fpath = os.path.join(gan_output_dir, fname)
            img.save(fpath)
            image_urls.append(f"/api/gan/images/{fname}")

        return success_response(data={
            "generated_count": result["num_samples"],
            "defect_class": defect_class,
            "class_names": result["class_names"],
            "image_urls": image_urls,
            "generate_time_ms": round(result["generate_time"] * 1000, 1),
            "model_info": {
                "architecture": "DCGAN (conditional)",
                "image_size": gan.image_size,
            },
        })
    except ValueError as e:
        return error_response(code=400, message=str(e))
    except Exception as e:
        return error_response(code=500, message=f"GAN 生成失败: {str(e)}")


@app.post("/api/gan/generate-grid", summary="GAN生成类别网格图")
async def gan_generate_grid(
    samples_per_class: int = Form(8, ge=1, le=16, description="每类样本数"),
):
    """
    生成所有缺陷类别的样本网格图，直观对比各类缺陷的生成效果

    - **samples_per_class**: 每个类别的样本数量 (1-16)
    """
    try:
        gan = get_gan_manager()
        if gan is None or not gan.is_loaded:
            return error_response(code=503, message="GAN 模型未加载或不可用")

        config = get_config()
        gan_output_dir = os.path.join(config["output_dir"], "gan_outputs")
        os.makedirs(gan_output_dir, exist_ok=True)

        import time as _time
        fname = f"gan_grid_{int(_time.time() * 1000)}.jpg"
        save_path = os.path.join(gan_output_dir, fname)

        t0 = _time.time()
        grid_result = gan.generate_grid(
            samples_per_class=samples_per_class,
            save_path=save_path,
        )
        elapsed = _time.time() - t0

        return success_response(data={
            "grid_image_url": f"/api/gan/images/{fname}",
            "samples_per_class": samples_per_class,
            "num_classes": gan.num_classes,
            "total_samples": grid_result["total_samples"],
            "generate_time_ms": round(elapsed * 1000, 1),
        })
    except Exception as e:
        return error_response(code=500, message=f"网格生成失败: {str(e)}")


@app.get("/api/gan/images/{filename}", summary="获取GAN生成图像")
async def get_gan_image(filename: str):
    """
    获取 GAN 生成的图像文件

    - **filename**: 图像文件名
    """
    try:
        config = get_config()
        gan_output_dir = os.path.join(config["output_dir"], "gan_outputs")
        file_path = os.path.join(gan_output_dir, filename)

        if not os.path.exists(file_path):
            return error_response(code=404, message="图像不存在")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="image/jpeg",
        )
    except Exception as e:
        return error_response(code=500, message=f"获取图像失败: {str(e)}")


@app.get("/api/gan/checkpoints", summary="列出GAN检查点")
async def list_gan_checkpoints():
    """
    列出所有可用的 GAN 检查点文件
    """
    try:
        from models.gan import list_available_checkpoints
        checkpoints = list_available_checkpoints()
        return success_response(data={
            "checkpoints": checkpoints,
            "count": len(checkpoints),
        })
    except Exception as e:
        return error_response(code=500, message=f"获取检查点列表失败: {str(e)}")


# ============================================================
# 根路径重定向到文档
# ============================================================

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "YOLO 缺陷质检系统 API 正在运行，请访问 /docs 查看接口文档"}


# ============================================================
# 启动入口（直接运行时）
# ============================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    print(f"启动 YOLO 质检 API 服务: {host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
    )
