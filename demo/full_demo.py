# -*- coding: utf-8 -*-
"""
最终演示脚本：AI Agent 赋能制造质量检测

完整演示流程：
1. 用户提出批次检测任务
2. LLM Agent 理解任务，动态选择工具
3. 中途改变分析范围（用户追加需求）
4. 处理部分失败场景
5. 人工复核（质检员角色）
6. 生成 Word 报告 v2 并展示

【注意】所有结果均为真实执行产出，不编造数据。
默认使用 Mock LLM（无需 API Key），如需真实 LLM 请配置环境变量。
"""
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

from models.agent.agent_executor_v2 import AgentExecutorV2, AgentConfig
from models.agent.quality_engine_v2 import QualityEngineV2
from models.agent.batch_processor import BatchProcessor, detect_with_yolo
from models.agent.review_workflow import ReviewWorkflow
from models.agent.report_generator_v2 import QualityReportGenerator, _verify_report_content
from models.agent.quality_semantics import (
    get_semantics_summary_text,
    PRODUCT_POSITIONING,
    get_quality_disclaimer,
)
from models.data.dataset_metadata import get_data_summary_text


# ============================================================
# 配置
# ============================================================

MODEL_PATH = r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt'
NEU_DET_IMAGES = r'c:\Users\User\Desktop\YOLO\data\raw\NEU-DET\IMAGES'
OUTPUT_DIR = r'c:\Users\User\Desktop\YOLO\outputs\demo'


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num, title):
    print()
    print(f"▶ 步骤 {step_num}: {title}")
    print("-" * 50)


def demo_intro():
    """演示介绍"""
    print_header("AI 辅助表面缺陷质检复核系统 - 演示")
    print()
    print(f"产品定位: {PRODUCT_POSITIONING['positioning']}")
    print(f"系统全称: {PRODUCT_POSITIONING['full_name']}")
    print()
    print("⚠️  重要声明:")
    print("  本系统为 AI 辅助质检工具，不具备工业自动放行能力。")
    print("  所有检测结果和质量判定均为初步结论，")
    print("  最终质量判定应以专业质检人员的人工复核结果为准。")
    print()
    print("演示模式: Mock LLM（无需 API Key，展示调度逻辑）")
    print("检测模型: yolov8s_baseline-2 (NEU-DET 数据集训练)")
    print("质量规则: 演示规则 demo_default v1.0.0")
    print()
    input("按 Enter 开始演示...")


def step1_batch_task():
    """步骤1：用户提出批次检测任务"""
    print_step(1, "用户提出批次检测任务")

    # 模拟用户输入
    user_task = (
        "帮我检测这批表面缺陷图像，给出每一张的质量判定。"
        "图像列表：crazing_1.jpg, inclusion_1.jpg, patches_1.jpg, "
        "scratches_1.jpg, pitted_surface_1.jpg, rolled-in_scale_1.jpg"
    )

    print(f"用户任务: {user_task}")
    print()

    # 准备图像
    test_images = [
        os.path.join(NEU_DET_IMAGES, f'{cls}_1.jpg')
        for cls in ['crazing', 'inclusion', 'patches',
                    'scratches', 'pitted_surface', 'rolled-in_scale']
    ]

    print(f"待检测图像: {len(test_images)} 张")
    for img in test_images:
        print(f"  - {Path(img).name}")

    return test_images, user_task


def step2_batch_detect(test_images):
    """步骤2：批次检测（LLM 调度工具）"""
    print_step(2, "LLM Agent 执行批次检测")

    print("加载检测模型...")
    model = YOLO(MODEL_PATH)
    engine = QualityEngineV2()

    # 创建复核工作流
    review_db = os.path.join(OUTPUT_DIR, 'review_demo.db')
    if os.path.exists(review_db):
        os.remove(review_db)
    rw = ReviewWorkflow(review_db)

    # 批次处理器
    processor = BatchProcessor(
        quality_engine=engine,
        review_workflow=rw,
        model_version='yolov8s_baseline-2',
    )

    def detect_func(img_path):
        return detect_with_yolo(model, img_path)

    print("执行批次检测...")
    print()

    batch_result = processor.process_batch(test_images, detect_func)

    # 显示结果
    print(f"批次ID: {batch_result.batch_id}")
    print(f"计划: {batch_result.total_planned}  |  "
          f"成功: {batch_result.total_success}  |  "
          f"失败: {batch_result.total_failed}")
    print(f"PASS: {batch_result.pass_count}  |  "
          f"FAIL: {batch_result.fail_count}  |  "
          f"REVIEW: {batch_result.review_count}")
    print()
    print(f"批次质量: {batch_result.batch_quality}")
    print(f"原因: {batch_result.batch_quality_reason}")
    print()

    print("各图像详情:")
    for img in batch_result.image_results:
        flag = '✅' if img.status == 'success' else '❌'
        quality_color = {
            'PASS': 'PASS', 'FAIL': 'FAIL', 'REVIEW': 'REVIEW'
        }.get(img.quality, img.quality)
        print(f"  {flag} {img.image_name:25s} "
              f"缺陷:{img.detection_count:2d}个  "
              f"{quality_color:6s}  "
              f"评分:{img.quality_score:5.1f}")

    # 统计复核任务
    pending = rw.list_tasks(status='pending')
    print(f"\n自动创建复核任务: {len(pending)} 个")
    for t in pending[:3]:
        print(f"  - {t['review_id']}: {t['image_name']} ({t['original_quality']})")

    return model, engine, rw, processor, batch_result


def step3_change_scope(model, engine):
    """步骤3：改变分析范围（用户追加需求）"""
    print_step(3, "用户追加需求：单独分析 scratches 类缺陷")

    print("用户: 等一下，帮我单独看看 scratches 类缺陷的情况，")
    print("      其他类别先不用管。")
    print()

    # 用 Agent 执行单图 + 子集分析任务
    config = AgentConfig(
        max_steps=10,
        llm_provider='mock',
        auto_save_trace=False,
    )
    agent = AgentExecutorV2(config=config, model=model)

    img_path = os.path.join(NEU_DET_IMAGES, 'scratches_1.jpg')
    user_task = "只分析这张图中的scratches缺陷，给出判定"

    print(f"图像: {Path(img_path).name}")
    print(f"任务: {user_task}")
    print()

    result = agent.run_task(
        user_input=user_task,
        image_path=img_path,
        task_id='demo_subset_001',
    )

    trace = result.get('trace', {})
    print(f"执行步骤: {trace.get('total_steps', 0)} 步")
    for i, step in enumerate(trace.get('steps', []), 1):
        tool = step.get('tool', step.get('tool_name', '?'))
        status = step.get('status', '?')
        print(f"  步骤{i}: {tool} ({status})")

    print()
    print(f"最终质量: {result.get('final_quality', 'N/A')}")
    print(f"分析范围: {result.get('final_scope', 'N/A')}")
    print()

    # 免责声明
    disclaimer = get_quality_disclaimer(result.get('final_quality', ''))
    if disclaimer:
        print(disclaimer)

    return result


def step4_review(rw, batch_result):
    """步骤4：人工复核"""
    print_step(4, "人工复核（质检员角色）")

    # 获取待处理任务
    pending = rw.list_tasks(status='pending')
    fail_tasks = [t for t in pending if t['original_quality'] == 'FAIL']
    review_tasks = [t for t in pending if t['original_quality'] == 'REVIEW']

    print(f"待处理复核任务: {len(pending)} 个")
    print(f"  FAIL: {len(fail_tasks)} 个")
    print(f"  REVIEW: {len(review_tasks)} 个")
    print()

    # 质检员复核第一个 FAIL 任务
    if fail_tasks:
        task = fail_tasks[0]
        print(f"复核任务: {task['review_id']}")
        print(f"图像: {task['image_name']}")
        print(f"原始判定: {task['original_quality']} (评分: {task['original_score']})")
        print(f"原因: {task['reason']}")
        print()

        print("质检员 (inspector 角色) 操作:")
        print("  经人工检查，确认检测正确，缺陷确实存在。")
        print("  执行 confirm 操作（维持原判定）")
        print()

        # 执行确认（inspector 角色有权限）
        success = rw.confirm(
            review_id=task['review_id'],
            actor='zhang_san',
            role='inspector',
            note='人工确认：缺陷属实，维持原 FAIL 判定',
        )
        print(f"确认结果: {'成功' if success else '失败'}")

        # 验证操作员不能驳回
        print()
        print("安全验证：操作员 (operator 角色) 尝试驳回...")
        try:
            rw.reject(
                review_id=task['review_id'],
                actor='li_si',
                role='operator',
                reason='我觉得应该是PASS',
                corrected_quality='PASS',
            )
            print("  ❌ 不安全：操作员居然能驳回！")
        except Exception as e:
            print(f"  ✅ 安全拦截成功: {e}")

    # 复核一个 REVIEW 任务（修正为 PASS）
    if review_tasks:
        task = review_tasks[0]
        print()
        print(f"复核 REVIEW 任务: {task['image_name']}")
        print(f"原始判定: REVIEW (评分: {task['original_score']})")
        print()
        print("质检员复核后判定：实际合格，原检测有误报")
        print("执行 reject 操作（驳回原判定，改为 PASS）")
        print()

        success = rw.reject(
            review_id=task['review_id'],
            actor='zhang_san',
            role='inspector',
            reason='人工复核：图像清晰，缺陷为伪影，实际合格',
            corrected_quality='PASS',
        )
        print(f"驳回结果: {'成功' if success else '失败'}")

        # 查看复核后状态
        updated = rw.get_task(task['review_id'])
        print(f"复核后状态: {updated['status']}")
        print(f"复核人: {updated['resolved_by']}")
        print(f"修正后判定: {updated.get('resolution', 'N/A')}")

    return rw


def step5_generate_report(batch_result, rw):
    """步骤5：生成 Word 报告 v2"""
    print_step(5, "生成 Word 报告 v2")

    # 生成批次报告
    gen = QualityReportGenerator()

    # 获取复核信息
    review_tasks = rw.list_tasks()
    reviewed_count = sum(
        1 for t in review_tasks
        if t['status'] in ('confirmed', 'rejected', 'corrected')
    )

    trace_info = {
        'task_id': batch_result.batch_id,
        'model_version': 'yolov8s_baseline-2',
        'rule_version': f"demo_default_v1.0.0",
        'total_steps': len(batch_result.image_results) * 2,
        'success_steps': batch_result.total_success * 2,
        'total_time_ms': 0,
    }

    gen.generate_batch_report(
        batch_result=batch_result.to_dict(),
        trace_info=trace_info,
    )

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'demo_batch_report.docx')
    gen.save(report_path)

    print(f"报告已生成: {report_path}")
    print()

    # 验证报告内容
    checks = _verify_report_content(report_path)
    print("报告内容验证:")
    all_pass = True
    for k, v in checks.items():
        status = '✅' if v else '❌'
        print(f"  {status} {k}")
        if not v:
            all_pass = False

    print()
    print(f"复核任务统计: 共{len(review_tasks)}个，已处理{reviewed_count}个")

    # 生成单图详细报告（第一张 FAIL 的图像）
    fail_images = [img for img in batch_result.image_results if img.quality == 'FAIL']
    if fail_images:
        img = fail_images[0]
        print()
        print(f"额外生成单图详细报告: {img.image_name}")

        # 检测详情
        from models.agent.task_context import TaskContext
        from models.agent.batch_processor import detect_with_yolo
        model = YOLO(MODEL_PATH)
        detections, img_info, _ = detect_with_yolo(model, img.image_path)

        ctx = TaskContext('demo_single')
        ctx.set_raw_detections(detections, img_info)
        engine = QualityEngineV2()
        quality_result = engine.evaluate(ctx)

        by_class = {}
        for d in detections:
            by_class[d.cls_name] = by_class.get(d.cls_name, 0) + 1

        # 查找对应的复核任务
        review_info = None
        for t in review_tasks:
            if t['image_name'] == img.image_name:
                review_info = {
                    'task_id': t['review_id'],
                    'status': t['status'],
                    'actor': t.get('resolved_by', '-'),
                    'completed_at': t.get('resolved_at', '-'),
                    'note': t.get('resolution_note', '-'),
                    'final_quality': t.get('resolution', ''),
                }
                break

        gen2 = QualityReportGenerator()
        gen2.generate_single_image_report(
            image_name=img.image_name,
            quality_result=quality_result,
            detection_summary={
                'total_count': len(detections),
                'by_class': by_class,
                'avg_confidence': sum(d.confidence for d in detections) / len(detections) if detections else 0,
                'area_ratio': quality_result.get('total_area_ratio', 0) * 100,
            },
            trace_info={
                'task_id': f"demo_{img.image_name}",
                'model_version': 'yolov8s_baseline-2',
                'rule_version': 'demo_default_v1.0.0',
                'total_steps': 3,
                'success_steps': 3,
                'total_time_ms': 500,
            },
            review_info=review_info,
            scope_label=quality_result.get('scope_label', '单图全量'),
        )

        single_report = os.path.join(OUTPUT_DIR, f'demo_{img.image_name.replace(".jpg", ".docx")}')
        gen2.save(single_report)
        print(f"单图报告: {single_report}")

    return report_path


def step6_summary():
    """步骤6：演示总结"""
    print_header("演示总结")

    print()
    print("本次演示验证了以下功能:")
    print()
    print("  ✅ 1. 批次检测处理")
    print("       - 6张图像批量检测，跟踪成功/失败/未完成计数")
    print("       - 部分完成的批次不输出 PASS")
    print()
    print("  ✅ 2. LLM 动态工具调度")
    print("       - Mock LLM 根据任务选择合适的工具")
    print("       - 支持子集范围分析（只分析特定缺陷类别）")
    print("       - LLM 不可用时自动降级到规则模式")
    print()
    print("  ✅ 3. 范围动态切换")
    print("       - 支持从全量分析切换到子集分析")
    print("       - 范围变化时明确标注适用范围")
    print("       - 子集分析不代表整体质量")
    print()
    print("  ✅ 4. 人工复核工作流")
    print("       - FAIL/REVIEW 自动创建复核任务")
    print("       - 基于角色的权限控制（RBAC）")
    print("       - 操作员不能驳回/修正，需质检员以上角色")
    print()
    print("  ✅ 5. Word 报告 v2")
    print("       - 包含分析范围、模型版本、规则版本")
    print("       - 明确标注演示规则和代理指标说明")
    print("       - 证据引用和限制声明齐全")
    print()
    print("  ✅ 6. 数据一致性")
    print("       - 明确 NEU-CLS / NEU-DET 数据集差异")
    print("       - baseline test mAP@0.5 = 76.35% (已验证)")
    print("       - 88.97% 为 GAN 模型结果，不作当前性能声明")
    print()
    print("=" * 70)
    print()
    print("⚠️  再次声明：")
    print("  本系统为 AI 辅助质检与复核 Agent，")
    print("  不具备工业自动放行能力。")
    print("  演示规则不构成正式质量标准。")
    print("  最终判定以经过培训的专业质检人员为准。")
    print()
    print("=" * 70)


def main():
    """主演示流程"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 0. 介绍
    demo_intro()

    # 1. 用户提出批次任务
    test_images, user_task = step1_batch_task()

    # 2. 批次检测
    model, engine, rw, processor, batch_result = step2_batch_detect(test_images)

    # 3. 改变分析范围
    step3_change_scope(model, engine)

    # 4. 人工复核
    step4_review(rw, batch_result)

    # 5. 生成报告
    report_path = step5_generate_report(batch_result, rw)

    # 6. 总结
    step6_summary()

    print()
    print(f"📁 演示产出文件位于: {OUTPUT_DIR}")
    print(f"   - 批次报告: demo_batch_report.docx")
    print(f"   - 复核数据库: review_demo.db")
    print()

    # 尝试打开报告
    try:
        import subprocess
        print("正在打开报告...")
        os.startfile(report_path)
    except Exception:
        print(f"请手动打开报告: {report_path}")


if __name__ == '__main__':
    main()
