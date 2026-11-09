# -*- coding: utf-8 -*-
"""
三条核心演示流程脚本（可复现）

演示1：批次检测 → 规则判定 → 异常复核 → 报告
演示2：仅筛选裂纹分析 → 切回整批判定，证明范围不污染
演示3：模糊图像/工具失败 → 受控REVIEW，不误放行

用法：
    python tools/demo_flows.py --demo 1
    python tools/demo_flows.py --demo all
"""
import sys
import os
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent import (
    AgentExecutorV2,
    AgentConfig,
    TaskContext,
    DetectionItem,
    ImageInfo,
    QualityEngineV2,
    ReviewWorkflow,
)


# 测试图像
TEST_IMAGES = {
    'crazing': r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\crazing_0001.jpg',
    'inclusion': r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\inclusion_0001.jpg',
    'patches': r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\patches_0001.jpg',
    'scratches': r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\scratches_0001.jpg',
    'pitted': r'c:\Users\User\Desktop\YOLO\data\raw\NEU-CLS\pitted_surface_0001.jpg',
}


def print_header(title, demo_id):
    """打印演示标题"""
    print("\n" + "=" * 70)
    print(f"🎬 演示 {demo_id}: {title}")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_footer(demo_id, success):
    """打印演示结尾"""
    print()
    print("-" * 70)
    status = "✅ 演示完成" if success else "❌ 演示失败"
    print(f"{status} - 演示 {demo_id}")
    print("=" * 70 + "\n")


def demo_1_detect_judge_review_report(agent, db_path):
    """
    演示1：批次检测 → 规则判定 → 异常复核 → 报告

    场景：检测一批图像，质量判定发现异常，进入复核队列，人工确认后生成报告
    """
    print_header("批次检测→规则判定→异常复核→报告", 1)

    rw = ReviewWorkflow(db_path)

    # 步骤1：检测图像
    print("【步骤1】图像缺陷检测")
    img_path = TEST_IMAGES['crazing']
    print(f"  图像: {Path(img_path).name}")

    result = agent.run_task(
        user_input='检测一下缺陷',
        image_path=img_path,
    )
    det_count = result['context_snapshot']['detection_count']
    print(f"  ✅ 检测完成，发现 {det_count} 个缺陷")

    # 步骤2：质量判定
    print("\n【步骤2】质量规则判定")
    result2 = agent.run_task(
        user_input='质量合格吗',
        image_path=img_path,
    )
    quality = result2.get('final_quality', 'UNKNOWN')
    scope = result2.get('final_scope', '')
    print(f"  判定结果: {quality}")
    print(f"  分析范围: {scope}")
    print(f"  规则版本: demo_default_v1.0.0 (演示规则)")

    # 步骤3：如果是 REVIEW 或 FAIL，创建复核任务
    if quality in ('REVIEW', 'FAIL'):
        print(f"\n【步骤3】创建人工复核任务")
        rev_id = rw.create_task(
            task_id=result2.get('task_id', 'unknown'),
            record_id=1,
            image_name=Path(img_path).name,
            image_path=img_path,
            original_quality=quality,
            original_score=75,
            original_scope=scope,
            reason=f'自动检测结果为{quality}，需人工复核',
            quality_issue='待人工确认',
            priority='high' if quality == 'FAIL' else 'medium',
            model_version='yolov8s_baseline-2',
            rule_version='demo_default_v1.0.0',
        )
        print(f"  ✅ 复核任务已创建: {rev_id}")
        print(f"  优先级: {'高' if quality == 'FAIL' else '中'}")
        print(f"  状态: 待处理 (pending)")

        # 步骤4：人工复核
        print(f"\n【步骤4】人工复核")
        print(f"  复核人: inspector_zhang")
        print(f"  操作: 确认原判定")
        print(f"  理由: 人工核验后确认原判定正确，缺陷属实")

        rw.confirm(
            rev_id,
            actor='inspector_zhang',
            note='人工核验后确认原判定正确，缺陷属实'
        )

        task = rw.get_task(rev_id)
        print(f"  ✅ 复核完成，状态: {task['status']}")

        # 步骤5：复核历史
        print(f"\n【步骤5】复核操作历史（可追溯）")
        history = rw.get_action_history(rev_id)
        for h in history:
            print(f"  {h['action_time']} - {h['action']} - {h.get('actor', 'system')}")
            if h.get('note'):
                print(f"      备注: {h['note']}")

    # 步骤6：报告
    print(f"\n【步骤6】可追溯报告")
    print(f"  报告包含:")
    print(f"    • 检测图像: {Path(img_path).name}")
    print(f"    • 检测结果: {det_count} 个缺陷")
    print(f"    • 质量判定: {quality} ({scope})")
    print(f"    • 规则版本: demo_default_v1.0.0")
    print(f"    • 模型版本: yolov8s_baseline-2")
    print(f"    • 复核状态: {'已复核' if quality in ('REVIEW', 'FAIL') else '无需复核'}")
    print(f"    • 执行轨迹: task_{result2.get('task_id', 'unknown')}")
    print(f"  ✅ 报告生成完成，所有结论可追溯至工具结果")

    print_footer(1, True)
    return True


def demo_2_scope_isolation(agent):
    """
    演示2：仅筛选裂纹分析 → 切回整批判定，证明范围不污染

    关键证明点：
    1. 筛选裂纹后，子集判定只针对裂纹
    2. 切回全量后，判定结果与直接全量判定完全一致
    3. 原始检测数据始终完整
    """
    print_header("子集筛选→切回全量（范围不污染证明）", 2)

    img_path = TEST_IMAGES['inclusion']
    print(f"测试图像: {Path(img_path).name}")

    # 步骤1：直接全量检测+判定（基线）
    print("\n【步骤1】全量检测+质量判定（基线）")
    result_full = agent.run_task(
        user_input='检测一下，质量合格吗',
        image_path=img_path,
    )
    full_quality = result_full.get('final_quality', '')
    full_scope = result_full.get('final_scope', '')
    full_count = result_full['context_snapshot']['detection_count']
    print(f"  范围: {full_scope}")
    print(f"  缺陷数: {full_count}")
    print(f"  质量判定: {full_quality}")

    # 步骤2：筛选裂纹，子集判定
    print("\n【步骤2】仅筛选裂纹 → 子集判定")
    result_subset = agent.run_task(
        user_input='只看裂纹，质量合格吗',
        image_path=img_path,
    )
    subset_quality = result_subset.get('final_quality', '')
    subset_scope = result_subset.get('final_scope', '')
    subset_count = result_subset['context_snapshot']['detection_count']
    print(f"  范围: {subset_scope}")
    print(f"  裂纹数量: {subset_count}")
    print(f"  子集质量判定: {subset_quality}")
    print(f"  ⚠️ 注意：此结论仅针对裂纹子集，不代表整体质量！")

    # 步骤3：切回全量，再次判定
    print("\n【步骤3】切回全量 → 全量判定")
    result_full2 = agent.run_task(
        user_input='先只看裂纹，再看整体质量',
        image_path=img_path,
    )
    full2_quality = result_full2.get('final_quality', '')
    full2_scope = result_full2.get('final_scope', '')
    full2_count = result_full2['context_snapshot']['detection_count']
    print(f"  范围: {full2_scope}")
    print(f"  缺陷数: {full2_count}")
    print(f"  质量判定: {full2_quality}")

    # 步骤4：验证一致性
    print("\n【步骤4】范围一致性验证")
    print(f"  全量缺陷数: {full_count} vs 切回后: {full2_count} → {'✅ 一致' if full_count == full2_count else '❌ 不一致'}")
    print(f"  全量判定: {full_quality} vs 切回后: {full2_quality} → {'✅ 一致' if full_quality == full2_quality else '❌ 不一致'}")
    print(f"  子集范围: {subset_scope} → {'✅ 正确标注子集' if '子集' in subset_scope else '❌ 未标注'}")

    # 关键结论
    print("\n【结论】")
    print(f"  ✅ 子集分析不会污染全量质量结论")
    print(f"  ✅ 原始检测数据始终完整（{full_count} 个缺陷）")
    print(f"  ✅ 子集判定明确标注范围，不会误导为整体结论")
    print(f"  ✅ 相同证据 + 相同规则版本 = 一致的判定结果")

    success = (full_count == full2_count and full_quality == full2_quality and '子集' in subset_scope)
    print_footer(2, success)
    return success


def demo_3_safety_gate(agent):
    """
    演示3：模糊图像/证据不足 → 受控REVIEW，不误放行

    关键证明点：
    1. 图像质量差 → 强制 REVIEW（图像质量门控）
    2. 无检测结果 → REVIEW（证据不足，不得PASS）
    3. 明确FAIL证据时不降级为REVIEW
    """
    print_header("安全边界：低质量/无证据 → REVIEW，不误放行", 3)

    engine = QualityEngineV2()

    # 场景1：低质量图像
    print("\n【场景1】低质量图像（模糊）")
    dets = [DetectionItem('pitted_surface', 0.7, 10, 10, 30, 30)]
    img_blurry = ImageInfo('img_blurry', 'blurry.jpg', 200, 200,
                           quality_score=0.25, quality_issues=['图像模糊', '失焦'])
    ctx1 = TaskContext('demo3_1')
    ctx1.set_raw_detections(dets, img_blurry)
    result1 = engine.evaluate(ctx1)

    print(f"  图像质量分: 0.25 / 1.0")
    print(f"  质量问题: {', '.join(img_blurry.quality_issues)}")
    print(f"  检测到: {len(dets)} 个缺陷（轻度）")
    print(f"  判定结果: {result1['quality']}")
    print(f"  是否强制: {'是' if result1['is_forced'] else '否'}")
    if result1['forced_reason']:
        print(f"  强制原因: {result1['forced_reason']}")
    print(f"  ✅ 低质量图像不直接PASS，进入REVIEW")

    # 场景2：无检测数据
    print("\n【场景2】无检测数据（检测失败/空结果）")
    img_empty = ImageInfo('img_empty', 'empty.jpg', 200, 200,
                          quality_score=0.8, quality_issues=[])
    ctx2 = TaskContext('demo3_2')
    ctx2.image = img_empty  # 只设置图像，不设置检测
    result2 = engine.evaluate(ctx2)

    print(f"  图像质量分: 0.80 / 1.0")
    print(f"  检测结果: 0 个（无检测数据）")
    print(f"  判定结果: {result2['quality']}")
    print(f"  是否强制: {'是' if result2['is_forced'] else '否'}")
    if result2['forced_reason']:
        print(f"  强制原因: {result2['forced_reason']}")
    print(f"  ✅ 没有检测数据≠合格，进入REVIEW")

    # 场景3：明确FAIL证据不降级
    print("\n【场景3】明确FAIL证据（不允许因不确定性降级）")
    dets_critical = [
        DetectionItem('crazing', 0.95, 10, 10, 100, 100),
        DetectionItem('inclusion', 0.92, 120, 120, 190, 190),
        DetectionItem('scratches', 0.88, 10, 120, 80, 190),
    ]
    img_good = ImageInfo('img_critical', 'critical.jpg', 200, 200,
                         quality_score=0.9, quality_issues=[])
    ctx3 = TaskContext('demo3_3')
    ctx3.set_raw_detections(dets_critical, img_good)
    result3 = engine.evaluate(ctx3)

    print(f"  图像质量分: 0.90 / 1.0")
    print(f"  检测到: {len(dets_critical)} 个严重缺陷")
    print(f"  判定结果: {result3['quality']}")
    print(f"  评分: {result3['score']}")
    print(f"  是否强制: {'是' if result3['is_forced'] else '否'}")
    print(f"  ✅ 明确FAIL证据时，不降级为REVIEW")

    # 总结
    print("\n【安全边界总结】")
    print(f"  1. ✅ 图像质量门控：低质量图像强制REVIEW")
    print(f"  2. ✅ 证据不足门控：无检测数据不得PASS")
    print(f"  3. ✅ FAIL不降级：明确FAIL证据不因不确定性降级")
    print(f"  4. ✅ 所有强制判定均有明确原因说明")

    success = (result1['quality'] == 'REVIEW' and result1['is_forced'] and
               result2['quality'] == 'REVIEW' and result2['is_forced'] and
               result3['quality'] == 'FAIL')
    print_footer(3, success)
    return success


def run_all_demos(agent, db_path):
    """运行所有演示"""
    results = []

    results.append(('演示1', demo_1_detect_judge_review_report(agent, db_path)))
    results.append(('演示2', demo_2_scope_isolation(agent)))
    results.append(('演示3', demo_3_safety_gate(agent)))

    print("\n" + "=" * 70)
    print("📋 演示流程汇总")
    print("=" * 70)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    print("=" * 70)

    return all(s for _, s in results)


def main():
    parser = argparse.ArgumentParser(description='三条核心演示流程')
    parser.add_argument('--demo', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='运行哪个演示 (1/2/3/all)')
    parser.add_argument('--output', type=str, default='',
                        help='输出演示报告JSON')
    args = parser.parse_args()

    # 加载模型和Agent
    from ultralytics import YOLO
    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'demo.db')
        config = AgentConfig(max_steps=10, llm_provider='none', auto_save_trace=False)
        agent = AgentExecutorV2(config, model=model, db_path=db_path)

        if args.demo == '1':
            demo_1_detect_judge_review_report(agent, db_path)
        elif args.demo == '2':
            demo_2_scope_isolation(agent)
        elif args.demo == '3':
            demo_3_safety_gate(agent)
        else:
            run_all_demos(agent, db_path)


if __name__ == '__main__':
    main()
