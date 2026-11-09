# -*- coding: utf-8 -*-
"""
Agent v2 端到端集成测试
验证：检测 → 统计 → 子集筛选 → 子集判定 → 切回全量 → 全量判定 → 轨迹记录
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent import (
    AgentExecutorV2,
    AgentConfig,
    QualityEngineV2,
    QualityRuleSet,
    TaskContext,
    DetectionItem,
    ImageInfo,
    ReviewWorkflow,
)


def test_e2e_detect_and_judge():
    """端到端测试1：检测 + 统计 + 判定"""
    print("=" * 60)
    print("E2E测试1：检测 → 统计 → 质量判定")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')
    config = AgentConfig(max_steps=10, llm_provider='none')

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        agent = AgentExecutorV2(config, model=model, db_path=db_path)

        img_path = r'c:\Users\User\Desktop\YOLO\data\processed\gan_dataset\train\crazing\crazing_0001.jpg'

        result = agent.run_task(
            user_input='检测一下缺陷，质量合格吗',
            image_path=img_path,
        )

        print(f"模式: {result['mode']}")
        print(f"任务ID: {result['task_id']}")
        print(f"最终结论: {result['final_conclusion']}")
        print(f"结论范围: {result['final_scope']}")

        trace = result['trace']
        print(f"\n执行轨迹 ({trace['total_steps']} 步，耗时 {trace['total_time_ms']:.0f}ms):")
        for step in trace['steps']:
            icon = '✅' if step['status'] == 'success' else '❌'
            print(f"  {step['step']}. {icon} {step['action']} [{step['tool']}] ({step['time_ms']:.0f}ms)")
            if step.get('result'):
                print(f"     → {step['result'][:80]}")

        ctx = result['context_snapshot']
        print(f"\n上下文快照:")
        print(f"  范围: {ctx['scope_label']}")
        print(f"  检测数: {ctx['detection_count']}")
        print(f"  模型版本: {ctx['model_version']}")
        print(f"  规则版本: {ctx['rule_version']}")

        assert trace['total_steps'] > 0, "应该有执行步骤"
        assert 'FAIL' in result['final_conclusion'] or 'PASS' in result['final_conclusion'] or 'REVIEW' in result['final_conclusion'], "应有质量等级结论"
        assert trace['llm_fallback'] == True, "规则模式下 llm_fallback 应为 True"

        print("🎉 E2E测试1通过\n")
        return True


def test_e2e_subset_no_pollution():
    """端到端测试2：子集筛选不污染全量判定"""
    print("=" * 60)
    print("E2E测试2：子集筛选 → 子集判定 → 切回全量 → 全量判定")
    print("=" * 60)

    # 构造已知数据，直接测试上下文和判定引擎
    dets = [
        DetectionItem('crazing', 0.9, 10, 10, 50, 50),      # 严重缺陷
        DetectionItem('pitted_surface', 0.7, 100, 10, 120, 30),  # 轻度
        DetectionItem('scratches', 0.8, 200, 20, 220, 80),    # 中等
        DetectionItem('crazing', 0.85, 300, 100, 340, 150),   # 严重
        DetectionItem('inclusion', 0.92, 400, 200, 450, 250), # 严重
    ]
    img = ImageInfo('img1', 'test.jpg', 640, 480)

    ctx = TaskContext('test_subset')
    ctx.set_raw_detections(dets, img)
    engine = QualityEngineV2()

    # 全量判定
    full_result = engine.evaluate(ctx, scope='full')
    print(f"全量判定: {full_result['quality']} (评分 {full_result['score']}, {full_result['total_defects']}个缺陷)")

    # 筛选裂纹
    view = ctx.filter_by_class('crazing')
    ctx.switch_to_view(view)
    print(f"筛选裂纹后: {len(ctx.get_current_detections())} 个 (范围: {ctx.scope_label})")

    # 子集判定
    subset_result = engine.evaluate(ctx, scope='current')
    print(f"子集判定: {subset_result['quality']} (评分 {subset_result['score']}, {subset_result['total_defects']}个)")
    assert subset_result['scope_label'] == '单图子集', "子集判定应有子集标签"
    assert any('子集' in n for n in subset_result.get('method_notes', [])), "应有子集方法备注"

    # 关键验证：原始数据没变
    assert len(ctx.get_raw_detections()) == 5, "原始检测数据被污染了！"
    print(f"✅ 原始数据完整：{len(ctx.get_raw_detections())} 个")

    # 切回全量
    ctx.switch_to_full()
    assert ctx.scope_label == '单图全量'

    full_result2 = engine.evaluate(ctx, scope='full')
    print(f"切回全量后判定: {full_result2['quality']} (评分 {full_result2['score']})")

    # 全量判定应该和第一次一致
    assert full_result['quality'] == full_result2['quality'], "切回全量后判定应该和原始全量一致"
    assert full_result['score'] == full_result2['score'], "评分应该一致"
    print(f"✅ 全量判定一致：{full_result['quality']} = {full_result2['quality']}, {full_result['score']} = {full_result2['score']}")

    # 子集判定的结果不应该和全量一样（因为数据不同）
    # （注意：可能碰巧一样，但大多数情况不一样）
    print(f"\n子集 vs 全量对比:")
    print(f"  子集: {subset_result['quality']} ({subset_result['total_defects']}个缺陷, {subset_result['scope_label']})")
    print(f"  全量: {full_result['quality']} ({full_result['total_defects']}个缺陷, {full_result['scope_label']})")

    print("🎉 E2E测试2通过：子集不污染全量\n")
    return True


def test_e2e_review_workflow():
    """端到端测试3：复核工作流完整流程"""
    print("=" * 60)
    print("E2E测试3：复核任务创建 → 确认 → 历史记录")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'review.db')
        rw = ReviewWorkflow(db_path)

        # 创建
        rid = rw.create_task(
            task_id='task_123', record_id=1,
            image_name='test.jpg', image_path='/tmp/test.jpg',
            original_quality='REVIEW', original_score=72,
            original_scope='full_image',
            reason='低置信缺陷过多', quality_issue='3个低置信',
            priority='high',
            model_version='yolov8s_test', rule_version='demo_v1',
        )
        print(f"✅ 创建复核任务: {rid}")

        # 初始统计
        stats = rw.get_stats()
        assert stats['pending'] == 1
        assert stats['today_new'] == 1
        print(f"✅ 初始统计: 待处理{stats['pending']}, 总计{stats['total']}")

        # 确认
        ok = rw.confirm(rid, actor='tester1', note='人工复核确认，原判定正确')
        assert ok
        print(f"✅ 确认原判定")

        # 确认后统计
        stats = rw.get_stats()
        assert stats['pending'] == 0
        assert stats['confirmed'] == 1
        print(f"✅ 确认后统计: 待处理{stats['pending']}, 已确认{stats['confirmed']}")

        # 操作历史
        history = rw.get_action_history(rid)
        assert len(history) >= 2  # created + confirmed
        actions = [h['action'] for h in history]
        assert 'created' in actions
        assert 'confirmed' in actions
        print(f"✅ 操作历史: {len(history)} 条 ({', '.join(actions)})")

        # 待标注池
        ok = rw.add_to_annotation_pool(rid, reason='困难样本，需进一步研究')
        assert ok
        pool = rw.list_annotation_pool()
        assert len(pool) == 1
        assert pool[0]['used_for_training'] == 0  # 安全：不自动用于训练
        print(f"✅ 待标注池: {len(pool)} 个, used_for_training={pool[0]['used_for_training']}")

        print("🎉 E2E测试3通过：复核工作流完整\n")
        return True


def test_e2e_image_quality_gate():
    """端到端测试4：图像质量门控"""
    print("=" * 60)
    print("E2E测试4：图像质量门控（低质量图像不得 PASS）")
    print("=" * 60)

    dets = [DetectionItem('pitted_surface', 0.6, 10, 10, 30, 30)]

    # 高质量图像
    img_good = ImageInfo('img1', 'good.jpg', 640, 480, quality_score=0.9, quality_issues=[])
    ctx_good = TaskContext('test_good')
    ctx_good.set_raw_detections(dets, img_good)

    engine = QualityEngineV2()
    result_good = engine.evaluate(ctx_good)
    print(f"高质量图像(0.9): {result_good['quality']} (强制={result_good['is_forced']})")
    # 1个轻度缺陷，可能 PASS 或 REVIEW，取决于评分
    assert result_good['is_forced'] == False, "高质量图像不应触发强制"

    # 低质量图像
    img_bad = ImageInfo('img2', 'bad.jpg', 640, 480, quality_score=0.2, quality_issues=['图像模糊'])
    ctx_bad = TaskContext('test_bad')
    ctx_bad.set_raw_detections(dets, img_bad)

    result_bad = engine.evaluate(ctx_bad)
    print(f"低质量图像(0.2): {result_bad['quality']} (强制={result_bad['is_forced']})")
    assert result_bad['is_forced'] == True, "低质量图像应触发强制 REVIEW"
    assert result_bad['quality'] == 'REVIEW', "低质量图像应为 REVIEW"
    assert result_bad['forced_reason'], "应有强制原因"
    print(f"  强制原因: {result_bad['forced_reason']}")

    # 无检测数据 + 低质量图像
    ctx_empty = TaskContext('test_empty_q')
    img_empty = ImageInfo('img3', 'empty.jpg', 640, 480, quality_score=0.15, quality_issues=['严重模糊'])
    ctx_empty.image = img_empty  # 只设置图像，不设置检测（模拟检测失败）
    # 注意：没有 set_raw_detections，所以 has_raw_detections = False

    result_empty = engine.evaluate(ctx_empty)
    print(f"无检测+低质量: {result_empty['quality']} (强制={result_empty['is_forced']})")
    assert result_empty['quality'] == 'REVIEW'
    assert result_empty['is_forced'] == True
    print(f"  强制原因: {result_empty['forced_reason']}")

    print("🎉 E2E测试4通过：图像质量门控正确\n")
    return True


def run_all():
    print("\n" + "=" * 60)
    print("Agent v2 端到端集成测试")
    print("=" * 60 + "\n")

    tests = [
        test_e2e_detect_and_judge,
        test_e2e_subset_no_pollution,
        test_e2e_review_workflow,
        test_e2e_image_quality_gate,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} 断言失败：{e}\n")
            failed += 1
        except Exception as e:
            print(f"💥 {test.__name__} 异常：{e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"集成测试结果：{passed} 通过，{failed} 失败，共 {len(tests)} 项")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
