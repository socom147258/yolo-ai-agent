# -*- coding: utf-8 -*-
"""
P0 级单元测试 —— 验证核心改造的正确性

测试范围：
1. 任务上下文 & 范围隔离（子集不污染全量）
2. 质量判定安全边界（图像门控/面积代理/证据不足）
3. 执行轨迹记录
4. 工具参数校验
5. 规则模式 Agent 执行
"""
import sys
import os
import json
import tempfile

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent import (
    TaskContext, DetectionItem, ImageInfo,
    SCOPE_FULL_IMAGE, SCOPE_SUBSET,
    QualityEngineV2, QualityRuleSet,
    ExecutionTrace, ToolResult, ToolStatus,
    get_tool_schema,
    AgentExecutorV2, AgentConfig,
    ReviewWorkflow,
)


def test_task_context_scope_isolation():
    """测试1：任务上下文范围隔离——筛选不污染原始数据"""
    print("=" * 60)
    print("测试1：任务上下文范围隔离")
    print("=" * 60)

    # 构造检测数据
    dets = [
        DetectionItem('crazing', 0.85, 10, 10, 50, 50),
        DetectionItem('inclusion', 0.92, 100, 100, 150, 200),
        DetectionItem('crazing', 0.78, 200, 200, 250, 280),
        DetectionItem('scratches', 0.65, 300, 100, 320, 300),
        DetectionItem('patches', 0.95, 400, 400, 480, 450),
    ]
    img = ImageInfo('img1', 'test.jpg', 640, 480)

    ctx = TaskContext('test_task_001')
    ctx.set_raw_detections(dets, img)

    # 初始状态
    assert ctx.current_scope == SCOPE_FULL_IMAGE
    assert len(ctx.get_current_detections()) == 5
    assert len(ctx.get_raw_detections()) == 5
    print(f"✅ 初始状态：全量范围，{len(ctx.get_current_detections())} 个检测")

    # 筛选裂纹
    view = ctx.filter_by_class('crazing')
    assert view.scope == SCOPE_SUBSET
    assert len(view.det_ids) == 2
    print(f"✅ 生成裂纹子集视图：{len(view.det_ids)} 个")

    # 切换到子集
    ctx.switch_to_view(view)
    assert ctx.is_subset == True
    assert len(ctx.get_current_detections()) == 2
    print(f"✅ 切换到子集范围：当前 {len(ctx.get_current_detections())} 个")

    # 关键验证：原始数据没变！
    assert len(ctx.get_raw_detections()) == 5, "原始检测数据被修改了！"
    print(f"✅ 原始数据完整：{len(ctx.get_raw_detections())} 个（未被筛选污染）")

    # 切回全量
    ctx.switch_to_full()
    assert ctx.current_scope == SCOPE_FULL_IMAGE
    assert len(ctx.get_current_detections()) == 5
    print(f"✅ 切回全量范围：{len(ctx.get_current_detections())} 个")

    # scope 标签
    assert ctx.scope_label == '单图全量'
    print(f"✅ 范围标签正确：{ctx.scope_label}")

    print("🎉 测试1通过：范围隔离正确\n")
    return True


def test_quality_safety_boundaries():
    """测试2：质量判定安全边界"""
    print("=" * 60)
    print("测试2：质量判定安全边界")
    print("=" * 60)

    engine = QualityEngineV2(QualityRuleSet.default_demo())

    # --- 2a: 无检测数据 → REVIEW（不是 PASS！）---
    ctx_empty = TaskContext('test_empty')
    # 不设置检测数据
    result = engine.evaluate(ctx_empty)
    assert result['quality'] == 'REVIEW', f"无数据时应为 REVIEW，实际是 {result['quality']}"
    assert result['is_forced'] == True
    assert result['forced_reason'], "应有强制原因说明"
    print(f"✅ 2a. 无检测数据 → {result['quality']}（强制），原因：{result['forced_reason']}")

    # --- 2b: 图像质量差 → REVIEW ---
    dets = [DetectionItem('crazing', 0.9, 10, 10, 50, 50)]
    img = ImageInfo('img1', 'test.jpg', 640, 480, quality_score=0.2, quality_issues=['图像模糊'])
    ctx_lowq = TaskContext('test_lowq')
    ctx_lowq.set_raw_detections(dets, img)

    result = engine.evaluate(ctx_lowq)
    assert result['quality'] == 'REVIEW', f"低质量图像应为 REVIEW"
    assert result['is_forced'] == True
    assert any('图像质量' in r for r in result['reasons_review'])
    print(f"✅ 2b. 图像质量差（0.2）→ {result['quality']}（强制）")

    # --- 2c: 有明确 FAIL 证据 → FAIL（不降级）---
    many_critical = [
        DetectionItem('crazing', 0.95, 10, 10, 50, 50),
        DetectionItem('crazing', 0.92, 100, 10, 140, 50),
        DetectionItem('inclusion', 0.90, 200, 10, 250, 50),
    ]
    img2 = ImageInfo('img2', 'test2.jpg', 640, 480, quality_score=0.9)
    ctx_fail = TaskContext('test_fail')
    ctx_fail.set_raw_detections(many_critical, img2)

    result = engine.evaluate(ctx_fail)
    assert result['quality'] == 'FAIL', f"多个严重缺陷应为 FAIL"
    assert result['reasons_fail'], "应有不合格原因"
    # 虽然可能也有 review 因素，但不能降为 REVIEW
    print(f"✅ 2c. 多个严重缺陷 → {result['quality']}（评分 {result['score']}，不降级）")

    # --- 2d: 面积标注为"检测框覆盖率"---
    assert '检测框覆盖率' in result.get('area_method', '')
    print(f"✅ 2d. 面积指标标注正确：{result['area_method']}")

    # --- 2e: 演示规则标记 ---
    assert result['is_demo_rule'] == True
    assert any('演示规则' in n for n in result['method_notes'])
    print(f"✅ 2e. 规则来源标记：{result['rule_source']}（is_demo={result['is_demo_rule']}）")

    # --- 2f: 子集判定标注 ---
    view = ctx_fail.filter_by_class('crazing')
    ctx_fail.switch_to_view(view)
    result_subset = engine.evaluate(ctx_fail)
    assert '子集' in result_subset['scope_label'] or 'subset' in result_subset['scope']
    print(f"✅ 2f. 子集判定标注：{result_subset['scope_label']}")

    print("🎉 测试2通过：安全边界正确\n")
    return True


def test_execution_trace():
    """测试3：执行轨迹记录"""
    print("=" * 60)
    print("测试3：执行轨迹记录")
    print("=" * 60)

    trace = ExecutionTrace('trace_test_001', model_version='yolov8s_test', rule_version='demo_v1')

    # 添加步骤
    trace.add_tool_step(1, 'detect_image', '图像缺陷检测',
                       'img=test.jpg, conf=0.25', '发现3个缺陷',
                       ToolStatus.SUCCESS, exec_ms=150.5, cost=0)

    trace.add_tool_step(2, 'quality_judge', '质量判定',
                       'scope=current', 'FAIL (45分)',
                       ToolStatus.SUCCESS, exec_ms=5.2, cost=0)

    trace.add_tool_step(3, 'save_record', '保存记录',
                       '', '记录ID=123',
                       ToolStatus.ERROR, error='数据库连接失败', exec_ms=100.0, cost=0)

    trace.mark_finished('质检结论：FAIL', '单图全量')

    # 验证
    assert trace.step_count == 3
    assert trace.success_count == 2
    assert trace.error_count == 1
    assert trace.total_cost == 0
    assert trace.final_conclusion == '质检结论：FAIL'
    assert trace.final_conclusion_scope == '单图全量'

    print(f"✅ 步骤数：{trace.step_count}（成功 {trace.success_count}，失败 {trace.error_count}）")
    print(f"✅ 总耗时：{trace.total_time_ms:.1f}ms")
    print(f"✅ 最终结论：{trace.final_conclusion}（{trace.final_conclusion_scope}）")

    # 导出
    data = trace.to_dict()
    assert 'steps' in data
    assert len(data['steps']) == 3
    assert data['steps'][0]['tool'] == 'detect_image'
    print(f"✅ 字典导出正确：{len(data['steps'])} 个步骤，keys={list(data.keys())[:5]}...")

    # 摘要文本
    summary = trace.summary_text()
    assert 'detect_image' in summary
    assert 'quality_judge' in summary
    assert 'save_record' in summary
    print(f"✅ 摘要文本生成正确（{len(summary)} 字符）")

    print("🎉 测试3通过：执行轨迹正确\n")
    return True


def test_tool_schema_validation():
    """测试4：工具参数校验"""
    print("=" * 60)
    print("测试4：工具参数校验")
    print("=" * 60)

    # 检测工具 schema
    schema = get_tool_schema('detect_image')
    assert schema is not None
    assert schema.name == 'detect_image'
    print(f"✅ 找到工具 Schema：{schema.display_name}")

    # 正确参数
    is_valid, errors, cleaned = schema.validate_params({
        'image_path': '/tmp/test.jpg',
        'conf_threshold': 0.3,
    })
    assert is_valid, f"校验失败：{errors}"
    print(f"✅ 正确参数校验通过：conf_threshold = {cleaned['conf_threshold']}")

    # 缺少必选参数
    is_valid, errors, _ = schema.validate_params({})
    assert not is_valid
    assert any('image_path' in e for e in errors)
    print(f"✅ 缺少必选参数 → 校验失败：{errors[0]}")

    # 数值超范围
    is_valid, errors, _ = schema.validate_params({
        'image_path': '/tmp/test.jpg',
        'conf_threshold': 1.5,  # 超过 max_value=0.99
    })
    assert not is_valid
    print(f"✅ 数值超范围 → 校验失败：{errors[0]}")

    # 质量判定工具
    schema2 = get_tool_schema('quality_judge')
    is_valid, errors, cleaned = schema2.validate_params({
        'scope': 'current',
        'rule_set': 'default',
    })
    assert is_valid
    print(f"✅ 质量判定工具参数校验通过")

    # 枚举值错误
    is_valid, errors, _ = schema2.validate_params({
        'scope': 'invalid_scope',  # 不在 choices 中
    })
    assert not is_valid
    print(f"✅ 枚举值错误 → 校验失败：{errors[0]}")

    print("🎉 测试4通过：参数校验正确\n")
    return True


def test_review_workflow():
    """测试5：复核工作流"""
    print("=" * 60)
    print("测试5：复核工作流")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_review.db')
        rw = ReviewWorkflow(db_path)

        # 创建复核任务
        review_id = rw.create_task(
            task_id='task_001',
            record_id=1,
            image_name='test.jpg',
            image_path='/tmp/test.jpg',
            original_quality='REVIEW',
            original_score=72,
            original_scope='full_image',
            reason='低置信缺陷过多',
            quality_issue='3个低置信检测',
            priority='high',
            model_version='yolov8s_test',
            rule_version='demo_v1',
        )

        assert review_id is not None
        assert review_id.startswith('rev_')
        print(f"✅ 创建复核任务：{review_id}")

        # 查询
        task = rw.get_task(review_id)
        assert task is not None
        assert task['status'] == 'pending'
        assert task['priority'] == 'high'
        print(f"✅ 查询任务：状态={task['status']}, 优先级={task['priority']}")

        # 列表
        tasks = rw.list_tasks(status='pending')
        assert len(tasks) == 1
        print(f"✅ 待复核列表：{len(tasks)} 个")

        # 统计
        stats = rw.get_stats()
        assert stats['pending'] == 1
        assert stats['total'] == 1
        print(f"✅ 统计正确：待处理 {stats['pending']}，总计 {stats['total']}")

        # 确认原判定
        ok = rw.confirm(review_id, actor='tester', note='人工确认，原判定正确')
        assert ok
        task = rw.get_task(review_id)
        assert task['status'] == 'confirmed'
        assert task['resolved_by'] == 'tester'
        print(f"✅ 确认原判定：状态={task['status']}, 操作人={task['resolved_by']}")

        # 操作历史
        history = rw.get_action_history(review_id)
        assert len(history) >= 2  # created + confirmed
        print(f"✅ 操作历史记录：{len(history)} 条")

        # 待标注池
        ok = rw.add_to_annotation_pool(review_id, reason='困难样本，边界模糊')
        assert ok
        pool = rw.list_annotation_pool()
        assert len(pool) == 1
        assert pool[0]['used_for_training'] == 0  # 默认不用于训练
        print(f"✅ 加入待标注池：{len(pool)} 个，used_for_training={pool[0]['used_for_training']}")
        print("   （安全：不会自动用于训练）")

        print("🎉 测试5通过：复核工作流正确\n")
        return True


def test_area_union_calculation():
    """测试6：面积并集计算（重叠去重）"""
    print("=" * 60)
    print("测试6：面积并集计算（重叠去重）")
    print("=" * 60)

    engine = QualityEngineV2()

    # 两个完全重叠的框
    dets_overlap = [
        DetectionItem('crazing', 0.9, 100, 100, 200, 200),
        DetectionItem('crazing', 0.8, 100, 100, 200, 200),
    ]
    total_pct, max_pct = engine._calculate_box_areas(dets_overlap, 640, 640)
    # 两个完全重叠的框，并集 = 单个框面积
    single_area = 100 * 100
    expected = single_area / (640 * 640) * 100
    assert abs(total_pct - expected) < 0.1, f"并集计算错误：{total_pct} vs {expected}"
    print(f"✅ 完全重叠的两个框：并集 {total_pct:.4f}% ≈ 单框 {expected:.4f}%")

    # 两个不重叠的框
    dets_separate = [
        DetectionItem('crazing', 0.9, 100, 100, 200, 200),
        DetectionItem('crazing', 0.8, 300, 300, 400, 400),
    ]
    total_pct2, _ = engine._calculate_box_areas(dets_separate, 640, 640)
    expected2 = 2 * single_area / (640 * 640) * 100
    assert abs(total_pct2 - expected2) < 0.1
    print(f"✅ 不重叠的两个框：并集 {total_pct2:.4f}% ≈ 两框之和 {expected2:.4f}%")

    # 部分重叠
    dets_partial = [
        DetectionItem('crazing', 0.9, 100, 100, 200, 200),  # 100x100
        DetectionItem('crazing', 0.8, 150, 150, 250, 250),  # 100x100，重叠50x50
    ]
    total_pct3, _ = engine._calculate_box_areas(dets_partial, 640, 640)
    # 并集 = 100*100 + 100*100 - 50*50 = 10000 + 10000 - 2500 = 17500
    expected3 = 17500 / (640 * 640) * 100
    assert abs(total_pct3 - expected3) < 0.5, f"部分重叠计算错误：{total_pct3} vs {expected3}"
    print(f"✅ 部分重叠的两个框：并集 {total_pct3:.4f}% ≈ 预期 {expected3:.4f}%")

    print("🎉 测试6通过：面积并集计算正确\n")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("P0 核心功能单元测试")
    print("=" * 60 + "\n")

    tests = [
        test_task_context_scope_isolation,
        test_quality_safety_boundaries,
        test_execution_trace,
        test_tool_schema_validation,
        test_review_workflow,
        test_area_union_calculation,
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
    print(f"测试结果：{passed} 通过，{failed} 失败，共 {len(tests)} 项")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
