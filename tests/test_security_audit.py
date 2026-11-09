# -*- coding: utf-8 -*-
"""
安全审计验证测试

验证修复的安全问题：
1. 上下文检测结果深拷贝（外部修改不影响原始数据）
2. 复核操作权限验证（低角色不能执行高权限操作）
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent.task_context import TaskContext, DetectionItem, ImageInfo
from models.agent.review_workflow import (
    ReviewWorkflow, AuthorizationError, _check_permission, ROLE_PERMISSIONS
)


def test_detection_deep_copy():
    """测试检测结果深拷贝——外部修改不影响原始数据"""
    print("=" * 60)
    print("测试1：检测结果深拷贝保护")
    print("=" * 60)

    ctx = TaskContext('test_security')
    dets = [
        DetectionItem(cls_name='crazing', confidence=0.9,
                      x1=10, y1=10, x2=50, y2=50),
        DetectionItem(cls_name='inclusion', confidence=0.85,
                      x1=60, y1=60, x2=100, y2=100),
    ]
    img_info = ImageInfo(image_id='test', image_name='test.jpg',
                         width=200, height=200)
    ctx.set_raw_detections(dets, img_info)

    # 获取当前检测结果
    current = ctx.get_current_detections()
    assert len(current) == 2

    # 修改返回的对象
    original_conf = dets[0].confidence
    current[0].confidence = 0.999  # 修改副本
    current[0].cls_name = 'modified'  # 修改副本

    # 验证原始数据未被修改
    raw = ctx.get_raw_detections()
    assert raw[0].confidence == original_conf, \
        f"原始数据被修改了！原置信度 {original_conf}，现 {raw[0].confidence}"
    assert raw[0].cls_name == 'crazing', \
        f"原始类别被修改了！原 crazing，现 {raw[0].cls_name}"

    print(f"  ✅ 修改返回值不影响原始数据")
    print(f"  ✅ 原始置信度: {original_conf} (未被修改)")
    print(f"  ✅ 原始类别: crazing (未被修改)")

    # 验证子集视图也返回深拷贝
    view = ctx.filter_by_class('crazing')
    ctx.switch_to_view(view)
    subset = ctx.get_current_detections()
    assert len(subset) == 1
    subset[0].confidence = 0.111  # 修改子集副本

    raw2 = ctx.get_raw_detections()
    assert raw2[0].confidence == original_conf, \
        "子集视图的修改影响了原始数据"
    print(f"  ✅ 子集视图修改也不影响原始数据")

    print("\n✅ 检测结果深拷贝保护测试通过\n")
    return True


def test_review_permissions():
    """测试复核操作权限验证"""
    print("=" * 60)
    print("测试2：复核操作权限验证")
    print("=" * 60)

    # 验证各角色权限
    print("  角色权限矩阵:")
    for role, perms in ROLE_PERMISSIONS.items():
        can_reject = '✓' if perms['can_reject'] else '✗'
        can_correct = '✓' if perms['can_correct'] else '✗'
        can_confirm = '✓' if perms['can_confirm'] else '✗'
        print(f"    {role:12s}: confirm={can_confirm} reject={can_reject} correct={can_correct}")

    # 测试权限检查函数
    assert _check_permission('inspector', 'can_reject') == True
    assert _check_permission('operator', 'can_reject') == False
    assert _check_permission('viewer', 'can_confirm') == False
    assert _check_permission('supervisor', 'can_correct') == True

    # 测试未知角色抛出异常
    try:
        _check_permission('hacker', 'can_reject')
        assert False, "未知角色应该抛出异常"
    except AuthorizationError as e:
        print(f"\n  ✅ 未知角色抛出 AuthorizationError: {e}")

    # 测试实际复核操作的权限控制
    tmpdir = tempfile.mkdtemp(prefix='security_test_')
    db_path = os.path.join(tmpdir, 'review.db')
    rw = ReviewWorkflow(db_path)

    # 创建一个复核任务（用 system 角色）
    review_id = rw.create_task(
        task_id='t1',
        record_id=1,
        image_name='test.jpg',
        image_path='/tmp/test.jpg',
        original_quality='REVIEW',
        original_score=50.0,
        original_scope='单图全量',
        reason='测试权限',
        quality_issue='需复核',
        priority='high',
        model_version='test',
        rule_version='test_v1',
    )
    assert review_id is not None
    print(f"  ✅ 创建复核任务成功 (system 角色)")

    # 操作员尝试驳回（应该失败）
    try:
        rw.reject(review_id, actor='op1', role='operator',
                   reason='我要驳回', corrected_quality='PASS')
        assert False, "操作员不应能驳回"
    except AuthorizationError as e:
        print(f"  ✅ 操作员驳回被拒绝: {e}")

    # 质检员驳回（应该成功）
    result = rw.reject(review_id, actor='inspector1', role='inspector',
                       reason='确实误判', corrected_quality='PASS')
    assert result == True
    task = rw.get_task(review_id)
    assert task['status'] == 'rejected'
    print(f"  ✅ 质检员驳回成功")

    # 查看员尝试确认（应该失败）
    try:
        rw.confirm(review_id, actor='viewer1', role='viewer')
        assert False, "查看员不应能确认"
    except AuthorizationError as e:
        print(f"  ✅ 查看员确认被拒绝: {e}")

    # 主管修正检测结果（应该成功）
    review_id2 = rw.create_task(
        task_id='t2', record_id=2, image_name='test2.jpg',
        image_path='/tmp/test2.jpg', original_quality='FAIL',
        original_score=30.0, original_scope='单图全量',
        reason='测试修正', quality_issue='不合格',
        priority='high', model_version='test', rule_version='test_v1',
    )
    result = rw.correct_detections(
        review_id2,
        corrected_detections=[],
        corrected_eval={'quality': 'PASS', 'score': 95},
        actor='super1',
        role='supervisor',
        note='修正后合格'
    )
    assert result == True
    print(f"  ✅ 主管修正检测结果成功")

    # 操作员修正（应该失败）
    review_id3 = rw.create_task(
        task_id='t3', record_id=3, image_name='test3.jpg',
        image_path='/tmp/test3.jpg', original_quality='REVIEW',
        original_score=60.0, original_scope='单图全量',
        reason='测试修正2', quality_issue='需复核',
        priority='medium', model_version='test', rule_version='test_v1',
    )
    try:
        rw.correct_detections(
            review_id3, corrected_detections=[],
            corrected_eval={'quality': 'PASS', 'score': 100},
            actor='op2', role='operator'
        )
        assert False, "操作员不应能修正检测结果"
    except AuthorizationError as e:
        print(f"  ✅ 操作员修正检测结果被拒绝: {e}")

    print("\n✅ 复核操作权限验证测试通过\n")
    return True


def test_scope_whitelist():
    """测试范围访问白名单（参数枚举校验）"""
    print("=" * 60)
    print("测试3：范围访问显式性（参数白名单校验）")
    print("=" * 60)

    from models.agent.tool_schema import get_tool_schema

    # 检查 filter_by_class 工具的类别白名单
    schema = get_tool_schema('filter_by_class')
    assert schema is not None
    target_param = [p for p in schema.params if p.name == 'target_class'][0]
    assert target_param.choices is not None
    assert len(target_param.choices) == 6
    assert 'crazing' in target_param.choices
    assert 'inclusion' in target_param.choices

    print(f"  ✅ filter_by_class 的 target_class 有 {len(target_param.choices)} 个白名单类别")

    # 验证 validate_params 会拒绝非法值
    is_valid, errors, _ = schema.validate_params({'target_class': 'invalid_class'})
    assert is_valid == False
    assert len(errors) > 0
    print(f"  ✅ 非法类别值被拒绝: {errors[0][:50]}")

    # 检查 quality_judge 的 scope 白名单
    q_schema = get_tool_schema('quality_judge')
    assert q_schema is not None
    scope_param = [p for p in q_schema.params if p.name == 'scope'][0]
    assert scope_param.choices is not None
    assert 'full' in scope_param.choices
    assert 'current' in scope_param.choices
    print(f"  ✅ quality_judge 的 scope 有白名单: {scope_param.choices}")

    # 验证非法 scope 被拒绝
    is_valid2, errors2, _ = q_schema.validate_params({'scope': 'invalid_scope'})
    assert is_valid2 == False
    print(f"  ✅ 非法 scope 被拒绝")

    print("\n✅ 范围访问白名单测试通过\n")
    return True


def run_all():
    tests = [
        test_detection_deep_copy,
        test_review_permissions,
        test_scope_whitelist,
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
            print(f"💥 {test.__name__} 异常：{e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"安全审计验证: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
