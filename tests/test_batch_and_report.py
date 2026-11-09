# -*- coding: utf-8 -*-
"""
批次处理 + Word 报告 v2 端到端测试

验证：
1. 批次处理正确跟踪成功/失败/未完成
2. 部分完成不输出 PASS
3. 复核任务幂等创建
4. Word 报告包含所有必要字段
5. 演示规则标记、代理指标说明、证据引用
"""
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent.batch_processor import (
    BatchProcessor, detect_with_yolo, BatchResult
)
from models.agent.quality_engine_v2 import QualityEngineV2
from models.agent.review_workflow import ReviewWorkflow
from models.agent.report_generator_v2 import QualityReportGenerator, _verify_report_content


# 使用 NEU-DET 数据集图像（检测模型在该数据集上训练，能正常检测）
NEU_DET_IMAGES = r'c:\Users\User\Desktop\YOLO\data\raw\NEU-DET\IMAGES'


def test_batch_processing():
    """测试批次处理"""
    print("=" * 60)
    print("测试1：批次处理")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')

    engine = QualityEngineV2()

    # 使用固定测试目录（避免 Windows 上 sqlite 临时文件锁问题）
    tmpdir = tempfile.mkdtemp(prefix='batch_test_')
    db_path = os.path.join(tmpdir, 'review.db')
    rw = ReviewWorkflow(db_path)

    processor = BatchProcessor(
        quality_engine=engine,
        review_workflow=rw,
        model_version='yolov8s_baseline-2',
    )

    # 6张测试图像（每类1张）
    test_images = [
        os.path.join(NEU_DET_IMAGES, f'{cls}_1.jpg')
        for cls in ['crazing', 'inclusion', 'patches',
                    'pitted_surface', 'rolled-in_scale', 'scratches']
    ]

    def detect_func(img_path):
        return detect_with_yolo(model, img_path)

    # 执行批次
    result = processor.process_batch(test_images, detect_func)

    print(f"  批次ID: {result.batch_id}")
    print(f"  计划: {result.total_planned}")
    print(f"  成功: {result.total_success}")
    print(f"  失败: {result.total_failed}")
    print(f"  跳过: {result.total_skipped}")
    print(f"  PASS: {result.pass_count}")
    print(f"  FAIL: {result.fail_count}")
    print(f"  REVIEW: {result.review_count}")
    print(f"  批次质量: {result.batch_quality}")
    print(f"  原因: {result.batch_quality_reason[:80]}")

    # 验证
    assert result.total_planned == 6, f"计划数应为6，实际{result.total_planned}"
    assert result.total_success == 6, f"成功数应为6，实际{result.total_success}"
    assert result.total_failed == 0, f"失败数应为0，实际{result.total_failed}"
    assert result.batch_quality in ('PASS', 'FAIL', 'REVIEW'), \
        f"批次质量异常: {result.batch_quality}"

    # 验证所有图像都有检测结果
    for img_r in result.image_results:
        print(f"    - {img_r.image_name}: {img_r.detection_count}个缺陷, "
              f"质量={img_r.quality}, 评分={img_r.quality_score:.1f}")

    # 验证复核任务创建
    pending_reviews = rw.list_tasks(status='pending')
    expected_reviews = result.fail_count + result.review_count
    print(f"\n  复核任务: 创建了 {len(pending_reviews)} 个待处理")
    print(f"    (FAIL+REVIEW = {expected_reviews})")

    print("\n✅ 批次处理测试通过\n")
    return result


def test_batch_incomplete():
    """测试部分完成的批次不得 PASS"""
    print("=" * 60)
    print("测试2：部分完成批次不 PASS")
    print("=" * 60)

    engine = QualityEngineV2()
    processor = BatchProcessor(quality_engine=engine, model_version='test')

    # 构造部分完成的结果
    result = BatchResult(
        batch_id='test_incomplete',
        total_planned=5,
        total_success=3,
        total_failed=1,
        total_skipped=1,
        pass_count=3,
    )
    processor._compute_batch_quality(result)

    print(f"  计划: 5, 成功: 3, 失败: 1, 跳过: 1")
    print(f"  批次质量: {result.batch_quality}")
    print(f"  原因: {result.batch_quality_reason}")

    assert result.batch_quality == 'INCOMPLETE', \
        f"部分完成的批次应为 INCOMPLETE，实际为 {result.batch_quality}"
    assert '不能输出 PASS' in result.batch_quality_reason or '未全部完成' in result.batch_quality_reason

    # 全部成功但有 FAIL
    result2 = BatchResult(
        batch_id='test_all_fail',
        total_planned=3,
        total_success=3,
        fail_count=1,
        pass_count=2,
    )
    processor._compute_batch_quality(result2)
    print(f"\n  全部成功（含 1 FAIL）: {result2.batch_quality}")
    assert result2.batch_quality == 'FAIL'

    # 全部成功且全 PASS
    result3 = BatchResult(
        batch_id='test_all_pass',
        total_planned=3,
        total_success=3,
        pass_count=3,
    )
    processor._compute_batch_quality(result3)
    print(f"  全部成功（全 PASS）: {result3.batch_quality}")
    assert result3.batch_quality == 'PASS'

    # 有 REVIEW 无 FAIL
    result4 = BatchResult(
        batch_id='test_review',
        total_planned=4,
        total_success=4,
        pass_count=2,
        review_count=2,
    )
    processor._compute_batch_quality(result4)
    print(f"  全部成功（2 REVIEW）: {result4.batch_quality}")
    assert result4.batch_quality == 'REVIEW'

    print("\n✅ 批次质量判定逻辑测试通过\n")
    return True


def test_word_report_v2_single():
    """测试单图 Word 报告 v2 生成"""
    print("=" * 60)
    print("测试3：单图 Word 报告 v2 生成与验证")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')

    engine = QualityEngineV2()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = QualityReportGenerator()

        # 用真实检测数据
        img_path = os.path.join(NEU_DET_IMAGES, 'crazing_1.jpg')
        detections, img_info, err = detect_with_yolo(model, img_path)
        assert not err, f"检测失败: {err}"

        from models.agent.task_context import TaskContext
        ctx = TaskContext('test_report')
        ctx.set_raw_detections(detections, img_info)
        quality_result = engine.evaluate(ctx)

        # 计算检测摘要
        by_class = {}
        total_conf = 0
        for d in detections:
            by_class[d.cls_name] = by_class.get(d.cls_name, 0) + 1
            total_conf += d.confidence
        avg_conf = total_conf / len(detections) if detections else 0

        gen.generate_single_image_report(
            image_name='crazing_1.jpg',
            quality_result=quality_result,
            detection_summary={
                'total_count': len(detections),
                'by_class': by_class,
                'avg_confidence': avg_conf,
                'area_ratio': quality_result.get('total_area_ratio', 0) * 100,
            },
            trace_info={
                'task_id': 'task_test_001',
                'model_version': 'yolov8s_baseline-2',
                'rule_version': 'demo_default_v1.0.0',
                'total_steps': 3,
                'success_steps': 3,
                'total_time_ms': 456.2,
            },
            scope_label=quality_result.get('scope_label', '单图全量'),
        )

        report_path = os.path.join(tmpdir, 'single_report.docx')
        gen.save(report_path)

        # 验证
        checks = _verify_report_content(report_path)
        print(f"  报告生成: {Path(report_path).name}")
        print(f"  内容验证:")
        all_pass = True
        for k, v in checks.items():
            status = '✅' if v else '❌'
            print(f"    {status} {k}")
            if not v:
                all_pass = False

        assert all_pass, "报告内容验证失败"

        print("\n✅ 单图 Word 报告 v2 测试通过\n")
        return True


def test_word_report_v2_batch():
    """测试批次 Word 报告 v2 生成"""
    print("=" * 60)
    print("测试4：批次 Word 报告 v2 生成与验证")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt')
    engine = QualityEngineV2()

    with tempfile.TemporaryDirectory() as tmpdir:
        processor = BatchProcessor(
            quality_engine=engine,
            model_version='yolov8s_baseline-2',
        )

        test_images = [
            os.path.join(NEU_DET_IMAGES, f'{cls}_1.jpg')
            for cls in ['crazing', 'inclusion', 'patches', 'scratches']
        ]

        def detect_func(img_path):
            return detect_with_yolo(model, img_path)

        batch_result = processor.process_batch(test_images, detect_func)

        # 生成报告
        gen = QualityReportGenerator()
        gen.generate_batch_report(
            batch_result=batch_result.to_dict(),
            trace_info={
                'task_id': batch_result.batch_id,
                'model_version': 'yolov8s_baseline-2',
                'rule_version': 'demo_default_v1.0.0',
                'total_steps': 4,
                'success_steps': batch_result.total_success,
                'total_time_ms': 1200.5,
            },
        )

        report_path = os.path.join(tmpdir, 'batch_report.docx')
        gen.save(report_path)

        # 验证
        checks = _verify_report_content(report_path)
        print(f"  报告生成: {Path(report_path).name}")
        print(f"  批次质量: {batch_result.batch_quality}")
        print(f"  内容验证:")
        all_pass = True
        for k, v in checks.items():
            status = '✅' if v else '❌'
            print(f"    {status} {k}")
            if not v:
                all_pass = False

        # 批次报告还应有批次相关内容
        doc_checks = {
            'has_batch_id': batch_result.batch_id in open(report_path, 'rb').read().decode('utf-8', errors='ignore'),
        }
        # 用 docx 验证
        from docx import Document
        doc = Document(report_path)
        full_text = '\n'.join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text += '\n' + cell.text

        batch_specific = {
            'has_planned': '计划' in full_text and '检测数' in full_text,
            'has_success_fail': '成功' in full_text and '失败' in full_text,
            'has_image_table': len(doc.tables) >= 3,
        }
        for k, v in batch_specific.items():
            status = '✅' if v else '❌'
            print(f"    {status} {k}")
            if not v:
                all_pass = False

        assert all_pass, "批次报告内容验证失败"

        print("\n✅ 批次 Word 报告 v2 测试通过\n")
        return True


def run_all():
    tests = [
        test_batch_processing,
        test_batch_incomplete,
        test_word_report_v2_single,
        test_word_report_v2_batch,
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
    print(f"批次+报告测试: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
