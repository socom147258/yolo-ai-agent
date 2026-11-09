# -*- coding: utf-8 -*-
"""
Agent 集成报告工具验证测试

验证 Agent 执行器中的 generate_report 工具可以生成真实的 Word 报告。
证明"报告闭环"功能已完成。
"""
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent.agent_executor_v2 import AgentExecutorV2, AgentConfig
from models.agent.report_generator_v2 import _verify_report_content
from models.agent.execution_trace import ExecutionTrace
from models.agent.task_context import TaskContext, DetectionItem, ImageInfo


NEU_DET_IMAGES = r'c:\Users\User\Desktop\YOLO\data\raw\NEU-DET\IMAGES'
MODEL_PATH = r'c:\Users\User\Desktop\YOLO\outputs\baseline\yolov8s_baseline-2\weights\best.pt'


def test_generate_report_docx_direct():
    """直接测试 generate_report 工具生成真实 Word 文件"""
    print("=" * 60)
    print("测试1：generate_report 工具生成 DOCX 报告")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)

    tmpdir = tempfile.mkdtemp(prefix='agent_report_test_')

    config = AgentConfig(
        max_steps=10,
        llm_provider='mock',
        auto_save_trace=False,
        report_output_dir=tmpdir,
    )
    agent = AgentExecutorV2(config=config, model=model)

    img_path = os.path.join(NEU_DET_IMAGES, 'crazing_1.jpg')

    # 先执行检测
    from models.agent.batch_processor import detect_with_yolo
    detections, img_info, err = detect_with_yolo(model, img_path)
    assert not err, f"检测失败: {err}"

    # 手动构建上下文和轨迹
    ctx = TaskContext('test_docx_report')
    ctx.model_version = 'yolov8s_baseline-2'
    ctx.set_raw_detections(detections, img_info)

    trace = ExecutionTrace(task_id='test_docx_report',
                           model_version='yolov8s_baseline-2',
                           rule_version='demo_default_v1.0.0')

    # 直接调用 generate_report 工具
    result = agent._tool_generate_report(
        report_type='detail',
        format='docx',
        _ctx=ctx,
        _trace=trace,
    )

    print(f"  返回 message: {result.get('message', '')}")
    print(f"  返回 format: {result.get('format', '')}")
    print(f"  download_path: {result.get('download_path', 'None')}")

    assert result.get('format') == 'docx', "格式应为 docx"
    assert result.get('download_path') is not None, "应返回文件路径"
    assert os.path.exists(result['download_path']), "文件应存在"

    # 验证报告内容
    checks = _verify_report_content(result['download_path'])
    print(f"\n  报告内容验证:")
    all_pass = True
    for k, v in checks.items():
        status = '✅' if v else '❌'
        print(f"    {status} {k}")
        if not v:
            all_pass = False

    assert all_pass, "报告内容验证失败"

    print("\n✅ generate_report DOCX 工具正常\n")
    return True


def test_generate_report_json_format():
    """测试 generate_report 工具的 JSON 格式"""
    print("=" * 60)
    print("测试2：generate_report JSON 格式")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)

    config = AgentConfig(
        max_steps=10,
        llm_provider='mock',
        auto_save_trace=False,
    )
    agent = AgentExecutorV2(config=config, model=model)

    img_path = os.path.join(NEU_DET_IMAGES, 'crazing_1.jpg')
    from models.agent.batch_processor import detect_with_yolo
    detections, img_info, err = detect_with_yolo(model, img_path)

    ctx = TaskContext('test_json_report')
    ctx.model_version = 'yolov8s_baseline-2'
    ctx.set_raw_detections(detections, img_info)
    trace = ExecutionTrace(task_id='test_json_report')

    result = agent._tool_generate_report(
        report_type='summary',
        format='json',
        _ctx=ctx,
        _trace=trace,
    )

    assert 'report_data' in result
    assert result.get('format') == 'json'
    assert result.get('download_path') is None  # JSON 格式不生成文件

    data = result['report_data']
    assert data['task_id'] == 'test_json_report'
    assert data['total_defects'] == len(detections)
    assert 'quality_result' in data
    assert 'model_version' in data
    assert 'rule_version' in data

    print(f"  报告数据包含字段: {list(data.keys())}")
    print(f"  缺陷数: {data['total_defects']}")
    print(f"  质量: {data['quality_result'].get('quality', 'N/A')}")

    print("\n✅ JSON 格式报告正常\n")
    return True


def test_subset_scope_report():
    """测试子集范围下生成报告（标注子集分析）"""
    print("=" * 60)
    print("测试3：子集范围下的报告生成")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)

    tmpdir = tempfile.mkdtemp(prefix='agent_subset_report_')

    config = AgentConfig(
        max_steps=10,
        llm_provider='mock',
        auto_save_trace=False,
        report_output_dir=tmpdir,
    )
    agent = AgentExecutorV2(config=config, model=model)

    img_path = os.path.join(NEU_DET_IMAGES, 'scratches_1.jpg')
    from models.agent.batch_processor import detect_with_yolo
    detections, img_info, err = detect_with_yolo(model, img_path)

    ctx = TaskContext('test_subset_report')
    ctx.model_version = 'yolov8s_baseline-2'
    ctx.set_raw_detections(detections, img_info)

    # 切换到子集范围（只看 scratches）
    view = ctx.filter_by_class('scratches')
    ctx.switch_to_view(view)

    trace = ExecutionTrace(task_id='test_subset_report')

    result = agent._tool_generate_report(
        report_type='detail',
        format='docx',
        _ctx=ctx,
        _trace=trace,
    )

    assert result.get('download_path') is not None

    # 读取报告内容，检查是否包含子集标注
    from docx import Document
    doc = Document(result['download_path'])
    all_text_parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                all_text_parts.append(cell.text)
    full_text = '\n'.join(all_text_parts)

    has_subset_warning = '子集' in full_text
    has_scope_info = '范围' in full_text

    print(f"  报告包含子集标注: {'✅' if has_subset_warning else '❌'}")
    print(f"  报告包含范围字段: {'✅' if has_scope_info else '❌'}")
    print(f"  范围标签: {ctx.scope_label}")

    assert has_subset_warning, f"报告应包含'子集'标注，实际内容含'范围':{has_scope_info}"

    # 内容验证
    checks = _verify_report_content(result['download_path'])
    passed = sum(1 for v in checks.values() if v)
    print(f"  内容验证: {passed}/{len(checks)} 通过")

    print("\n✅ 子集范围报告生成正常\n")
    return True


def test_generate_report_in_tool_list():
    """验证 generate_report 在可用工具列表中（LLM 可调用）"""
    print("=" * 60)
    print("测试4：generate_report 在工具白名单中")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)

    config = AgentConfig(llm_provider='mock')
    agent = AgentExecutorV2(config=config, model=model)

    tools = agent.get_available_tools()
    has_generate_report = 'generate_report' in tools

    print(f"  可用工具数: {len(tools)}")
    print(f"  包含 generate_report: {'✅ 是' if has_generate_report else '❌ 否'}")

    # 打印所有工具
    for t in tools:
        print(f"    - {t}")

    assert has_generate_report, "generate_report 应在工具白名单中"

    print("\n✅ generate_report 在工具白名单中，LLM 可调用\n")
    return True


def run_all():
    tests = [
        test_generate_report_docx_direct,
        test_generate_report_json_format,
        test_subset_scope_report,
        test_generate_report_in_tool_list,
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
    print(f"Agent报告集成测试: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 60)

    if failed == 0:
        print("🎉 报告闭环功能已完成：Agent 可直接调用 generate_report 生成 Word 报告")
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
