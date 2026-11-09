# -*- coding: utf-8 -*-
"""
报告中心页面 —— v2 版 8 大章节 Word 报告

功能：
1. 接入 QualityReportGenerator v2（8 大章节）
2. 支持批次报告和单图报告
3. 显示报告内容验证状态
4. 直接下载 Word 报告
5. 历史记录查询
"""

import streamlit as st
import os
import json
import io
import pandas as pd
from datetime import datetime

from models.agent.report_generator_v2 import (
    QualityReportGenerator,
    _verify_report_content,
)
from models.agent.review_workflow import ReviewWorkflow


# 类别中文名映射
CLASS_NAMES_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化皮',
    'scratches': '划痕',
}


def render_report_page(agent, report_gen_class, review_workflow: ReviewWorkflow, base_dir: str):
    """渲染报告中心页面"""

    st.markdown('<div class="main-header">📋 报告中心（v2）</div>', unsafe_allow_html=True)
    st.caption("基于 QualityReportGenerator v2，生成包含 8 大章节的可追溯 Word 报告")

    # 报告类型选择
    tab1, tab2 = st.tabs(["📄 单图报告", "📊 批次/统计报告"])

    with tab1:
        _render_single_report_tab(agent, report_gen_class, review_workflow)

    with tab2:
        _render_batch_report_tab(agent, report_gen_class)


def _render_single_report_tab(agent, report_gen_class, review_workflow):
    """单图报告生成"""
    st.subheader("📄 单图检测报告")

    # 记录选择
    st.markdown("#### 选择检测记录")

    col1, col2 = st.columns([1, 3])
    with col1:
        filter_quality = st.selectbox(
            "质量筛选",
            ["全部", "PASS", "REVIEW", "FAIL"],
            key="report_quality_filter",
        )

    quality_param = None if filter_quality == "全部" else filter_quality

    try:
        records = agent.query_records(quality=quality_param, limit=100)
    except Exception:
        records = []

    st.metric("可用记录", len(records))

    if not records:
        st.info("暂无检测记录，请先在检测模块进行检测")
        return

    # 记录列表
    df = pd.DataFrame(records)
    df_display = df[['id', 'image_name', 'timestamp', 'total_defects', 'quality', 'quality_score']].copy()
    df_display.columns = ['ID', '图片名', '检测时间', '缺陷数', '质量等级', '评分']

    # 转换质量等级为徽章
    def quality_badge(q):
        badges = {
            'PASS': '<span class="quality-badge quality-pass" style="font-size:12px;padding:2px 8px;">✅ PASS</span>',
            'REVIEW': '<span class="quality-badge quality-review" style="font-size:12px;padding:2px 8px;">⚠️ REVIEW</span>',
            'FAIL': '<span class="quality-badge quality-fail" style="font-size:12px;padding:2px 8px;">❌ FAIL</span>',
        }
        return badges.get(q, q)

    # 使用 HTML 表格显示可选择的记录
    selected_id = st.session_state.get('report_selected_id', records[0]['id'] if records else None)

    for _, row in df_display.iterrows():
        rid = int(row['ID'])
        is_selected = rid == selected_id

        col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 2, 1])
        with col1:
            st.write(f"**#{rid}**")
        with col2:
            st.write(row['图片名'])
            st.caption(str(row['检测时间'])[:19])
        with col3:
            st.write(f"{row['缺陷数']} 个")
        with col4:
            st.markdown(quality_badge(row['质量等级']), unsafe_allow_html=True)
            st.caption(f"评分: {row['评分']}")
        with col5:
            if st.button("选择", key=f"select_record_{rid}", use_container_width=True):
                st.session_state['report_selected_id'] = rid
                st.rerun()

        if is_selected:
            st.markdown("---")
            # 显示选中记录的详情
            _render_record_detail(agent, rid)
            st.markdown("---")

    # 生成报告按钮
    if selected_id:
        st.markdown("#### 📤 生成报告")

        # 报告选项
        col1, col2 = st.columns(2)
        with col1:
            include_review = st.checkbox(
                "包含复核信息",
                value=False,
                help="如果该记录有关联的复核任务，将复核信息加入报告",
                key="report_include_review",
            )
        with col2:
            include_trace = st.checkbox(
                "包含执行轨迹摘要",
                value=True,
                help="在报告中加入执行轨迹的摘要信息",
                key="report_include_trace",
            )

        if st.button("📄 生成 Word 报告", type="primary", use_container_width=True):
            _generate_single_report(agent, report_gen_class, review_workflow, selected_id,
                                    include_review, include_trace)


def _render_record_detail(agent, record_id):
    """渲染记录详情"""
    try:
        records = agent.query_records(limit=1000)
        record = next((r for r in records if r['id'] == record_id), None)
    except Exception:
        record = None

    if not record:
        st.warning("未找到记录")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("图片名", record['image_name'])
    col2.metric("缺陷数量", record['total_defects'])

    quality = record['quality']
    badges = {
        'PASS': '<span class="quality-badge quality-pass">✅ PASS 合格</span>',
        'REVIEW': '<span class="quality-badge quality-review">⚠️ REVIEW 待复核</span>',
        'FAIL': '<span class="quality-badge quality-fail">❌ FAIL 不合格</span>',
    }
    col3.markdown(f"### {badges.get(quality, quality)}", unsafe_allow_html=True)

    col1.metric("综合评分", f"{record['quality_score']} 分")

    # 检测详情
    with st.expander("🔍 检测详情"):
        try:
            dets = json.loads(record.get('detections_json', '[]'))
        except Exception:
            dets = []

        if dets:
            det_df = pd.DataFrame(dets, columns=['类别', '置信度', 'x1', 'y1', 'x2', 'y2'])
            det_df['类别中文'] = det_df['类别'].map(
                lambda x: CLASS_NAMES_CN.get(x, x)
            )
            st.dataframe(det_df[['类别', '类别中文', '置信度']], use_container_width=True)
        else:
            st.caption("无检测数据")

    # 判定详情
    with st.expander("📖 判定详情"):
        try:
            eval_data = json.loads(record.get('eval_json', '{}'))
        except Exception:
            eval_data = {}
        st.write(eval_data.get('explanation', '暂无说明'))


def _generate_single_report(agent, report_gen_class, review_workflow, record_id,
                            include_review, include_trace):
    """生成单图报告"""
    with st.spinner("正在生成报告..."):
        try:
            # 获取记录
            records = agent.query_records(limit=1000)
            record = next((r for r in records if r['id'] == record_id), None)

            if not record:
                st.error("未找到记录")
                return

            # 解析检测数据
            try:
                dets = json.loads(record.get('detections_json', '[]'))
            except Exception:
                dets = []

            # 解析评估数据
            try:
                eval_data = json.loads(record.get('eval_json', '{}'))
            except Exception:
                eval_data = {}

            # 构建检测摘要
            by_class = {}
            total_conf = 0
            for d in dets:
                cls_name = d[0] if isinstance(d, (list, tuple)) else d.get('cls_name', 'unknown')
                conf = d[1] if isinstance(d, (list, tuple)) else d.get('confidence', 0)
                by_class[cls_name] = by_class.get(cls_name, 0) + 1
                total_conf += conf
            avg_conf = total_conf / len(dets) if dets else 0

            total_area_pct = eval_data.get('total_box_area_pct',
                                           eval_data.get('total_area_ratio', 0))

            detection_summary = {
                'total_count': len(dets),
                'by_class': by_class,
                'avg_confidence': avg_conf,
                'area_ratio': total_area_pct,
            }

            # 构建轨迹信息
            trace_info = {
                'task_id': record.get('task_id', f'record_{record_id}'),
                'model_version': record.get('model_version', getattr(agent, 'model_description', 'unknown')),
                'rule_version': record.get('rule_version', 'demo_default_v1.0.0'),
                'total_steps': 0,
                'success_steps': 0,
                'total_time_ms': 0,
            }

            # 构建质量结果（确保包含所有必要字段）
            quality_result = {
                'quality': record.get('quality', ''),
                'score': record.get('quality_score', 0),
                'explanation': eval_data.get('explanation', ''),
                'scope_label': record.get('scope_label', '单图全量'),
                'is_demo_rule': True,
                'rule_id': 'demo_default',
                'rule_version': '1.0.0',
                'rule_source': '演示规则',
                'total_box_area_pct': total_area_pct,
                'is_forced': eval_data.get('is_forced', False),
                'forced_reason': eval_data.get('forced_reason', ''),
                'method_notes': eval_data.get('method_notes', [
                    '⚠️ 演示规则：本判定基于实验演示规则，不构成工业质量判定依据',
                    '面积指标为「检测框覆盖率」，是真实缺陷面积的代理指标，非物理测量值',
                ]),
                'reasons_fail': eval_data.get('reasons_fail', []),
                'reasons_review': eval_data.get('reasons_review', []),
            }

            # 复核信息
            review_info = None
            if include_review:
                # 尝试查找相关复核任务
                try:
                    review_tasks = review_workflow.list_tasks(limit=100)
                    for rt in review_tasks:
                        if rt.get('record_id') == record_id or rt.get('image_name') == record['image_name']:
                            review_info = {
                                'task_id': rt.get('review_id', ''),
                                'status': rt.get('status', ''),
                                'actor': rt.get('resolved_by', ''),
                                'completed_at': rt.get('resolved_at', ''),
                                'note': rt.get('resolution_note', ''),
                                'final_quality': rt.get('resolution', ''),
                                'final_reason': rt.get('resolution_note', ''),
                            }
                            break
                except Exception:
                    pass

            # 生成报告
            gen = QualityReportGenerator()
            gen.generate_single_image_report(
                image_name=record['image_name'],
                quality_result=quality_result,
                detection_summary=detection_summary,
                trace_info=trace_info,
                review_info=review_info,
                scope_label=record.get('scope_label', '单图全量'),
                is_subset=record.get('scope') in ('subset', 'batch_subset'),
                subset_class='',
            )

            # 保存到内存
            buf = io.BytesIO()
            gen.doc.save(buf)
            buf.seek(0)

            # 内容验证
            file_bytes = buf.getvalue()
            tmp_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs", "temp",
                f"report_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            )
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, 'wb') as f:
                f.write(file_bytes)

            checks = _verify_report_content(tmp_path)

            # 清理临时文件
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            # 显示验证结果
            st.success("✅ 报告生成成功！")

            col1, col2 = st.columns(2)
            with col1:
                passed = sum(1 for v in checks.values() if v is True)
                total = len(checks)
                st.metric("内容验证", f"{passed}/{total} 通过")

            with col2:
                # 下载按钮
                st.download_button(
                    label="📥 下载 Word 报告",
                    data=file_bytes,
                    file_name=f"质检报告_{record['image_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    type="primary",
                )

            # 详细验证结果
            with st.expander("📋 报告内容验证详情"):
                for k, v in checks.items():
                    status_icon = "✅" if v else "❌"
                    st.write(f"{status_icon} {k}")

            # 报告章节预览
            with st.expander("📑 报告章节概览"):
                chapters = [
                    "一、报告基本信息",
                    "二、质量结论",
                    "三、检测结果",
                    "四、方法说明",
                    "五、版本信息",
                    "六、人工复核（如有）",
                    "七、证据引用",
                    "八、限制与声明",
                ]
                for ch in chapters:
                    st.write(f"📄 {ch}")

        except ImportError as e:
            st.error(f"报告生成失败：缺少依赖 - {str(e)}")
            st.caption("请确保已安装 python-docx：pip install python-docx")
        except Exception as e:
            st.error(f"报告生成失败：{str(e)}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())


def _render_batch_report_tab(agent, report_gen_class):
    """批次/统计报告"""
    st.subheader("📊 批次/统计汇总报告")

    st.info(
        "📌 批次报告功能基于所有历史检测记录生成统计汇总。\n\n"
        "报告包含：批次基本信息、质量分布、各图像检测详情、版本信息、方法说明、证据引用、限制声明等 8 大章节。"
    )

    # 统计数据
    try:
        stats = agent.get_statistics()
    except Exception:
        stats = {}

    # 概览
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总记录数", stats.get('total_records', 0))
    col2.metric("合格数", stats.get('pass_count', 0))
    col3.metric("不合格数", stats.get('fail_count', 0))
    col4.metric("待复核数", stats.get('review_count', 0))

    st.markdown("---")

    # 报告选项
    st.markdown("#### 报告选项")

    col1, col2 = st.columns(2)
    with col1:
        report_scope = st.selectbox(
            "报告范围",
            ["全部记录", "最近 50 条", "最近 100 条", "最近 200 条"],
            key="batch_report_scope",
        )
    with col2:
        include_details = st.checkbox(
            "包含每条记录详情",
            value=True,
            help="在报告中列出每条检测记录的详细信息",
            key="batch_include_details",
        )

    # 获取记录
    scope_limit_map = {
        "全部记录": 1000,
        "最近 50 条": 50,
        "最近 100 条": 100,
        "最近 200 条": 200,
    }
    limit = scope_limit_map.get(report_scope, 100)

    try:
        records = agent.query_records(limit=limit)
    except Exception:
        records = []

    st.caption(f"将基于 {len(records)} 条记录生成报告")

    if not records:
        st.info("暂无检测记录")
        return

    # 生成按钮
    if st.button("📊 生成批次统计报告", type="primary", use_container_width=True):
        _generate_batch_report(agent, report_gen_class, records)


def _generate_batch_report(agent, report_gen_class, records):
    """生成批次统计报告"""
    with st.spinner("正在生成批次报告..."):
        try:
            # 计算统计数据
            total = len(records)
            pass_count = sum(1 for r in records if r.get('quality') == 'PASS')
            fail_count = sum(1 for r in records if r.get('quality') == 'FAIL')
            review_count = sum(1 for r in records if r.get('quality') == 'REVIEW')

            # 批次质量判定
            if fail_count > 0:
                batch_quality = 'FAIL'
                batch_reason = f'批次中有 {fail_count} 个不合格产品'
            elif review_count > 0:
                batch_quality = 'REVIEW'
                batch_reason = f'批次中有 {review_count} 个待复核产品'
            else:
                batch_quality = 'PASS'
                batch_reason = '全部产品检测合格'

            # 构建各图像结果
            images = []
            for r in records:
                images.append({
                    'image_name': r.get('image_name', ''),
                    'status': 'success',
                    'detection_count': r.get('total_defects', 0),
                    'quality': r.get('quality', ''),
                    'quality_score': r.get('quality_score', 0),
                    'is_forced': False,
                })

            batch_result = {
                'batch_id': f'batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'summary': {
                    'planned': total,
                    'success': total,
                    'failed': 0,
                    'skipped': 0,
                    'pass': pass_count,
                    'fail': fail_count,
                    'review': review_count,
                },
                'batch_quality': batch_quality,
                'batch_quality_reason': batch_reason,
                'model_version': getattr(agent, 'model_description', 'unknown'),
                'rule_version': 'demo_default_v1.0.0',
                'rule_is_demo': True,
                'start_time': records[-1].get('timestamp', '') if records else '',
                'end_time': records[0].get('timestamp', '') if records else '',
                'images': images,
            }

            # 轨迹信息
            trace_info = {
                'task_id': f'batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'model_version': getattr(agent, 'model_description', 'unknown'),
                'rule_version': 'demo_default_v1.0.0',
                'total_steps': len(records),
                'success_steps': total,
                'total_time_ms': 0,
            }

            # 生成报告
            gen = QualityReportGenerator()
            gen.generate_batch_report(
                batch_result=batch_result,
                trace_info=trace_info,
            )

            # 保存到内存
            buf = io.BytesIO()
            gen.doc.save(buf)
            buf.seek(0)

            # 内容验证
            file_bytes = buf.getvalue()
            tmp_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs", "temp",
                f"batch_report_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            )
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, 'wb') as f:
                f.write(file_bytes)

            checks = _verify_report_content(tmp_path)

            # 清理临时文件
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            # 显示结果
            st.success("✅ 批次报告生成成功！")

            col1, col2 = st.columns(2)
            with col1:
                passed = sum(1 for v in checks.values() if v is True)
                total_checks = len(checks)
                st.metric("内容验证", f"{passed}/{total_checks} 通过")

            with col2:
                st.download_button(
                    label="📥 下载批次报告",
                    data=file_bytes,
                    file_name=f"批次质检报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    type="primary",
                )

            # 质量分布
            st.markdown("#### 📊 质量分布")
            dist_col1, dist_col2, dist_col3 = st.columns(3)
            dist_col1.metric("✅ PASS", pass_count)
            dist_col2.metric("⚠️ REVIEW", review_count)
            dist_col3.metric("❌ FAIL", fail_count)

            # 详细验证
            with st.expander("📋 报告内容验证详情"):
                for k, v in checks.items():
                    status_icon = "✅" if v else "❌"
                    st.write(f"{status_icon} {k}")

        except ImportError as e:
            st.error(f"报告生成失败：缺少依赖 - {str(e)}")
            st.caption("请确保已安装 python-docx：pip install python-docx")
        except Exception as e:
            st.error(f"报告生成失败：{str(e)}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
