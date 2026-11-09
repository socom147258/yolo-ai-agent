# -*- coding: utf-8 -*-
"""
Word 报告 v2 生成器

生成包含以下内容的可追溯报告：
- 分析范围（单图/批次、全量/子集）
- 模型版本、规则版本、演示规则标记
- 检测结果汇总
- 质量判定依据（规则、阈值）
- 覆盖率方法说明（检测框覆盖率，代理指标）
- 证据引用（执行轨迹、检测记录）
- 人工复核状态（复核前后结论）
- 方法备注和限制说明
"""
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 确保可以导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


class QualityReportGenerator:
    """
    质检报告生成器 v2

    所有结论可追溯至工具结果和执行轨迹。
    明确标注演示规则、代理指标、范围限制。
    """

    def __init__(self):
        self.doc = Document()
        self._setup_styles()

    def _setup_styles(self):
        """设置样式"""
        style = self.doc.styles['Normal']
        style.font.name = '微软雅黑'
        style.font.size = Pt(10.5)
        style._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

        # 标题样式
        for level in range(1, 4):
            heading = self.doc.styles[f'Heading {level}']
            heading.font.name = '微软雅黑'
            heading._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
            heading.font.color.rgb = RGBColor(0x2E, 0x86, 0xAB)

    def add_title(self, title: str):
        """添加报告标题"""
        p = self.doc.add_heading(title, level=0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_info_table(self, info_pairs: List[tuple]):
        """添加信息表（键值对）"""
        table = self.doc.add_table(rows=len(info_pairs), cols=2)
        table.style = 'Light Grid Accent 1'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        for i, (key, value) in enumerate(info_pairs):
            cell_key = table.cell(i, 0)
            cell_val = table.cell(i, 1)
            cell_key.text = str(key)
            cell_val.text = str(value)

            # 键单元格加粗
            for p in cell_key.paragraphs:
                for run in p.runs:
                    run.bold = True

            # 设置宽度
            cell_key.width = Cm(4)
            cell_val.width = Cm(12)

        return table

    def add_section_title(self, title: str, level: int = 1):
        """添加章节标题"""
        self.doc.add_heading(title, level=level)

    def add_paragraph(self, text: str, bold: bool = False, color: str = None):
        """添加段落"""
        p = self.doc.add_paragraph()
        run = p.add_run(text)
        run.bold = bold
        if color:
            run.font.color.rgb = RGBColor.from_string(color)
        return p

    def add_warning_box(self, text: str):
        """添加警告框（黄色背景）"""
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.right_indent = Cm(0.5)

        # 添加底纹
        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'FFF3CD')
        pPr.append(shd)

        run = p.add_run(f'⚠️  {text}')
        run.font.color.rgb = RGBColor(0x85, 0x64, 0x03)
        return p

    def add_info_box(self, text: str):
        """添加信息框（蓝色背景）"""
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.right_indent = Cm(0.5)

        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'D1ECF1')
        pPr.append(shd)

        run = p.add_run(f'ℹ️  {text}')
        run.font.color.rgb = RGBColor(0x0C, 0x54, 0x6B)
        return p

    def add_quality_badge(self, quality: str, reason: str = ''):
        """添加质量等级徽章"""
        colors = {
            'PASS': ('28A745', '通过'),
            'FAIL': ('DC3545', '不合格'),
            'REVIEW': ('FFC107', '需复核'),
            'INCOMPLETE': ('6C757D', '未完成'),
        }

        color_hex, label = colors.get(quality, ('6C757D', quality))

        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f'【{quality}】 {label}')
        run.bold = True
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor.from_string(color_hex)

        if reason:
            p2 = self.doc.add_paragraph()
            p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run2 = p2.add_run(reason)
            run2.font.size = Pt(10)
            run2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    def add_data_table(self, headers: List[str], rows: List[List[str]]):
        """添加数据表"""
        table = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.style = 'Light Grid Accent 1'

        # 表头
        for i, h in enumerate(headers):
            cell = table.cell(0, i)
            cell.text = h
            for p in cell.paragraphs:
                for run in p.runs:
                    run.bold = True

        # 数据
        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                table.cell(r_idx + 1, c_idx).text = str(val)

        return table

    def add_bullet_list(self, items: List[str]):
        """添加项目符号列表"""
        for item in items:
            p = self.doc.add_paragraph(item, style='List Bullet')
        return p

    # ============================================================
    # 生成完整报告
    # ============================================================

    def generate_single_image_report(self, image_name: str, quality_result: Dict,
                                     detection_summary: Dict, trace_info: Dict,
                                     review_info: Dict = None,
                                     scope_label: str = '单图全量',
                                     is_subset: bool = False,
                                     subset_class: str = '') -> str:
        """
        生成单图检测报告

        Args:
            image_name: 图像文件名
            quality_result: 质量判定结果（来自 QualityEngineV2）
            detection_summary: 检测结果汇总
            trace_info: 执行轨迹信息
            review_info: 复核信息（如有）
            scope_label: 范围标签
            is_subset: 是否为子集分析
            subset_class: 子集类别

        Returns:
            报告文件路径
        """
        # 标题
        self.add_title('表面缺陷质量检测报告')

        # 基本信息
        self.add_section_title('一、报告基本信息', 1)
        quality = quality_result.get('quality', 'UNKNOWN')
        info = [
            ('报告编号', f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            ('检测对象', image_name),
            ('分析范围', scope_label),
            ('报告生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            ('报告类型', '演示报告' if quality_result.get('is_demo_rule', True) else '正式报告'),
        ]
        self.add_info_table(info)

        # 子集分析警告
        if is_subset:
            self.add_warning_box(
                f'本报告为子集分析结果（仅分析 {subset_class} 类缺陷），'
                f'不代表整图整体质量。如需整体质量结论，请切换至全量范围重新判定。'
            )

        # 质量结论
        self.add_section_title('二、质量结论', 1)
        self.add_quality_badge(
            quality,
            quality_result.get('forced_reason', '')
        )

        if quality_result.get('is_forced'):
            self.add_warning_box(
                f'本结果为强制判定：{quality_result.get("forced_reason", "")}'
            )

        # 评分详情
        self.add_paragraph(f'质量评分：{quality_result.get("score", 0):.1f} / 100')
        if quality_result.get('scoring_details'):
            details = quality_result['scoring_details']
            self.add_paragraph('评分构成：')
            for item in details:
                self.add_paragraph(f'  • {item}')

        # 检测结果
        self.add_section_title('三、检测结果', 1)
        det_info = [
            ('检测缺陷总数', str(detection_summary.get('total_count', 0))),
            ('缺陷类别数', str(len(detection_summary.get('by_class', {})))),
            ('平均置信度', f"{detection_summary.get('avg_confidence', 0):.3f}"),
            ('检测框覆盖率', f"{detection_summary.get('area_ratio', 0):.2f}%"),
        ]
        self.add_info_table(det_info)

        # 按类别分布
        by_class = detection_summary.get('by_class', {})
        if by_class:
            self.add_paragraph('各类缺陷分布：', bold=True)
            rows = [[cls, str(cnt)] for cls, cnt in sorted(by_class.items())]
            self.add_data_table(['缺陷类别', '数量'], rows)

        # 方法说明
        self.add_section_title('四、方法说明', 1)
        self.add_info_box(
            '面积指标为「检测框覆盖率」，使用检测框并集面积（扫描线法去重）计算，'
            '是真实缺陷面积的代理指标，不等同于实际缺陷面积。本系统未进行物理标定，'
            '不提供像素与实际尺寸的换算。'
        )
        self.add_paragraph('')
        self.add_info_box(
            f'当前使用的质量规则为演示规则 '
            f'（{quality_result.get("rule_id", "demo_default")} '
            f'v{quality_result.get("rule_version", "1.0.0")}），'
            f'仅供验证系统功能使用，不构成工业质量判定依据。'
        )

        # 版本信息
        self.add_section_title('五、版本信息', 1)
        version_info = [
            ('检测模型', trace_info.get('model_version', 'unknown')),
            ('质量规则版本', trace_info.get('rule_version', 'unknown')),
            ('规则来源', quality_result.get('rule_source', '演示规则')),
            ('执行引擎版本', 'Agent v2'),
        ]
        self.add_info_table(version_info)

        # 复核信息（如有）
        if review_info:
            self.add_section_title('六、人工复核', 1)
            review_status = review_info.get('status', 'unknown')
            review_status_map = {
                'pending': '待处理',
                'confirmed': '已确认（原判定正确）',
                'rejected': '已驳回（原判定有误）',
                'modified': '已修正',
                'closed': '已关闭',
            }
            review_data = [
                ('复核任务号', review_info.get('task_id', '')),
                ('复核状态', review_status_map.get(review_status, review_status)),
                ('初始判定', quality),
                ('初始评分', f"{quality_result.get('score', 0):.1f}"),
                ('复核人', review_info.get('actor', '-')),
                ('复核时间', review_info.get('completed_at', '-')),
                ('复核备注', review_info.get('note', '-')),
            ]
            self.add_info_table(review_data)

            if review_info.get('final_quality'):
                final_q = review_info['final_quality']
                self.add_paragraph('复核后结论：', bold=True)
                self.add_quality_badge(final_q, review_info.get('final_reason', ''))

        # 证据引用
        self.add_section_title('七、证据引用', 1)
        self.add_bullet_list([
            f"任务ID: {trace_info.get('task_id', 'N/A')}",
            f"执行轨迹: 共 {trace_info.get('total_steps', 0)} 步，"
            f"成功 {trace_info.get('success_steps', 0)} 步，"
            f"总耗时 {trace_info.get('total_time_ms', 0):.0f}ms",
            f"检测数据来源: {image_name}",
            f"轨迹可导出为 JSON 用于审计追溯",
        ])

        # 限制声明
        self.add_section_title('八、限制与声明', 1)
        self.add_warning_box(
            '本报告基于演示规则生成，仅供系统功能验证使用，不构成任何工业质量判定依据。'
            '实际生产环境中的质量判定应由经过培训的质检人员结合专业标准进行。'
        )
        self.add_paragraph('')
        self.add_paragraph(
            '未检测到缺陷仅表示模型未识别出已知类别的缺陷，不等于产品合格。'
            'REVIEW 结论表示系统无法确定，需人工复核后才能最终判定。'
        )

        return self

    def generate_batch_report(self, batch_result: Dict, trace_info: Dict) -> str:
        """
        生成批次检测报告

        Args:
            batch_result: 批次处理结果（来自 BatchProcessor）
            trace_info: 执行轨迹信息

        Returns:
            报告文件路径
        """
        # 标题
        self.add_title('批次表面缺陷质量检测报告')

        # 基本信息
        self.add_section_title('一、批次基本信息', 1)
        summary = batch_result.get('summary', {})
        info = [
            ('批次编号', batch_result.get('batch_id', '')),
            ('计划检测数', str(summary.get('planned', 0))),
            ('成功检测数', str(summary.get('success', 0))),
            ('失败数', str(summary.get('failed', 0))),
            ('跳过数', str(summary.get('skipped', 0))),
            ('检测开始时间', batch_result.get('start_time', '')),
            ('检测完成时间', batch_result.get('end_time', '')),
        ]
        self.add_info_table(info)

        # 批次质量结论
        self.add_section_title('二、批次质量结论', 1)
        batch_quality = batch_result.get('batch_quality', 'INCOMPLETE')
        batch_reason = batch_result.get('batch_quality_reason', '')

        self.add_quality_badge(batch_quality, batch_reason)

        if batch_quality == 'INCOMPLETE':
            self.add_warning_box(
                '批次未全部完成，不得输出 PASS 结论。请处理失败或跳过的图像后重新评估。'
            )

        # 质量分布
        self.add_section_title('三、质量分布', 1)
        dist_data = [
            ['PASS', str(summary.get('pass', 0))],
            ['FAIL', str(summary.get('fail', 0))],
            ['REVIEW', str(summary.get('review', 0))],
        ]
        self.add_data_table(['质量等级', '图像数'], dist_data)

        # 各图像结果
        self.add_section_title('四、各图像检测详情', 1)
        images = batch_result.get('images', [])
        if images:
            rows = []
            for img in images:
                status_map = {'success': '成功', 'failed': '失败', 'pending': '等待', 'skipped': '跳过'}
                status_cn = status_map.get(img.get('status', ''), img.get('status', ''))
                rows.append([
                    img.get('image_name', ''),
                    status_cn,
                    str(img.get('detection_count', 0)),
                    img.get('quality', '-'),
                    f"{img.get('quality_score', 0):.1f}",
                    '是' if img.get('is_forced') else '否',
                ])
            self.add_data_table(
                ['图像名', '状态', '缺陷数', '质量等级', '评分', '强制判定'],
                rows
            )

        # 版本信息
        self.add_section_title('五、版本信息', 1)
        version_info = [
            ('检测模型', batch_result.get('model_version', 'unknown')),
            ('质量规则版本', batch_result.get('rule_version', 'unknown')),
            ('规则类型', '演示规则' if batch_result.get('rule_is_demo', True) else '正式规则'),
            ('执行引擎', 'Agent v2 Batch Processor'),
        ]
        self.add_info_table(version_info)

        # 方法说明
        self.add_section_title('六、方法说明', 1)
        self.add_info_box(
            '面积指标为「检测框覆盖率」，使用检测框并集面积（扫描线法去重）计算，'
            '是真实缺陷面积的代理指标，不等同于实际缺陷面积。'
        )
        self.add_paragraph('')
        self.add_info_box(
            f'质量规则为演示规则（{batch_result.get("rule_version", "")}），'
            f'仅供验证系统功能使用，不构成工业质量判定依据。'
        )

        # 证据引用
        self.add_section_title('七、证据引用', 1)
        self.add_bullet_list([
            f"批次ID: {batch_result.get('batch_id', 'N/A')}",
            f"执行轨迹: 共 {trace_info.get('total_steps', 0)} 步，"
            f"成功 {trace_info.get('success_steps', 0)} 步，"
            f"总耗时 {trace_info.get('total_time_ms', 0):.0f}ms",
            f"检测模型: {trace_info.get('model_version', 'unknown')}",
            f"轨迹可导出为 JSON 用于审计追溯",
        ])

        # 限制声明
        self.add_section_title('八、限制与声明', 1)
        self.add_warning_box(
            '本报告基于演示规则生成，仅供系统功能验证使用，不构成任何工业质量判定依据。'
        )
        self.add_paragraph('')
        self.add_paragraph(
            '批次质量判定仅针对本次检测的图像范围，未检测的图像不包含在结论中。'
            '部分完成的批次不得输出 PASS 结论。'
        )

        return self

    def save(self, output_path: str) -> str:
        """保存报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(output_path))
        return str(output_path)


def _verify_report_content(file_path: str) -> Dict[str, bool]:
    """
    验证报告内容是否包含必要字段

    返回各项检查结果
    """
    try:
        doc = Document(str(file_path))

        # 收集所有文本（段落 + 表格）
        all_text_parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    all_text_parts.append(cell.text)
        full_text = '\n'.join(all_text_parts)

        checks = {
            'has_title': len(doc.paragraphs) > 0 and doc.paragraphs[0].text,
            'has_quality': any(q in full_text for q in ['PASS', 'FAIL', 'REVIEW', 'INCOMPLETE']),
            'has_scope': '范围' in full_text,
            'has_model_version': '模型' in full_text,
            'has_rule_version': '规则' in full_text,
            'has_demo_marker': '演示规则' in full_text or '演示' in full_text,
            'has_method_note': '代理指标' in full_text or '检测框覆盖率' in full_text,
            'has_evidence': '任务ID' in full_text or '证据' in full_text,
            'has_limitations': '限制' in full_text or '声明' in full_text,
            'has_tables': len(doc.tables) > 0,
        }

        return checks
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    # 简单测试
    gen = QualityReportGenerator()
    gen.generate_single_image_report(
        image_name='test.jpg',
        quality_result={
            'quality': 'PASS',
            'score': 85.5,
            'scope_label': '单图全量',
            'is_forced': False,
            'forced_reason': '',
            'is_demo_rule': True,
            'rule_id': 'demo_default',
            'rule_version': '1.0.0',
            'rule_source': '演示规则',
            'scoring_details': ['缺陷数量得分: 30/30', '严重程度得分: 25.5/30', '置信度得分: 30/40'],
        },
        detection_summary={
            'total_count': 5,
            'by_class': {'crazing': 3, 'scratches': 2},
            'avg_confidence': 0.87,
            'area_ratio': 2.35,
        },
        trace_info={
            'task_id': 'task_test_001',
            'model_version': 'yolov8s_test',
            'rule_version': 'demo_default_v1.0.0',
            'total_steps': 3,
            'success_steps': 3,
            'total_time_ms': 523.5,
        },
        scope_label='单图全量',
    )

    out = gen.save('outputs/reports/test_report.docx')
    print(f"测试报告已生成: {out}")

    # 验证
    checks = _verify_report_content(out)
    print(f"\n内容验证:")
    for k, v in checks.items():
        status = '✅' if v else '❌'
        print(f"  {status} {k}")
