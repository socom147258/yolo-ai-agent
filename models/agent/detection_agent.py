# -*- coding: utf-8 -*-
"""
AI Agent 智能质量检测系统
==========================
- 三级质量判定引擎（PASS / REVIEW / FAIL）
- 工具化 Agent 协调层（任务理解 → 工具选择 → 执行 → 结果整合 → 解释）
- SQLite 持久化存储
"""

import os
import re
import json
import sqlite3
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# 1. 三级质量判定引擎
# ============================================================

class QualityRuleEngine:
    """
    三级质量判定规则引擎
    - PASS：合格
    - REVIEW：待人工复核（置信度低/边界模糊）
    - FAIL：不合格

    判定维度：
    1. 缺陷类别严重等级
    2. 缺陷数量密度
    3. 缺陷面积占比
    4. 检测置信度（模型可信度）
    """

    # 缺陷类别严重等级（1-5，5最严重）
    CLASS_SEVERITY = {
        'crazing': 5,      # 裂纹 - 最严重（结构性缺陷）
        'inclusion': 4,    # 夹杂 - 严重（异物嵌入）
        'scratches': 3,    # 划痕 - 中等
        'patches': 3,      # 斑块 - 中等
        'pitted_surface': 2,  # 麻点 - 较轻
        'rolled-in_scale': 2,  # 氧化皮 - 较轻
    }

    def __init__(self, config=None):
        """
        初始化规则引擎（所有参数可配置）
        """
        # 默认配置（实验演示用，非生产标准）
        self.config = {
            # 缺陷数量阈值
            'max_minor_defects': 5,       # 轻度缺陷超过多少个 -> FAIL
            'max_critical_defects': 1,    # 严重缺陷超过多少个 -> FAIL
            'review_low_confidence': 0.5,  # 低于此置信度 -> REVIEW
            'fail_severity_threshold': 4,  # 严重等级>=此值视为致命缺陷
            # 面积阈值（相对图像面积百分比）
            'max_total_area_pct': 15.0,    # 总缺陷面积超过15% -> FAIL
            'max_single_area_pct': 5.0,    # 单个缺陷面积超过5% -> FAIL
            # 复核阈值
            'review_low_conf_count': 2,    # 低置信度检测>=2个 -> REVIEW
            'review_borderline_score': 70, # 评分接近合格线(60-70) -> REVIEW
        }
        if config:
            self.config.update(config)

    def evaluate(self, detections, img_width=640, img_height=640):
        """
        评估检测结果，给出三级质量判定
        Args:
            detections: list of [cls_name, conf, x1, y1, x2, y2]
            img_width: 图像实际宽度（用于计算面积占比）
            img_height: 图像实际高度（用于计算面积占比）
        Returns:
            dict: 完整的质量评估结果

        注意：本判定为演示规则。当检测结果为零时判定为 PASS，
        仅表示模型未检出已知缺陷，不代表产品一定符合实际生产质量标准。
        """
        total_count = len(detections)
        img_area = max(1, img_width) * max(1, img_height)  # 防止除零

        # 分类统计
        by_class = {}
        critical_count = 0  # 严重缺陷数
        low_conf_count = 0  # 低置信度数
        total_defect_area = 0.0  # 总缺陷面积
        max_single_area = 0.0  # 最大单缺陷面积
        max_severity = 0  # 最高严重等级

        for det in detections:
            cls_name, conf, x1, y1, x2, y2 = det
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h
            area_pct = area / img_area * 100

            severity = self.CLASS_SEVERITY.get(cls_name, 2)

            if cls_name not in by_class:
                by_class[cls_name] = {'count': 0, 'total_area': 0.0, 'max_conf': 0, 'min_conf': 1.0}
            by_class[cls_name]['count'] += 1
            by_class[cls_name]['total_area'] += area_pct
            by_class[cls_name]['max_conf'] = max(by_class[cls_name]['max_conf'], conf)
            by_class[cls_name]['min_conf'] = min(by_class[cls_name]['min_conf'], conf)

            if severity >= self.config['fail_severity_threshold']:
                critical_count += 1

            if conf < self.config['review_low_confidence']:
                low_conf_count += 1

            total_defect_area += area_pct
            max_single_area = max(max_single_area, area_pct)
            max_severity = max(max_severity, severity)

        # ---- 评分计算（百分制）----
        score = 100
        reasons_fail = []
        reasons_review = []

        # 扣分：严重缺陷
        if critical_count > self.config['max_critical_defects']:
            penalty = (critical_count - self.config['max_critical_defects']) * 25
            score -= penalty
            reasons_fail.append(
                f'严重缺陷过多 ({critical_count} > {self.config["max_critical_defects"]}个，'
                f'涉及裂纹/夹杂等高危险性缺陷)'
            )

        # 扣分：总缺陷数量
        if total_count > self.config['max_minor_defects']:
            penalty = (total_count - self.config['max_minor_defects']) * 5
            score -= penalty
            reasons_fail.append(
                f'缺陷数量偏多 ({total_count} > {self.config["max_minor_defects"]}个)'
            )

        # 扣分：总缺陷面积
        if total_defect_area > self.config['max_total_area_pct']:
            penalty = 15
            score -= penalty
            reasons_fail.append(
                f'缺陷总面积过大 ({total_defect_area:.1f}% > {self.config["max_total_area_pct"]}%)'
            )

        # 扣分：单缺陷超大
        if max_single_area > self.config['max_single_area_pct']:
            penalty = 10
            score -= penalty
            reasons_fail.append(
                f'存在超大缺陷 (单缺陷{max_single_area:.1f}% > {self.config["max_single_area_pct"]}%)'
            )

        # 复核因素：低置信度
        if low_conf_count >= self.config['review_low_conf_count']:
            reasons_review.append(
                f'{low_conf_count} 个检测置信度低于 {self.config["review_low_confidence"]}，模型识别存疑'
            )

        # 复核因素：评分接近合格线
        if 60 <= score < self.config['review_borderline_score']:
            reasons_review.append(
                f'评分接近合格线 ({score:.0f}分)，建议人工复核确认'
            )

        # 复核因素：边界模糊缺陷
        borderline_classes = [c for c, d in by_class.items() if d['max_conf'] - d['min_conf'] > 0.4]
        if borderline_classes:
            reasons_review.append(
                f'部分缺陷置信度差异较大 ({"、".join(borderline_classes)})，可能存在边界模糊'
            )

        score = max(0, min(100, int(score)))

        # ---- 三级判定 ----
        if score < 60:
            quality = 'FAIL'
        elif reasons_review and score < 80:
            quality = 'REVIEW'
        elif low_conf_count >= 3:
            quality = 'REVIEW'
        else:
            quality = 'PASS'

        # 生成解释文本
        explanation = self._generate_explanation(
            quality, score, total_count, critical_count,
            total_defect_area, by_class, reasons_fail, reasons_review
        )

        return {
            'quality': quality,
            'score': score,
            'total_defects': total_count,
            'critical_count': critical_count,
            'low_conf_count': low_conf_count,
            'total_area_pct': round(total_defect_area, 2),
            'max_single_area_pct': round(max_single_area, 2),
            'max_severity': max_severity,
            'by_class': by_class,
            'reasons_fail': reasons_fail,
            'reasons_review': reasons_review,
            'explanation': explanation,
        }

    def _generate_explanation(self, quality, score, total_count, critical_count,
                              total_area, by_class, reasons_fail, reasons_review):
        """生成自然语言解释"""
        lines = []

        if quality == 'PASS':
            lines.append(f"✅ 质量判定：合格（PASS），综合评分 {score} 分。")
            if total_count == 0:
                lines.append("未检测到本系统识别范围内的缺陷。")
                lines.append("（注：此为演示判定规则，仅表示模型未检出已知缺陷类别，不代表产品符合实际生产质量标准。）")
            else:
                lines.append(f"共检测到 {total_count} 个缺陷，均在可接受范围内。")
                cls_summary = "、".join([f"{c}({d['count']}个)" for c, d in by_class.items()])
                lines.append(f"缺陷类型：{cls_summary}。")
        elif quality == 'REVIEW':
            lines.append(f"⚠️ 质量判定：待复核（REVIEW），综合评分 {score} 分。")
            lines.append(f"共检测到 {total_count} 个缺陷。")
            if reasons_review:
                lines.append("复核原因：")
                for r in reasons_review:
                    lines.append(f"  • {r}")
            lines.append("建议质检人员进行人工确认。")
        else:  # FAIL
            lines.append(f"❌ 质量判定：不合格（FAIL），综合评分 {score} 分。")
            lines.append(f"共检测到 {total_count} 个缺陷，其中严重缺陷 {critical_count} 个。")
            if reasons_fail:
                lines.append("不合格原因：")
                for r in reasons_fail:
                    lines.append(f"  • {r}")
            if reasons_review:
                lines.append("附加说明：")
                for r in reasons_review:
                    lines.append(f"  • {r}")

        return "\n".join(lines)


# ============================================================
# 2. 工具集（Agent 可调用的工具）
# ============================================================

class ToolRegistry:
    """工具注册表 - Agent 可以调用的所有工具"""

    def __init__(self, agent):
        self.agent = agent
        self.tools = {
            'detect': {
                'name': '缺陷检测',
                'description': '对图像进行缺陷检测，返回缺陷类别、位置、置信度',
                'keywords': ['检测', '识别', '找出', '看看有什么缺陷', '检测一下', '查一下'],
                'func': self.tool_detect,
            },
            'count_defects': {
                'name': '缺陷统计',
                'description': '统计缺陷数量和类别分布',
                'keywords': ['统计', '数量', '多少个', '计数', '分布'],
                'func': self.tool_count_defects,
            },
            'classify_filter': {
                'name': '缺陷类别筛选',
                'description': '按类别筛选特定类型的缺陷',
                'keywords': ['裂纹', '夹杂', '划痕', '斑块', '麻点', '氧化皮',
                             'crazing', 'inclusion', 'scratches', 'patches'],
                'func': self.tool_classify_filter,
            },
            'quality_evaluate': {
                'name': '质量判定',
                'description': '根据检测结果进行质量等级判定',
                'keywords': ['合格吗', '质量', '判定', '评估', '评级', 'pass还是fail', '能不能过'],
                'func': self.tool_quality_evaluate,
            },
            'explain_result': {
                'name': '结果解释',
                'description': '解释为什么判定为某个质量等级',
                'keywords': ['为什么', '原因', '解释', '说明', '哪里有问题', '为什么不合格'],
                'func': self.tool_explain_result,
            },
            'save_record': {
                'name': '保存记录',
                'description': '将检测结果保存到数据库',
                'keywords': ['保存', '记录', '存起来', '存档'],
                'func': self.tool_save_record,
            },
            'generate_report': {
                'name': '生成报告',
                'description': '生成检测报告',
                'keywords': ['报告', '生成报告', '导出', '报表'],
                'func': self.tool_generate_report,
            },
            'batch_detect_folder': {
                'name': '批量检测文件夹',
                'description': '批量检测文件夹中的所有图片',
                'keywords': ['批量检测', '文件夹', '全部检测', '所有图片', '批次'],
                'func': self.tool_batch_detect_folder,
            },
            'batch_quality_stats': {
                'name': '批次质量统计',
                'description': '统计批次合格率、不合格数等',
                'keywords': ['合格率', '质量统计', '批次统计', '合格多少'],
                'func': self.tool_batch_quality_stats,
            },
            'batch_defect_distribution': {
                'name': '批次缺陷分布',
                'description': '统计批次中各类缺陷的分布',
                'keywords': ['缺陷分布', '各类缺陷', '分别有多少'],
                'func': self.tool_batch_defect_distribution,
            },
            'batch_list_failed': {
                'name': '不合格列表',
                'description': '列出不合格和待复核的图片',
                'keywords': ['哪些不合格', '不合格列表', '失败列表'],
                'func': self.tool_batch_list_failed,
            },
        }

    def get_tool(self, tool_name):
        return self.tools.get(tool_name, {}).get('func')

    def list_tools(self):
        return [(k, v['name'], v['description']) for k, v in self.tools.items()]

    # ---- 工具实现 ----

    def tool_detect(self, image_path, **kwargs):
        """工具：缺陷检测"""
        results = self.agent._run_detection(image_path)
        return {
            'tool': 'detect',
            'status': 'success',
            'detections': results['detections'],
            'total_count': len(results['detections']),
            'inference_time_ms': results['inference_time_ms'],
        }

    def tool_count_defects(self, detections=None, **kwargs):
        """工具：缺陷统计"""
        if detections is None:
            return {'tool': 'count_defects', 'status': 'error', 'message': '需要先执行检测'}

        by_class = {}
        for det in detections:
            cls_name = det[0]
            by_class[cls_name] = by_class.get(cls_name, 0) + 1

        return {
            'tool': 'count_defects',
            'status': 'success',
            'total_count': len(detections),
            'by_class': by_class,
            'summary': f'共检测到 {len(detections)} 个缺陷，涉及 {len(by_class)} 种类别',
        }

    def tool_classify_filter(self, detections=None, target_class=None, **kwargs):
        """工具：按类别筛选缺陷"""
        if detections is None:
            return {'tool': 'classify_filter', 'status': 'error', 'message': '需要先执行检测'}

        # 中英文映射
        class_map = {
            '裂纹': 'crazing', 'crazing': 'crazing',
            '夹杂': 'inclusion', 'inclusion': 'inclusion',
            '划痕': 'scratches', 'scratches': 'scratches',
            '斑块': 'patches', 'patches': 'patches',
            '麻点': 'pitted_surface', 'pitted_surface': 'pitted_surface',
            '氧化皮': 'rolled-in_scale', 'rolled-in_scale': 'rolled-in_scale',
        }

        mapped_class = class_map.get(target_class, target_class)
        filtered = [d for d in detections if d[0] == mapped_class]

        return {
            'tool': 'classify_filter',
            'status': 'success',
            'target_class': mapped_class,
            'filtered_count': len(filtered),
            'filtered_detections': filtered,
            'summary': f'筛选出 {len(filtered)} 个{mapped_class}类缺陷' if filtered else f'未检测到{mapped_class}类缺陷',
        }

    def tool_quality_evaluate(self, detections=None, img_width=640, img_height=640, **kwargs):
        """工具：质量判定"""
        if detections is None:
            return {'tool': 'quality_evaluate', 'status': 'error', 'message': '需要先执行检测'}

        eval_result = self.agent.rule_engine.evaluate(
            detections,
            img_width=img_width,
            img_height=img_height,
        )
        return {
            'tool': 'quality_evaluate',
            'status': 'success',
            **eval_result,
        }

    def tool_explain_result(self, detections=None, eval_result=None, **kwargs):
        """工具：结果解释"""
        if eval_result is None:
            if detections is None:
                return {'tool': 'explain_result', 'status': 'error', 'message': '需要先执行检测和判定'}
            eval_result = self.agent.rule_engine.evaluate(detections)

        return {
            'tool': 'explain_result',
            'status': 'success',
            'explanation': eval_result['explanation'],
            'quality': eval_result['quality'],
            'score': eval_result['score'],
        }

    def tool_save_record(self, image_name=None, detections=None, eval_result=None, **kwargs):
        """工具：保存记录"""
        if detections is None:
            return {'tool': 'save_record', 'status': 'error', 'message': '需要先执行检测'}

        if eval_result is None:
            eval_result = self.agent.rule_engine.evaluate(detections)

        record_id = self.agent._save_record(
            image_name=image_name or 'unknown',
            detections=detections,
            eval_result=eval_result,
        )
        return {
            'tool': 'save_record',
            'status': 'success',
            'record_id': record_id,
            'message': f'记录已保存 (ID: {record_id})',
        }

    def tool_generate_report(self, **kwargs):
        """工具：生成报告"""
        stats = self.agent.get_statistics()
        return {
            'tool': 'generate_report',
            'status': 'success',
            'statistics': stats,
            'message': f'报告生成完成，累计检测 {stats["total_records"]} 条记录',
        }

    def tool_batch_detect_folder(self, folder_path=None, **kwargs):
        """工具：批量检测文件夹中的所有图片"""
        if folder_path is None or not folder_path.strip():
            return {
                'tool': 'batch_detect_folder',
                'status': 'error',
                'message': '请提供文件夹路径'
            }

        folder_path = folder_path.strip().strip('"').strip("'")

        if not os.path.isdir(folder_path):
            return {
                'tool': 'batch_detect_folder',
                'status': 'error',
                'message': f'文件夹不存在：{folder_path}'
            }

        # 支持的图片扩展名
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        image_files = []
        for f in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(f)[1]
            if ext in exts:
                image_files.append(os.path.join(folder_path, f))

        if not image_files:
            return {
                'tool': 'batch_detect_folder',
                'status': 'error',
                'message': f'文件夹中没有找到支持的图片文件（支持 jpg/jpeg/png/bmp）'
            }

        # 批量检测
        results = []
        total_detections = 0
        by_class_total = {}
        pass_count = 0
        review_count = 0
        fail_count = 0

        for img_path in image_files:
            try:
                det_result = self.agent._run_detection(img_path)
                detections = det_result['detections']
                total_detections += len(detections)

                # 统计类别
                for det in detections:
                    cls_name = det[0]
                    by_class_total[cls_name] = by_class_total.get(cls_name, 0) + 1

                # 质量判定
                eval_result = self.agent.rule_engine.evaluate(
                    detections,
                    img_width=det_result.get('img_width', 640),
                    img_height=det_result.get('img_height', 640),
                )
                quality = eval_result.get('quality', 'UNKNOWN')
                if quality == 'PASS':
                    pass_count += 1
                elif quality == 'REVIEW':
                    review_count += 1
                else:
                    fail_count += 1

                results.append({
                    'image_name': os.path.basename(img_path),
                    'image_path': img_path,
                    'detection_count': len(detections),
                    'quality': quality,
                    'score': eval_result.get('score', 0),
                    'detections': detections,
                })
            except Exception as e:
                results.append({
                    'image_name': os.path.basename(img_path),
                    'image_path': img_path,
                    'error': str(e),
                    'detection_count': 0,
                    'quality': 'ERROR',
                })

        total_images = len(image_files)
        pass_rate = (pass_count / total_images * 100) if total_images > 0 else 0

        summary = (
            f'批量检测完成：共 {total_images} 张图片，检测到 {total_detections} 个缺陷\n'
            f'质量分布：✅ 合格 {pass_count} 张（{pass_rate:.1f}%）| '
            f'⚠️ 待复核 {review_count} 张 | ❌ 不合格 {fail_count} 张'
        )

        return {
            'tool': 'batch_detect_folder',
            'status': 'success',
            'folder_path': folder_path,
            'total_images': total_images,
            'total_detections': total_detections,
            'by_class_total': by_class_total,
            'pass_count': pass_count,
            'review_count': review_count,
            'fail_count': fail_count,
            'pass_rate': round(pass_rate, 1),
            'results': results,
            'summary': summary,
        }

    def tool_batch_quality_stats(self, batch_results=None, **kwargs):
        """工具：批量质量统计（合格率、不合格数等）"""
        if not batch_results:
            return {
                'tool': 'batch_quality_stats',
                'status': 'error',
                'message': '暂无批量检测结果，请先执行批量检测'
            }

        total = len(batch_results)
        pass_count = sum(1 for r in batch_results if r.get('quality') == 'PASS')
        fail_count = sum(1 for r in batch_results if r.get('quality') == 'FAIL')
        review_count = sum(1 for r in batch_results if r.get('quality') == 'REVIEW')
        total_detections = sum(r.get('detection_count', 0) for r in batch_results)
        pass_rate = (pass_count / total * 100) if total > 0 else 0

        summary = (
            f'📊 批次质量统计（共 {total} 张）：\n'
            f'  ✅ 合格：{pass_count} 张（{pass_rate:.1f}%）\n'
            f'  ⚠️ 待复核：{review_count} 张\n'
            f'  ❌ 不合格：{fail_count} 张\n'
            f'  🔍 缺陷总数：{total_detections} 个'
        )

        return {
            'tool': 'batch_quality_stats',
            'status': 'success',
            'total': total,
            'pass_count': pass_count,
            'fail_count': fail_count,
            'review_count': review_count,
            'pass_rate': round(pass_rate, 1),
            'total_detections': total_detections,
            'summary': summary,
        }

    def tool_batch_defect_distribution(self, batch_results=None, **kwargs):
        """工具：批量缺陷类别分布统计"""
        if not batch_results:
            return {
                'tool': 'batch_defect_distribution',
                'status': 'error',
                'message': '暂无批量检测结果，请先执行批量检测'
            }

        by_class = {}
        total_detections = 0
        for r in batch_results:
            detections = r.get('detections', [])
            total_detections += len(detections)
            for det in detections:
                cls_name = det[0] if isinstance(det, (list, tuple)) else det.get('class_name', 'unknown')
                by_class[cls_name] = by_class.get(cls_name, 0) + 1

        # 中文名映射
        class_cn = {
            'crazing': '裂纹', 'inclusion': '夹杂', 'patches': '斑块',
            'pitted_surface': '麻点', 'rolled-in_scale': '氧化皮', 'scratches': '划痕'
        }

        # 按数量排序
        sorted_items = sorted(by_class.items(), key=lambda x: x[1], reverse=True)

        lines = [f'📈 缺陷类别分布（共 {total_detections} 个缺陷）：']
        for cls, cnt in sorted_items:
            cn = class_cn.get(cls, cls)
            pct = (cnt / total_detections * 100) if total_detections > 0 else 0
            bar = '█' * int(pct / 2)
            lines.append(f'  {cn}({cls}): {cnt} 个 ({pct:.1f}%) {bar}')

        if not sorted_items:
            lines.append('  （无缺陷）')

        return {
            'tool': 'batch_defect_distribution',
            'status': 'success',
            'by_class': by_class,
            'total_detections': total_detections,
            'summary': '\n'.join(lines),
        }

    def tool_batch_list_failed(self, batch_results=None, **kwargs):
        """工具：列出不合格/待复核的图片"""
        if not batch_results:
            return {
                'tool': 'batch_list_failed',
                'status': 'error',
                'message': '暂无批量检测结果，请先执行批量检测'
            }

        failed = [r for r in batch_results if r.get('quality') == 'FAIL']
        review = [r for r in batch_results if r.get('quality') == 'REVIEW']

        lines = []
        if failed:
            lines.append(f'❌ 不合格图片（{len(failed)} 张）：')
            for r in failed:
                lines.append(f'  • {r["image_name"]} - 缺陷 {r["detection_count"]} 个，评分 {r.get("score", 0):.1f}')

        if review:
            lines.append(f'⚠️ 待复核图片（{len(review)} 张）：')
            for r in review:
                lines.append(f'  • {r["image_name"]} - 缺陷 {r["detection_count"]} 个，评分 {r.get("score", 0):.1f}')

        if not failed and not review:
            lines.append('✅ 所有图片均合格，没有不合格或待复核的图片')

        return {
            'tool': 'batch_list_failed',
            'status': 'success',
            'failed': failed,
            'review': review,
            'summary': '\n'.join(lines),
        }


# ============================================================
# 3. AI Agent 协调层（任务理解 + 工具规划 + 执行）
# ============================================================

class AIAgent:
    """
    AI Agent 智能体协调层

    工作流程：
    1. 任务理解（解析自然语言指令，识别意图和参数）
    2. 任务规划（选择需要调用的工具序列）
    3. 工具执行（按顺序调用工具，传递中间结果）
    4. 结果整合（汇总所有工具输出，生成最终回答）
    5. 结果解释（用自然语言解释结果）

    说明：本系统采用基于关键词匹配的有限任务规划实现，
    不依赖外部大语言模型，可在本地独立运行。
    """

    CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    CLASS_NAMES_CN = ['裂纹', '夹杂', '斑块', '麻点', '氧化皮', '划痕']

    def __init__(self, model_path, db_path=None):
        # 加载检测模型
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, 'names') else self.CLASS_NAMES

        # 规则引擎
        self.rule_engine = QualityRuleEngine()

        # 工具集
        self.tools = ToolRegistry(self)

        # 数据库
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'outputs', 'agent', 'detection_records.db'
            )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

        # 对话上下文（保留最近一次检测结果）
        self.context = {
            'last_image': None,
            'last_detections': None,
            'last_eval': None,
            'last_img_width': 640,
            'last_img_height': 640,
            'batch_results': None,       # 批量检测结果
            'batch_folder': None,        # 批量检测的文件夹
            'batch_summary': None,       # 批量汇总信息
            'conversation_history': [],
        }

    # ---- 核心：任务理解 + 规划 + 执行 ----

    def execute_task(self, user_input, image=None, folder=None):
        """
        执行用户任务（主入口）

        Args:
            user_input: 用户的自然语言指令
            image: 图像路径（可选，如果任务需要检测单张图片）
            folder: 文件夹路径（可选，如果任务需要批量检测）
        Returns:
            dict: {task_type, tool_calls, final_response, ...}
        """
        has_image = image is not None
        has_folder = folder is not None

        # 1. 任务理解
        intent = self._understand_intent(user_input, has_folder=has_folder)

        # 2. 任务规划
        plan = self._plan_tools(intent, has_image=has_image, has_folder=has_folder)

        # 3. 执行工具
        tool_results = []
        current_detections = self.context.get('last_detections')
        current_eval = self.context.get('last_eval')
        current_image = image or self.context.get('last_image')
        current_img_w = self.context.get('last_img_width', 640)
        current_img_h = self.context.get('last_img_height', 640)
        current_folder = folder or self.context.get('batch_folder')

        for step in plan:
            tool_name = step['tool']
            params = step.get('params', {})

            # 注入上下文
            params['detections'] = current_detections
            params['eval_result'] = current_eval
            params['image_path'] = current_image
            params['image_name'] = os.path.basename(current_image) if current_image else None
            params['img_width'] = current_img_w
            params['img_height'] = current_img_h
            params['folder_path'] = current_folder
            params['batch_results'] = self.context.get('batch_results')

            # 传入意图中的特定参数
            if 'target_class' in intent:
                params['target_class'] = intent['target_class']

            # 调用工具
            func = self.tools.get_tool(tool_name)
            if func:
                result = func(**params)
                tool_results.append(result)

                # 更新上下文
                if tool_name == 'detect':
                    current_detections = result['detections']
                    self.context['last_detections'] = current_detections
                    self.context['last_image'] = current_image
                    if 'img_width' in result:
                        current_img_w = result['img_width']
                        current_img_h = result['img_height']
                        self.context['last_img_width'] = current_img_w
                        self.context['last_img_height'] = current_img_h
                elif tool_name == 'classify_filter':
                    if result.get('status') == 'success' and 'filtered_detections' in result:
                        current_detections = result['filtered_detections']
                elif tool_name == 'quality_evaluate':
                    current_eval = result
                    self.context['last_eval'] = current_eval
                elif tool_name == 'batch_detect_folder':
                    if result.get('status') == 'success':
                        self.context['batch_results'] = result.get('results', [])
                        self.context['batch_folder'] = result.get('folder_path')
                        self.context['batch_summary'] = {
                            'total_images': result.get('total_images', 0),
                            'total_detections': result.get('total_detections', 0),
                            'pass_count': result.get('pass_count', 0),
                            'review_count': result.get('review_count', 0),
                            'fail_count': result.get('fail_count', 0),
                            'pass_rate': result.get('pass_rate', 0),
                            'by_class_total': result.get('by_class_total', {}),
                        }
            else:
                tool_results.append({'tool': tool_name, 'status': 'error', 'message': f'未知工具: {tool_name}'})

        # 4. 结果整合
        final_response = self._synthesize_response(user_input, intent, tool_results)

        # 5. 保存对话历史
        self.context['conversation_history'].append({
            'user_input': user_input,
            'intent': intent['type'],
            'response': final_response,
            'timestamp': datetime.now().isoformat(),
        })

        return {
            'intent': intent['type'],
            'confidence': intent['confidence'],
            'tool_calls': [r['tool'] for r in tool_results],
            'tool_results': tool_results,
            'final_response': final_response,
            'detections': current_detections,
            'eval_result': current_eval,
            'batch_results': self.context.get('batch_results'),
        }

    def _understand_intent(self, user_input, has_folder=False):
        """
        任务理解：解析用户意图
        返回: {type, confidence, target_class, ...}
        """
        text = user_input.strip()
        text_lower = text.lower()

        # 检测意图关键词
        detect_kw = ['检测', '识别', '看看', '找出', '查一下', '有没有缺陷', '缺陷检测',
                    '帮我看', '分析一下', '检查一下']
        count_kw = ['多少个', '几个', '数量', '统计', '计数', '多少']
        quality_kw = ['合格', '不合格', '质量', '判定', '评级', '评估', 'pass', 'fail',
                      '能不能过', '过得了吗', '合格吗', '行不行', '合格率']
        explain_kw = ['为什么', '原因', '解释', '哪里有问题', '说明一下', '怎么回事']
        report_kw = ['报告', '生成报告', '导出', '报表', '生成记录']
        save_kw = ['保存', '存档', '存起来', '记录一下']
        # 批量相关关键词
        batch_kw = ['批量', '文件夹', '全部', '所有', '批次', '一批', '整个',
                    '这些', '那些', '每张', '逐个', '分别']
        # 列表/分布相关
        list_kw = ['哪些', '列表', '分布', '分别是', '不合格的', '不合格列表',
                   '各有多少', '各类', '缺陷分布']

        # 识别缺陷类别
        target_class = None
        class_kw = {
            '裂纹': 'crazing', 'crazing': 'crazing',
            '夹杂': 'inclusion', 'inclusion': 'inclusion',
            '划痕': 'scratches', 'scratches': 'scratches',
            '斑块': 'patches', 'patches': 'patches',
            '麻点': 'pitted_surface', 'pitted_surface': 'pitted_surface',
            '氧化皮': 'rolled-in_scale', 'rolled-in_scale': 'rolled-in_scale',
        }
        for kw, cls in class_kw.items():
            if kw in text:
                target_class = cls
                break

        # 判断是否为批量场景
        is_batch = has_folder  # 有文件夹输入默认是批量
        for kw in batch_kw:
            if kw in text:
                is_batch = True
                break

        # 判断是否为列表/分布类意图
        is_list = False
        for kw in list_kw:
            if kw in text:
                is_list = True
                break

        # "不合格+哪些/列表" 组合优先判定为列表意图
        if ('不合格' in text or '合格的' in text) and ('哪些' in text or '列表' in text or '分别' in text):
            is_list = True

        # "缺陷+分布/各类" 组合直接判定为分布意图
        if ('缺陷' in text or '各类' in text) and ('分布' in text or '各类' in text or '分别' in text):
            is_list = True

        # 简单意图分类（基于关键词权重）
        scores = {
            'detect': 0,
            'count': 0,
            'quality': 0,
            'explain': 0,
            'report': 0,
            'save': 0,
        }

        for kw in detect_kw:
            if kw in text:
                scores['detect'] += 2
        for kw in count_kw:
            if kw in text:
                scores['count'] += 2
        for kw in quality_kw:
            if kw in text_lower:
                scores['quality'] += 2
        for kw in explain_kw:
            if kw in text:
                scores['explain'] += 2
        for kw in report_kw:
            if kw in text:
                scores['report'] += 2
        for kw in save_kw:
            if kw in text:
                scores['save'] += 2

        # 找出最高得分意图
        max_score = max(scores.values())
        if max_score == 0:
            # 默认：有文件夹就批量检测，有图片就单图检测
            if has_folder:
                if is_list:
                    if '分布' in text or '各类' in text:
                        intent_type = 'batch_distribution'
                    else:
                        intent_type = 'batch_list'
                else:
                    intent_type = 'batch_detect'
                confidence = 0.6
            else:
                intent_type = 'detect'
                confidence = 0.5
        else:
            intent_type = max(scores, key=scores.get)
            confidence = min(0.95, max_score / 4)

        # 批量模式下调整意图类型
        if is_batch:
            if is_list:
                # 列表/分布类优先
                if '分布' in text or '各类' in text:
                    intent_type = 'batch_distribution'
                else:
                    intent_type = 'batch_list'
            elif intent_type == 'detect':
                intent_type = 'batch_detect'
            elif intent_type == 'quality':
                intent_type = 'batch_quality'
            elif intent_type == 'count':
                intent_type = 'batch_count'

        return {
            'type': intent_type,
            'confidence': round(confidence, 2),
            'target_class': target_class,
            'scores': scores,
            'is_batch': is_batch,
            'is_list': is_list,
        }

    def _plan_tools(self, intent, has_image=False, has_folder=False):
        """
        任务规划：根据意图生成工具调用序列
        """
        plan = []
        intent_type = intent['type']
        is_batch = intent.get('is_batch', False)
        has_batch_results = self.context.get('batch_results') is not None

        # ---- 批量相关意图 ----
        if intent_type == 'batch_detect':
            # 批量检测文件夹
            if has_folder:
                plan.append({'tool': 'batch_detect_folder'})
            return plan

        if intent_type == 'batch_quality':
            # 批次质量统计：如果没有结果先检测，再看合格率
            if has_folder and not has_batch_results:
                plan.append({'tool': 'batch_detect_folder'})
            plan.append({'tool': 'batch_quality_stats'})
            return plan

        if intent_type == 'batch_count':
            # 批量统计缺陷数量
            if has_folder and not has_batch_results:
                plan.append({'tool': 'batch_detect_folder'})
            plan.append({'tool': 'batch_quality_stats'})
            return plan

        if intent_type == 'batch_distribution':
            # 各类缺陷分布
            if has_folder and not has_batch_results:
                plan.append({'tool': 'batch_detect_folder'})
            plan.append({'tool': 'batch_defect_distribution'})
            return plan

        if intent_type == 'batch_list':
            # 列出不合格的图片
            if has_folder and not has_batch_results:
                plan.append({'tool': 'batch_detect_folder'})
            plan.append({'tool': 'batch_list_failed'})
            return plan

        # ---- 单图相关意图（保持原有逻辑） ----
        if intent_type == 'detect':
            # 检测 → 统计 → 质量判定 → 保存
            if has_image:
                plan.append({'tool': 'detect'})
            plan.append({'tool': 'count_defects'})
            plan.append({'tool': 'quality_evaluate'})
            if has_image:
                plan.append({'tool': 'save_record'})

        elif intent_type == 'count':
            # 统计数量（可能有类别筛选）
            if has_image:
                plan.append({'tool': 'detect'})
            if intent.get('target_class'):
                plan.append({'tool': 'classify_filter'})
            plan.append({'tool': 'count_defects'})

        elif intent_type == 'quality':
            # 质量判定 + 解释
            if has_image:
                plan.append({'tool': 'detect'})
            plan.append({'tool': 'quality_evaluate'})
            plan.append({'tool': 'explain_result'})

        elif intent_type == 'explain':
            # 解释原因
            plan.append({'tool': 'explain_result'})

        elif intent_type == 'report':
            # 生成报告
            plan.append({'tool': 'generate_report'})

        elif intent_type == 'save':
            # 保存记录
            if has_image:
                plan.append({'tool': 'detect'})
                plan.append({'tool': 'quality_evaluate'})
            plan.append({'tool': 'save_record'})

        return plan

    def _synthesize_response(self, user_input, intent, tool_results):
        """
        结果整合：把多个工具的输出整合成自然语言回答
        """
        lines = []

        # 开头问候
        greetings = {
            'detect': '好的，我来帮您检测一下。',
            'count': '好的，我来统计一下缺陷数量。',
            'quality': '好的，我来评估一下质量情况。',
            'explain': '我来解释一下判定原因。',
            'report': '好的，正在生成报告。',
            'save': '好的，我来保存这条记录。',
            'batch_detect': '好的，我来批量检测这个文件夹。',
            'batch_quality': '好的，我来统计一下批次合格率。',
            'batch_count': '好的，我来统计一下缺陷总数。',
            'batch_distribution': '好的，我来统计一下缺陷分布。',
            'batch_list': '好的，我来列出不合格的图片。',
        }
        lines.append(greetings.get(intent['type'], '好的。'))
        lines.append('')

        # 依次呈现各工具结果
        for result in tool_results:
            tool = result['tool']
            status = result.get('status', 'unknown')

            if status == 'error':
                lines.append(f"⚠️ {result.get('message', '执行出错')}")
                continue

            if tool == 'detect':
                count = result.get('total_count', 0)
                ms = result.get('inference_time_ms', 0)
                if count == 0:
                    lines.append(f"🔍 检测完成：未检测到缺陷（耗时 {ms:.1f}ms）")
                else:
                    lines.append(f"🔍 检测完成：发现 {count} 个缺陷（推理耗时 {ms:.1f}ms）")

            elif tool == 'count_defects':
                by_class = result.get('by_class', {})
                total = result.get('total_count', 0)
                if by_class:
                    parts = [f"{c}：{n}个" for c, n in by_class.items()]
                    lines.append(f"📊 缺陷统计：共 {total} 个，其中" + "、".join(parts))
                else:
                    lines.append(f"📊 缺陷统计：共 {total} 个缺陷")

            elif tool == 'classify_filter':
                lines.append(f"🔎 {result.get('summary', '')}")

            elif tool == 'quality_evaluate':
                quality = result.get('quality', 'UNKNOWN')
                score = result.get('score', 0)
                badge = {'PASS': '✅', 'REVIEW': '⚠️', 'FAIL': '❌'}.get(quality, '❓')
                lines.append(f"{badge} 质量判定：{quality}（综合评分 {score} 分）")

            elif tool == 'explain_result':
                explanation = result.get('explanation', '')
                # 跳过第一行（已经显示过判定了）
                exp_lines = explanation.split('\n')[1:]
                for l in exp_lines:
                    lines.append(l)

            elif tool == 'save_record':
                lines.append(f"💾 {result.get('message', '记录已保存')}")

            elif tool == 'generate_report':
                stats = result.get('statistics', {})
                lines.append(f"📋 报告生成：累计 {stats.get('total_records', 0)} 条检测记录，"
                           f"合格率 {stats.get('pass_rate', 0):.1f}%")

            elif tool == 'batch_detect_folder':
                # 批量检测结果
                summary = result.get('summary', '')
                lines.append(summary)

            elif tool == 'batch_quality_stats':
                # 批量质量统计
                summary = result.get('summary', '')
                lines.append(summary)

            elif tool == 'batch_defect_distribution':
                # 缺陷分布
                summary = result.get('summary', '')
                lines.append(summary)

            elif tool == 'batch_list_failed':
                # 不合格列表
                summary = result.get('summary', '')
                lines.append(summary)

        return "\n".join(lines)

    # ---- 底层检测功能 ----

    def _run_detection(self, image_path, conf_threshold=0.25):
        """运行YOLO检测，返回标准化结果（含原始图像尺寸）"""
        import time
        from PIL import Image as PILImage

        # 获取原始图像尺寸
        try:
            with PILImage.open(image_path) as img:
                orig_w, orig_h = img.size
        except Exception:
            orig_w, orig_h = 640, 640

        t0 = time.time()

        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False,
        )

        inference_time = (time.time() - t0) * 1000

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_idx = int(box.cls[0].item())
                    cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([cls_name, round(conf, 4),
                                      round(x1, 1), round(y1, 1),
                                      round(x2, 1), round(y2, 1)])

        return {
            'detections': detections,
            'inference_time_ms': round(inference_time, 1),
            'img_width': orig_w,
            'img_height': orig_h,
        }

    # ---- 数据库功能 ----

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                timestamp TEXT,
                total_defects INTEGER,
                quality TEXT,
                quality_score INTEGER,
                detections_json TEXT,
                eval_json TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _save_record(self, image_name, detections, eval_result):
        """保存检测记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detection_records "
            "(image_name, timestamp, total_defects, quality, quality_score, "
            "detections_json, eval_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                image_name,
                datetime.now().isoformat(),
                eval_result['total_defects'],
                eval_result['quality'],
                eval_result['score'],
                json.dumps(detections, ensure_ascii=False),
                json.dumps(eval_result, ensure_ascii=False),
            )
        )
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id

    def query_records(self, quality=None, limit=50):
        """查询历史记录"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if quality:
            cursor.execute(
                "SELECT * FROM detection_records WHERE quality=? "
                "ORDER BY id DESC LIMIT ?",
                (quality, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM detection_records ORDER BY id DESC LIMIT ?",
                (limit,)
            )

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_statistics(self):
        """获取统计数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM detection_records")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT quality, COUNT(*) FROM detection_records GROUP BY quality"
        )
        quality_counts = dict(cursor.fetchall())

        pass_count = quality_counts.get('PASS', 0)
        fail_count = quality_counts.get('FAIL', 0)
        review_count = quality_counts.get('REVIEW', 0)

        pass_rate = (pass_count / total * 100) if total > 0 else 0

        cursor.execute(
            "SELECT AVG(quality_score) FROM detection_records"
        )
        avg_score = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_records': total,
            'pass_count': pass_count,
            'fail_count': fail_count,
            'review_count': review_count,
            'pass_rate': round(pass_rate, 1),
            'avg_score': round(avg_score, 1),
            'quality_distribution': quality_counts,
        }


# ============================================================
# 兼容旧接口（DetectionAgent 别名，保持向后兼容）
# ============================================================
DetectionAgent = AIAgent
