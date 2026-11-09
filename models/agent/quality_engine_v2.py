# -*- coding: utf-8 -*-
"""
质量规则引擎 v2 —— 安全边界重构

改进点：
1. 规则版本化（rule_id / version / source）
2. 演示规则明确标记，不冒充工业标准
3. 图像质量门控（模糊/失败 → REVIEW，不能 PASS）
4. 面积使用"检测框覆盖率"代理指标，不冒称真实缺陷面积
5. 重叠框并集计算，避免重复累计
6. 高置信度≠可接受，关键缺陷不管置信度都要考虑
7. 有明确 FAIL 证据时，不因其他不确定性降为 REVIEW
8. 无缺陷只能表述为"未发现"，只有满足放行前提才能 PASS
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .task_context import TaskContext, DetectionItem, SCOPE_SUBSET, SCOPE_BATCH_SUBSET


# 缺陷类别严重等级（1-5，5最严重）
CLASS_SEVERITY = {
    'crazing': 5,
    'inclusion': 4,
    'scratches': 3,
    'patches': 3,
    'pitted_surface': 2,
    'rolled-in_scale': 2,
}

# 类别中文名
CLASS_NAMES_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化铁皮',
    'scratches': '划痕',
}


@dataclass
class QualityRuleSet:
    """规则集定义"""
    rule_id: str = "demo_default"
    version: str = "1.0.0"
    source: str = "演示规则"  # 演示规则 / 企业标准 / 自定义
    is_demo: bool = True
    description: str = "实验演示用规则，不构成工业质量判定依据"

    # 缺陷数量阈值
    max_minor_defects: int = 5          # 轻度缺陷最大允许数量
    max_critical_defects: int = 1       # 严重缺陷最大允许数量
    critical_severity: int = 4          # 严重等级 >= 此值视为致命缺陷

    # 面积阈值（检测框覆盖率，百分比）
    max_total_box_area_pct: float = 15.0   # 总检测框面积上限
    max_single_box_area_pct: float = 5.0   # 单检测框面积上限

    # 置信度相关
    review_low_confidence: float = 0.5    # 低于此置信度 → 低置信
    review_low_conf_count: int = 2        # 低置信检测达到多少个 → REVIEW

    # 评分
    pass_score_threshold: int = 80        # 评分 >= 此值 → PASS（需同时满足其他条件）
    fail_score_threshold: int = 60        # 评分 < 此值 → FAIL
    review_borderline_min: int = 60       # 评分在此区间 → REVIEW（边界）
    review_borderline_max: int = 80

    # 图像质量门控
    min_image_quality: float = 0.5        # 图像质量评分低于此值 → REVIEW
    auto_fail_on_no_detection: bool = False  # 无检测时是否自动PASS（演示规则默认True，但有免责声明）

    # 备注
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'version': self.version,
            'source': self.source,
            'is_demo': self.is_demo,
            'description': self.description,
            'max_minor_defects': self.max_minor_defects,
            'max_critical_defects': self.max_critical_defects,
            'critical_severity': self.critical_severity,
            'max_total_box_area_pct': self.max_total_box_area_pct,
            'max_single_box_area_pct': self.max_single_box_area_pct,
            'review_low_confidence': self.review_low_confidence,
            'review_low_conf_count': self.review_low_conf_count,
            'pass_score_threshold': self.pass_score_threshold,
            'fail_score_threshold': self.fail_score_threshold,
            'min_image_quality': self.min_image_quality,
            'notes': self.notes,
        }

    @classmethod
    def default_demo(cls) -> 'QualityRuleSet':
        """默认演示规则集"""
        return cls(
            rule_id="demo_default",
            version="1.0.0",
            source="演示规则",
            is_demo=True,
            description="实验演示用规则集，阈值为经验设定，不构成工业质量判定标准",
            notes=[
                "本规则为实验演示用途，不代表任何企业质量标准",
                "面积指标为检测框覆盖率，非真实缺陷面积",
                "无缺陷判定为PASS仅表示模型未检出已知缺陷类别",
            ]
        )


class QualityEngineV2:
    """
    质量判定引擎 v2 —— 安全边界增强版

    核心原则：
    1. 证据不足 → REVIEW，不能 PASS
    2. 有明确 FAIL 证据 → FAIL，不能降为 REVIEW
    3. 子集结论不能代表整体
    4. 面积指标必须标注为"检测框覆盖率"
    5. 所有判定都标明规则版本和适用范围
    """

    def __init__(self, rule_set: Optional[QualityRuleSet] = None):
        self.rule_set = rule_set or QualityRuleSet.default_demo()

    # ==================== 公开入口 ====================

    def evaluate(self, ctx: TaskContext, scope: str = 'current') -> Dict[str, Any]:
        """
        执行质量判定

        Args:
            ctx: 任务上下文
            scope: 判定范围 ('current' / 'full')

        Returns:
            完整的判定结果字典
        """
        # 确定实际判定范围
        if scope == 'full':
            dets = ctx.get_raw_detections() if not ctx.is_batch else self._get_all_batch_dets(ctx)
            actual_scope = SCOPE_BATCH if ctx.is_batch else 'full_image'
            scope_label = '批次全量' if ctx.is_batch else '单图全量'
        else:
            dets = ctx.get_current_detections()
            actual_scope = ctx.current_scope
            scope_label = ctx.scope_label

        img_w = ctx.img_width
        img_h = ctx.img_height

        # ---- 安全检查 ----
        safety = self._safety_check(ctx, actual_scope)
        if safety['forced_result']:
            # 安全规则强制覆盖判定结果
            return self._build_result(
                quality=safety['forced_result'],
                score=0,
                dets=dets,
                img_w=img_w,
                img_h=img_h,
                reasons_fail=safety.get('reasons_fail', []),
                reasons_review=safety.get('reasons_review', []),
                scope=actual_scope,
                scope_label=scope_label,
                method_notes=safety.get('notes', []),
                forced=True,
                forced_reason=safety['reason'],
            )

        # ---- 正常判定 ----
        return self._do_evaluate(dets, img_w, img_h, actual_scope, scope_label)

    # ==================== 安全检查 ====================

    def _safety_check(self, ctx: TaskContext, scope: str) -> Dict:
        """
        安全边界检查 —— 某些情况必须强制输出特定结果

        返回: {
            forced_result: None/'PASS'/'REVIEW'/'FAIL',
            reason: str,
            reasons_fail: [],
            reasons_review: [],
            notes: []
        }
        """
        reasons_review = []
        reasons_fail = []
        notes = []
        forced = None

        # 1. 子集范围：不能输出整体质量结论
        if scope in (SCOPE_SUBSET, SCOPE_BATCH_SUBSET):
            notes.append("⚠️ 子集范围判定：此结论仅针对筛选后的缺陷子集，不代表整图或整批质量")
            # 子集仍然可以判定，但必须标注清楚
            # 不强制覆盖，只是加备注

        # 2. 图像质量问题 → 强制 REVIEW
        if ctx.image and ctx.image.quality_score is not None:
            if ctx.image.quality_score < self.rule_set.min_image_quality:
                reasons_review.append(
                    f"图像质量评分过低 ({ctx.image.quality_score:.2f} < {self.rule_set.min_image_quality})"
                )
                if ctx.image.quality_issues:
                    reasons_review.append(
                        "质量问题: " + "、".join(ctx.image.quality_issues)
                    )
                notes.append("图像质量不满足检测前提，检测结果可信度降低")
                forced = 'REVIEW'
                forced_reason = "图像质量门控：低质量图像不得直接PASS"

        # 3. 检测失败 → 强制 REVIEW
        if ctx.detection_errors:
            reasons_review.append(f"检测过程出错: {'; '.join(ctx.detection_errors)}")
            notes.append("检测失败或部分失败，不能得出合格结论")
            forced = 'REVIEW'
            forced_reason = "检测异常：检测失败的图像不得判定为PASS"

        # 4. 批次检测未完成 → 不能输出整批结论
        if scope in ('batch', 'batch_full') and ctx.batch_errors:
            err_count = len(ctx.batch_errors)
            reasons_review.append(f"批次中 {err_count} 张图像检测失败")
            forced = 'REVIEW'
            forced_reason = "批次不完整：存在检测失败的图像，不得输出整批PASS"

        # 5. 没有检测数据 → REVIEW（不是 PASS！）
        # （只有在明确满足放行前提时才能 PASS）
        if forced is None and not ctx.has_raw_detections:
            reasons_review.append("没有可用的检测数据")
            forced = 'REVIEW'
            forced_reason = "证据不足：没有检测数据，不能判定合格"

        if forced:
            return {
                'forced_result': forced,
                'reason': forced_reason,
                'reasons_fail': reasons_fail,
                'reasons_review': reasons_review,
                'notes': notes,
            }

        return {'forced_result': None}

    # ==================== 核心判定逻辑 ====================

    def _do_evaluate(self, dets: List[DetectionItem], img_w: int, img_h: int,
                     scope: str, scope_label: str) -> Dict[str, Any]:
        """执行正常的三级判定"""
        rs = self.rule_set
        img_area = max(1, img_w) * max(1, img_h)

        # ---- 统计 ----
        total_count = len(dets)
        by_class = {}
        critical_count = 0
        low_conf_count = 0
        max_severity = 0

        for d in dets:
            severity = CLASS_SEVERITY.get(d.cls_name, 2)
            max_severity = max(max_severity, severity)

            if severity >= rs.critical_severity:
                critical_count += 1
            if d.confidence < rs.review_low_confidence:
                low_conf_count += 1

            if d.cls_name not in by_class:
                by_class[d.cls_name] = {'count': 0, 'total_box_area_pct': 0.0,
                                         'max_conf': 0.0, 'min_conf': 1.0}
            by_class[d.cls_name]['count'] += 1
            box_area_pct = d.area / img_area * 100
            by_class[d.cls_name]['total_box_area_pct'] += box_area_pct
            by_class[d.cls_name]['max_conf'] = max(by_class[d.cls_name]['max_conf'], d.confidence)
            by_class[d.cls_name]['min_conf'] = min(by_class[d.cls_name]['min_conf'], d.confidence)

        # 面积计算（并集去重近似）
        total_box_area_pct, max_single_area_pct = self._calculate_box_areas(dets, img_w, img_h)

        # ---- 评分 ----
        score = 100
        reasons_fail = []
        reasons_review = []

        # 扣分：严重缺陷
        if critical_count > rs.max_critical_defects:
            penalty = (critical_count - rs.max_critical_defects) * 25
            score -= penalty
            reasons_fail.append(
                f'严重缺陷过多 ({critical_count} > {rs.max_critical_defects}个，'
                f'涉及裂纹/夹杂等高危险性缺陷)'
            )

        # 扣分：总缺陷数量
        if total_count > rs.max_minor_defects:
            penalty = (total_count - rs.max_minor_defects) * 5
            score -= penalty
            reasons_fail.append(
                f'缺陷数量偏多 ({total_count} > {rs.max_minor_defects}个)'
            )

        # 扣分：总检测框面积
        if total_box_area_pct > rs.max_total_box_area_pct:
            penalty = 15
            score -= penalty
            reasons_fail.append(
                f'检测框总面积过大 ({total_box_area_pct:.1f}% > {rs.max_total_box_area_pct}%)'
            )

        # 扣分：单检测框超大
        if max_single_area_pct > rs.max_single_box_area_pct:
            penalty = 10
            score -= penalty
            reasons_fail.append(
                f'存在超大检测框 (单框{max_single_area_pct:.1f}% > {rs.max_single_box_area_pct}%)'
            )

        # 复核因素：低置信度
        if low_conf_count >= rs.review_low_conf_count:
            reasons_review.append(
                f'{low_conf_count} 个检测置信度低于 {rs.review_low_confidence}，模型识别存疑'
            )

        # 复核因素：评分接近合格线
        if rs.review_borderline_min <= score < rs.review_borderline_max:
            reasons_review.append(f'评分接近合格线 ({score:.0f}分)，建议人工复核确认')

        score = max(0, min(100, int(score)))

        # ---- 三级判定 ----
        # 有明确 FAIL 证据 → FAIL（不因其他不确定性降为 REVIEW）
        if score < rs.fail_score_threshold or reasons_fail:
            quality = 'FAIL'
        # 有复核因素 → REVIEW
        elif reasons_review:
            quality = 'REVIEW'
        # 评分达标且无复核因素 → PASS
        elif score >= rs.pass_score_threshold:
            quality = 'PASS'
        else:
            quality = 'REVIEW'

        # ---- 方法备注 ----
        method_notes = [
            "面积指标为「检测框覆盖率」，是真实缺陷面积的代理指标，非物理测量值",
            f"规则版本: {rs.rule_id} v{rs.version} ({rs.source})",
            f"判定范围: {scope_label}",
            f"参与判定的检测数量: {total_count} 个",
        ]
        if rs.is_demo:
            method_notes.insert(0, "⚠️ 演示规则：本判定基于实验演示规则，不构成工业质量判定依据")

        # 无缺陷时的特殊说明
        if total_count == 0:
            method_notes.append("未检测到缺陷仅表示模型未识别出已知缺陷类别，不等于产品合格")

        return self._build_result(
            quality=quality,
            score=score,
            dets=dets,
            img_w=img_w,
            img_h=img_h,
            by_class=by_class,
            total_count=total_count,
            critical_count=critical_count,
            low_conf_count=low_conf_count,
            max_severity=max_severity,
            total_box_area_pct=total_box_area_pct,
            max_single_area_pct=max_single_area_pct,
            reasons_fail=reasons_fail,
            reasons_review=reasons_review,
            scope=scope,
            scope_label=scope_label,
            method_notes=method_notes,
            forced=False,
        )

    # ==================== 辅助方法 ====================

    def _calculate_box_areas(self, dets: List[DetectionItem], img_w: int, img_h: int
                              ) -> Tuple[float, float]:
        """
        计算检测框面积占比（并集去重近似）

        简化并集计算：按行扫描，每行统计被覆盖的像素宽度
        对于少量检测框来说足够准确，且避免了复杂的多边形并集运算
        """
        if not dets:
            return 0.0, 0.0

        img_area = max(1, img_w) * max(1, img_h)
        max_single = max(d.area for d in dets) if dets else 0.0

        # 简单求和（如果框重叠多会高估，但作为代理指标可接受）
        # 并集近似：使用扫描线法
        # 收集所有水平边
        events = []
        for d in dets:
            y1 = max(0, int(d.y1))
            y2 = min(img_h, int(d.y2))
            if y1 < y2:
                events.append((y1, 1, d.x1, d.x2))  # 进入
                events.append((y2, -1, d.x1, d.x2))  # 离开

        events.sort(key=lambda x: (x[0], x[1]))

        union_area = 0.0
        active_intervals = []
        prev_y = 0

        for y, typ, x1, x2 in events:
            if y > prev_y and active_intervals:
                # 计算 [prev_y, y] 区间内的覆盖宽度
                intervals = sorted(active_intervals)
                merged = []
                for s, e in intervals:
                    if merged and s <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                    else:
                        merged.append((s, e))
                coverage_width = sum(e - s for s, e in merged)
                union_area += coverage_width * (y - prev_y)
            if typ == 1:
                active_intervals.append((x1, x2))
            else:
                # 离开（移除一个区间）
                try:
                    active_intervals.remove((x1, x2))
                except ValueError:
                    pass
            prev_y = y

        total_pct = union_area / img_area * 100
        max_single_pct = max_single / img_area * 100

        return round(total_pct, 2), round(max_single_pct, 2)

    def _get_all_batch_dets(self, ctx: TaskContext) -> List[DetectionItem]:
        """获取批次所有检测结果"""
        all_dets = []
        for dets in ctx._batch_detections.values():
            all_dets.extend(dets)
        return all_dets

    def _build_result(self, quality, score, dets, img_w, img_h,
                      by_class=None, total_count=None, critical_count=0,
                      low_conf_count=0, max_severity=0,
                      total_box_area_pct=0.0, max_single_area_pct=0.0,
                      reasons_fail=None, reasons_review=None,
                      scope='', scope_label='',
                      method_notes=None, forced=False, forced_reason=''):
        """构建标准化结果字典"""
        total = total_count if total_count is not None else len(dets)

        explanation = self._generate_explanation(
            quality, score, total, critical_count,
            total_box_area_pct, by_class or {},
            reasons_fail or [], reasons_review or [],
            scope_label, forced, forced_reason
        )

        return {
            'quality': quality,
            'score': score,
            'total_defects': total,
            'critical_count': critical_count,
            'low_conf_count': low_conf_count,
            'total_box_area_pct': total_box_area_pct,
            'max_single_box_area_pct': max_single_area_pct,
            'max_severity': max_severity,
            'by_class': by_class or {},
            'reasons_fail': reasons_fail or [],
            'reasons_review': reasons_review or [],
            'explanation': explanation,
            'scope': scope,
            'scope_label': scope_label,
            'method_notes': method_notes or [],
            'is_forced': forced,
            'forced_reason': forced_reason,
            'rule_id': self.rule_set.rule_id,
            'rule_version': self.rule_set.version,
            'rule_source': self.rule_set.source,
            'is_demo_rule': self.rule_set.is_demo,
            'area_method': '检测框覆盖率（并集去重），代理指标',
        }

    def _generate_explanation(self, quality, score, total_count, critical_count,
                               total_area, by_class, reasons_fail, reasons_review,
                               scope_label, forced, forced_reason):
        """生成自然语言解释"""
        lines = []

        scope_info = f"[{scope_label}] "

        if quality == 'PASS':
            lines.append(f"{scope_info}✅ 质量判定：合格（PASS），综合评分 {score} 分。")
            if total_count == 0:
                lines.append("本次检测未发现系统识别范围内的缺陷。")
            else:
                lines.append(f"共检测到 {total_count} 个缺陷，均在规则允许范围内。")
                cls_parts = []
                for c, d in by_class.items():
                    cn = CLASS_NAMES_CN.get(c, c)
                    cls_parts.append(f"{cn}({d['count']}个)")
                lines.append("缺陷类型：" + "、".join(cls_parts) + "。")

        elif quality == 'REVIEW':
            lines.append(f"{scope_info}⚠️ 质量判定：待复核（REVIEW），综合评分 {score} 分。")
            if forced:
                lines.append(f"强制复核原因：{forced_reason}")
            lines.append(f"共检测到 {total_count} 个缺陷。")
            if reasons_review:
                lines.append("复核原因：")
                for r in reasons_review:
                    lines.append(f"  • {r}")
            if reasons_fail:
                lines.append("不合格因素：")
                for r in reasons_fail:
                    lines.append(f"  • {r}")
            lines.append("建议质检人员进行人工确认。")

        else:  # FAIL
            lines.append(f"{scope_info}❌ 质量判定：不合格（FAIL），综合评分 {score} 分。")
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


# ==================== 规则存储 ====================

class RuleStore:
    """规则集存储（文件系统）"""

    def __init__(self, rules_dir: str):
        self.rules_dir = rules_dir
        os.makedirs(rules_dir, exist_ok=True)

    def save(self, rule_set: QualityRuleSet):
        """保存规则集"""
        path = os.path.join(self.rules_dir, f"{rule_set.rule_id}_v{rule_set.version}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(rule_set.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, rule_id: str, version: str = None) -> Optional[QualityRuleSet]:
        """加载规则集"""
        if version:
            path = os.path.join(self.rules_dir, f"{rule_id}_v{version}.json")
        else:
            # 找最新版本
            import glob
            files = glob.glob(os.path.join(self.rules_dir, f"{rule_id}_v*.json"))
            if not files:
                return None
            files.sort(reverse=True)
            path = files[0]

        if not os.path.exists(path):
            return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return QualityRuleSet(**data)

    def list_rules(self) -> List[Dict]:
        """列出所有规则集"""
        import glob
        files = glob.glob(os.path.join(self.rules_dir, "*.json"))
        rules = []
        for f in sorted(files):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                rules.append({
                    'rule_id': data.get('rule_id'),
                    'version': data.get('version'),
                    'source': data.get('source'),
                    'is_demo': data.get('is_demo', True),
                    'description': data.get('description', ''),
                    'file': os.path.basename(f),
                })
            except Exception:
                pass
        return rules
