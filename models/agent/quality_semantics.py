# -*- coding: utf-8 -*-
"""
统一判定语义模块

明确所有质量判定状态的含义、优先级和使用限制。

核心原则：
1. 区分零目标、失败、未检测和部分完成
2. 演示 PASS 不等于工业放行
3. FAIL 优先级高于 REVIEW
4. 产品定位为辅助质检与复核 Agent，不宣称工业自动放行能力
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


# ============================================================
# 质量等级定义
# ============================================================

QUALITY_LEVELS = {
    'PASS': {
        'level': 4,
        'cn_name': '通过',
        'color': 'green',
        'description': '检测结果满足当前质量规则的所有通过条件',
        'limitations': [
            '仅基于当前检测模型和演示规则',
            '未检测到缺陷不等于产品合格',
            '演示环境下的 PASS 不构成工业放行依据',
            '仅覆盖已检测的范围（全量或子集）',
        ],
    },
    'REVIEW': {
        'level': 3,
        'cn_name': '需复核',
        'color': 'yellow',
        'description': '系统无法给出确定结论，需要人工复核后才能最终判定',
        'limitations': [
            'REVIEW 不是 PASS，也不是 FAIL',
            '常见原因：图像质量不足、置信度偏低、边界情况',
            '必须经过人工复核才能进入下一步',
            'REVIEW 的优先级低于 FAIL（有 FAIL 时先处理 FAIL）',
        ],
    },
    'FAIL': {
        'level': 2,
        'cn_name': '不合格',
        'color': 'red',
        'description': '检测结果超出了当前质量规则的通过阈值',
        'limitations': [
            'FAIL 表示系统判定不合格',
            'FAIL 的优先级高于 REVIEW',
            '建议进入人工复核确认',
            '最终判定以人工复核结果为准',
        ],
    },
    'INCOMPLETE': {
        'level': 1,
        'cn_name': '未完成',
        'color': 'gray',
        'description': '检测或分析未全部完成，无法给出最终结论',
        'limitations': [
            '部分完成的批次不得输出 PASS',
            '需要处理完所有失败/跳过项后重新评估',
            '常见原因：图像读取失败、检测超时、工具调用失败',
        ],
    },
}


# ============================================================
# 特殊状态区分
# ============================================================

@dataclass
class DetectionStatus:
    """检测状态（区分零目标、失败、未检测）"""
    status: str  # 'detected' / 'zero_targets' / 'failed' / 'not_tested'
    message: str = ''

    @property
    def is_detected(self) -> bool:
        """有检测结果（可能是零目标，也可能是有缺陷）"""
        return self.status in ('detected', 'zero_targets')

    @property
    def has_defects(self) -> bool:
        """检测到缺陷"""
        return self.status == 'detected'

    @property
    def is_zero_targets(self) -> bool:
        """零目标：检测成功但未发现任何缺陷"""
        return self.status == 'zero_targets'

    @property
    def is_failed(self) -> bool:
        """检测失败"""
        return self.status == 'failed'

    @property
    def is_not_tested(self) -> bool:
        """未检测（还没做检测）"""
        return self.status == 'not_tested'


def classify_detection_status(detection_count: int, error_msg: str = '',
                               tested: bool = True) -> DetectionStatus:
    """
    分类检测状态

    区分：
    - 未检测 (not_tested): 尚未执行检测
    - 检测失败 (failed): 执行了检测但出错
    - 零目标 (zero_targets): 检测成功，0 个缺陷
    - 有检测结果 (detected): 检测成功，有缺陷
    """
    if not tested:
        return DetectionStatus('not_tested', '尚未执行检测')

    if error_msg:
        return DetectionStatus('failed', f'检测失败: {error_msg}')

    if detection_count == 0:
        return DetectionStatus(
            'zero_targets',
            '检测成功，未发现任何已知类别的缺陷。'
            '注意：零目标不等于产品合格，仅表示模型未识别出缺陷。'
        )

    return DetectionStatus('detected', f'检测到 {detection_count} 个缺陷')


# ============================================================
# 优先级规则
# ============================================================

def priority_compare(quality_a: str, quality_b: str) -> int:
    """
    比较两个质量等级的优先级

    返回:
      1: a 优先级更高（更严重）
      0: 同级
     -1: b 优先级更高

    优先级从高到低：FAIL > REVIEW > PASS > INCOMPLETE
    （注意：INCOMPLETE 是状态不是质量等级，但用于批次汇总时最低）
    """
    order = {'FAIL': 3, 'REVIEW': 2, 'PASS': 1, 'INCOMPLETE': 0}
    a_val = order.get(quality_a, -1)
    b_val = order.get(quality_b, -1)
    if a_val > b_val:
        return 1
    elif a_val < b_val:
        return -1
    return 0


def batch_quality_from_individuals(qualities: List[str],
                                    total_planned: int,
                                    total_completed: int) -> tuple:
    """
    从个体质量等级计算批次质量等级

    规则：
    1. 有未完成 → INCOMPLETE（部分完成不能 PASS）
    2. 有任何 FAIL → FAIL（FAIL 优先级最高）
    3. 有任何 REVIEW 且无 FAIL → REVIEW
    4. 全部 PASS 且全部完成 → PASS

    返回: (quality, reason)
    """
    if total_completed < total_planned:
        return 'INCOMPLETE', (
            f'批次未全部完成：成功 {total_completed}/{total_planned}，'
            f'未完成 {total_planned - total_completed}。'
            f'未完成的批次不能输出 PASS。'
        )

    fail_count = sum(1 for q in qualities if q == 'FAIL')
    review_count = sum(1 for q in qualities if q == 'REVIEW')
    pass_count = sum(1 for q in qualities if q == 'PASS')

    if fail_count > 0:
        return 'FAIL', f'批次中有 {fail_count} 张图像判定为 FAIL，整批评定为 FAIL。'
    if review_count > 0:
        return 'REVIEW', f'批次中有 {review_count} 张图像需人工复核，整批评定为 REVIEW。'
    if pass_count == len(qualities) and len(qualities) == total_planned:
        return 'PASS', f'全部 {total_planned} 张图像均判定为 PASS，批次判定为 PASS。'

    return 'INCOMPLETE', '无法确定批次质量等级。'


# ============================================================
# 产品定位声明
# ============================================================

PRODUCT_POSITIONING = {
    'name': '表面缺陷质检复核 Agent',
    'full_name': 'AI 辅助表面缺陷质量检测与复核系统',
    'positioning': '辅助质检与复核 Agent',
    'capabilities': [
        '自动检测表面缺陷（基于 YOLO 检测模型）',
        '根据质量规则给出初步质量判定',
        '支持全量/子集范围分析',
        '生成可追溯的检测报告',
        '创建人工复核任务',
        '支持自然语言交互（LLM 调度工具）',
    ],
    'not_capabilities': [
        '不具备工业自动放行能力',
        '不替代专业质检人员的最终判定',
        '检测结果仅供参考，不作为质量仲裁依据',
        '演示规则不构成正式质量标准',
        '未经过工业场景的长期验证和校准',
    ],
    'disclaimer': (
        '本系统为 AI 辅助质检工具，所有检测结果和质量判定均为初步结论，'
        '最终质量判定应以经过培训的专业质检人员的人工复核结果为准。'
        '演示版本使用演示规则和基线模型，不代表工业级精度。'
    ),
}


def get_quality_disclaimer(quality: str) -> str:
    """获取对应质量等级的免责声明"""
    if quality == 'PASS':
        return (
            '⚠️ 演示环境 PASS 说明：本 PASS 基于演示规则和检测模型得出，'
            '仅供系统功能验证使用，不构成工业放行依据。'
            '零缺陷检测结果不等于产品合格。'
        )
    elif quality == 'FAIL':
        return (
            '⚠️ FAIL 为系统初步判定，建议进入人工复核确认。'
            '最终判定以人工复核结果为准。'
        )
    elif quality == 'REVIEW':
        return (
            '⚠️ REVIEW 表示系统无法给出确定结论，必须经过人工复核才能最终判定。'
            'REVIEW 不是 PASS，也不是 FAIL。'
        )
    elif quality == 'INCOMPLETE':
        return (
            '⚠️ 检测未全部完成，当前结论不完整。'
            '请处理完所有失败项后重新评估。'
        )
    return ''


def get_semantics_summary_text() -> str:
    """获取判定语义摘要（用于报告/日志）"""
    lines = [
        "=" * 60,
        "判定语义说明",
        "=" * 60,
        "",
        "质量等级：",
        "  PASS     - 通过。满足当前规则的通过条件",
        "  REVIEW   - 需复核。系统无法确定，需人工确认",
        "  FAIL     - 不合格。超出通过阈值",
        "  INCOMPLETE - 未完成。检测未全部完成",
        "",
        "检测状态区分：",
        "  有缺陷 (detected)   - 检测成功，发现缺陷",
        "  零目标 (zero_targets) - 检测成功，未发现缺陷（≠产品合格）",
        "  检测失败 (failed)    - 执行了检测但出错",
        "  未检测 (not_tested)  - 尚未执行检测",
        "",
        "优先级规则：",
        "  FAIL > REVIEW > PASS > INCOMPLETE",
        "  批次中只要有 FAIL，整批评 FAIL",
        "  部分完成的批次不能输出 PASS",
        "",
        "产品定位：",
        f"  {PRODUCT_POSITIONING['positioning']}",
        "  辅助质检，不宣称工业自动放行能力",
        "  最终判定以人工复核为准",
    ]
    return "\n".join(lines)


if __name__ == '__main__':
    print(get_semantics_summary_text())
