# -*- coding: utf-8 -*-
"""
质检 Agent 任务案例库（30条可复现测试案例）

每个案例包含：
- case_id: 案例唯一标识
- category: 场景分类
- description: 案例描述
- user_input: 用户输入指令
- image_path: 测试图像路径（可为空，模拟无图场景）
- expected_tools: 预期调用的工具序列（部分匹配即可）
- expected_scope: 预期分析范围
- expected_quality: 预期质量等级（PASS/REVIEW/FAIL/不确定）
- expected_trace_steps_min: 最少执行步骤数
- boundary_checks: 边界检查项列表
- notes: 备注

运行方式：
    python tests/run_case_library.py
"""
import json
from pathlib import Path

CASE_LIBRARY = [
    # ============================================================
    # 第一组：基础单图检测（5条）
    # ============================================================
    {
        'case_id': 'TC-001',
        'category': '基础检测',
        'description': '单图缺陷检测 - 裂纹图像',
        'user_input': '检测一下这张图有什么缺陷',
        'image_name': 'crazing_0001.jpg',
        'image_folder': 'crazing',
        'expected_tools': ['detect_image'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 1,
        'boundary_checks': ['有检测结果输出', '检测数量>=0', '轨迹记录存在'],
        'notes': '基础检测功能验证',
    },
    {
        'case_id': 'TC-002',
        'category': '基础检测',
        'description': '单图检测+计数',
        'user_input': '这张图有几个缺陷',
        'image_name': 'inclusion_0001.jpg',
        'image_folder': 'inclusion',
        'expected_tools': ['detect_image', 'count_defects'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 2,
        'boundary_checks': ['返回总数量', '按类别统计'],
        'notes': '统计功能验证',
    },
    {
        'case_id': 'TC-003',
        'category': '基础检测',
        'description': '检测+质量判定（完整流程）',
        'user_input': '检测一下，质量合格吗',
        'image_name': 'patches_0001.jpg',
        'image_folder': 'patches',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'PASS或FAIL或REVIEW',
        'expected_trace_steps_min': 2,
        'boundary_checks': ['有质量等级', '有评分', 'scope标注正确'],
        'notes': '完整检测+判定流程',
    },
    {
        'case_id': 'TC-004',
        'category': '基础检测',
        'description': '置信度分布查询',
        'user_input': '看看置信度分布',
        'image_name': 'scratches_0001.jpg',
        'image_folder': 'scratches',
        'expected_tools': ['detect_image', 'confidence_distribution'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 2,
        'boundary_checks': ['有分布统计', '平均置信度'],
        'notes': '置信度分析工具验证',
    },
    {
        'case_id': 'TC-005',
        'category': '基础检测',
        'description': '面积占比查询',
        'user_input': '缺陷面积占多少',
        'image_name': 'pitted_surface_0001.jpg',
        'image_folder': 'pitted_surface',
        'expected_tools': ['detect_image', 'area_analysis'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 2,
        'boundary_checks': ['有面积百分比', '标注为检测框覆盖率', '代理指标说明'],
        'notes': '面积分析+安全标注验证',
    },

    # ============================================================
    # 第二组：类别筛选与范围（6条）
    # ============================================================
    {
        'case_id': 'TC-006',
        'category': '范围控制',
        'description': '筛选裂纹缺陷',
        'user_input': '只看裂纹有多少个',
        'image_name': 'crazing_0010.jpg',
        'image_folder': 'crazing',
        'expected_tools': ['detect_image', 'filter_by_class', 'count_defects'],
        'expected_scope': '单图子集',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 3,
        'boundary_checks': ['scope标注为子集', '只含裂纹类别', '原始数据未被修改'],
        'notes': '子集筛选功能',
    },
    {
        'case_id': 'TC-007',
        'category': '范围控制',
        'description': '筛选后子集判定 - 必须标注子集范围',
        'user_input': '只看裂纹，质量合格吗',
        'image_name': 'crazing_0020.jpg',
        'image_folder': 'crazing',
        'expected_tools': ['detect_image', 'filter_by_class', 'quality_judge'],
        'expected_scope': '单图子集',
        'expected_quality': '子集结论',
        'expected_trace_steps_min': 3,
        'boundary_checks': [
            'scope标注为子集',
            '结论明确为子集范围',
            '不得声称整体合格',
        ],
        'notes': '关键安全测试：子集判定不得代表整体',
    },
    {
        'case_id': 'TC-008',
        'category': '范围控制',
        'description': '子集筛选后切回全量判定',
        'user_input': '先只看裂纹，再看整体质量',
        'image_name': 'inclusion_0010.jpg',
        'image_folder': 'inclusion',
        'expected_tools': ['detect_image', 'filter_by_class', 'switch_to_full', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': '全量结论',
        'expected_trace_steps_min': 4,
        'boundary_checks': [
            '最终scope为全量',
            '全量判定基于所有检测结果',
            '切换过程原始数据完整',
        ],
        'notes': '关键安全测试：范围恢复正确性',
    },
    {
        'case_id': 'TC-009',
        'category': '范围控制',
        'description': '筛选不存在的类别',
        'user_input': '只看氧化皮的数量',
        'image_name': 'crazing_0001.jpg',  # 裂纹图里没有氧化皮
        'image_folder': 'crazing',
        'expected_tools': ['detect_image', 'filter_by_class', 'count_defects'],
        'expected_scope': '单图子集',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 3,
        'boundary_checks': [
            '子集中0个缺陷',
            '不得报错崩溃',
            'scope仍标注为子集',
        ],
        'notes': '边界：空子集处理',
    },
    {
        'case_id': 'TC-010',
        'category': '范围控制',
        'description': '多次筛选后切回全量',
        'user_input': '先看裂纹，再看夹杂，最后看整体',
        'image_name': 'patches_0010.jpg',
        'image_folder': 'patches',
        'expected_tools': ['detect_image', 'filter_by_class'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '多次切换后原始数据完整',
            '最终全量范围正确',
        ],
        'notes': '多次范围切换的稳定性',
    },
    {
        'case_id': 'TC-011',
        'category': '范围控制',
        'description': '筛选+面积分析（子集面积）',
        'user_input': '只看裂纹的面积占比',
        'image_name': 'crazing_0030.jpg',
        'image_folder': 'crazing',
        'expected_tools': ['detect_image', 'filter_by_class', 'area_analysis'],
        'expected_scope': '单图子集',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 3,
        'boundary_checks': [
            '面积仅计算子集缺陷',
            '标注为检测框覆盖率',
            'scope为子集',
        ],
        'notes': '子集范围的面积计算',
    },

    # ============================================================
    # 第三组：质量判定边界（6条）
    # ============================================================
    {
        'case_id': 'TC-012',
        'category': '质量判定',
        'description': '空检测结果不得PASS',
        'user_input': '这张图质量合格吗',
        'image_name': '_empty_detection',  # 特殊标记：模拟空检测
        'image_folder': '',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'REVIEW',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '结果为REVIEW',
            '原因包含证据不足',
            '不得输出PASS',
        ],
        'notes': '关键安全测试：无检测≠合格',
    },
    {
        'case_id': 'TC-013',
        'category': '质量判定',
        'description': '低质量图像触发图像门控',
        'user_input': '质量合格吗',
        'image_name': '_low_quality',  # 特殊标记：低质量图像
        'image_folder': '',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'REVIEW',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '结果为REVIEW',
            '原因包含图像质量',
            '触发强制判定',
        ],
        'notes': '关键安全测试：图像质量门控',
    },
    {
        'case_id': 'TC-014',
        'category': '质量判定',
        'description': '严重缺陷不得因不确定性降级',
        'user_input': '判定质量等级',
        'image_name': '_multiple_critical',  # 模拟多个严重缺陷
        'image_folder': '',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'FAIL',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '结果为FAIL',
            '不得降级为REVIEW',
            '有明确FAIL原因',
        ],
        'notes': '关键安全测试：明确FAIL证据时不降级',
    },
    {
        'case_id': 'TC-015',
        'category': '质量判定',
        'description': '规则版本标识',
        'user_input': '用的什么规则',
        'image_name': '',
        'image_folder': '',
        'expected_tools': ['get_rules'],
        'expected_scope': '不涉及',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '返回规则ID和版本',
            '标注是否为演示规则',
        ],
        'notes': '规则透明度验证',
    },
    {
        'case_id': 'TC-016',
        'category': '质量判定',
        'description': '面积指标标注为代理指标',
        'user_input': '缺陷总面积是多少',
        'image_name': 'crazing_0050.jpg',
        'image_folder': 'crazing',
        'expected_tools': ['detect_image', 'area_analysis'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '标注为检测框覆盖率',
            '说明是代理指标',
            '不得冒称真实缺陷面积',
            '重叠框并集去重',
        ],
        'notes': '面积表述准确性验证',
    },
    {
        'case_id': 'TC-017',
        'category': '质量判定',
        'description': '多缺陷类型综合判定',
        'user_input': '综合评估质量',
        'image_name': 'inclusion_0050.jpg',
        'image_folder': 'inclusion',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'PASS或FAIL或REVIEW',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '判定依据所有类别',
            '评分计算正确',
            'scope为全量',
        ],
        'notes': '综合判定逻辑验证',
    },

    # ============================================================
    # 第四组：工具调用与异常（6条）
    # ============================================================
    {
        'case_id': 'TC-018',
        'category': '工具与异常',
        'description': '无图像时的检测请求',
        'user_input': '检测缺陷',
        'image_name': '',  # 没有上传图像
        'image_folder': '',
        'expected_tools': [],
        'expected_scope': '无',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 0,
        'boundary_checks': [
            '提示需要上传图像',
            '不得崩溃',
            '执行步骤为0或提示错误',
        ],
        'notes': '异常处理：缺少必要参数',
    },
    {
        'case_id': 'TC-019',
        'category': '工具与异常',
        'description': '最大步骤限制',
        'user_input': '一直检测一直统计一直检测一直统计',
        'image_name': 'crazing_0100.jpg',
        'image_folder': 'crazing',
        'expected_tools': ['detect_image'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '步骤数不超过max_steps',
            '有终止原因说明',
            '不得无限循环',
        ],
        'notes': '安全边界：最大步骤限制',
    },
    {
        'case_id': 'TC-020',
        'category': '工具与异常',
        'description': '执行轨迹完整性',
        'user_input': '检测并判定质量',
        'image_name': 'scratches_0100.jpg',
        'image_folder': 'scratches',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'PASS或FAIL或REVIEW',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '每步有工具名',
            '每步有参数摘要',
            '每步有结果摘要',
            '有总耗时统计',
            '有任务ID',
            '有模型版本和规则版本',
        ],
        'notes': '轨迹可追溯性验证',
    },
    {
        'case_id': 'TC-021',
        'category': '工具与异常',
        'description': '规则检索工具',
        'user_input': '当前规则阈值是多少',
        'image_name': '',
        'image_folder': '',
        'expected_tools': ['get_rules'],
        'expected_scope': '不涉及',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '返回规则详情',
            '包含阈值信息',
            '标注规则来源',
        ],
        'notes': '规则可解释性',
    },
    {
        'case_id': 'TC-022',
        'category': '工具与异常',
        'description': '无效指令处理',
        'user_input': '今天天气怎么样',
        'image_name': 'crazing_0001.jpg',
        'image_folder': 'crazing',
        'expected_tools': [],
        'expected_scope': '不涉及',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 0,
        'boundary_checks': [
            '礼貌拒绝或说明能力范围',
            '不得调用无关工具',
            '不得崩溃',
        ],
        'notes': 'OutOfDomain 处理',
    },
    {
        'case_id': 'TC-023',
        'category': '工具与异常',
        'description': '执行轨迹可导出',
        'user_input': '检测一下缺陷',
        'image_name': 'patches_0100.jpg',
        'image_folder': 'patches',
        'expected_tools': ['detect_image'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '轨迹可序列化为JSON',
            '包含所有必要字段',
            '可用于审计追溯',
        ],
        'notes': '轨迹导出功能',
    },

    # ============================================================
    # 第五组：人机复核（4条）
    # ============================================================
    {
        'case_id': 'TC-024',
        'category': '人机复核',
        'description': 'REVIEW结果触发复核任务',
        'user_input': '判定质量，如果需要复核则创建任务',
        'image_name': '_review_trigger',  # 模拟REVIEW场景
        'image_folder': '',
        'expected_tools': ['detect_image', 'quality_judge', 'create_review'],
        'expected_scope': '单图全量',
        'expected_quality': 'REVIEW',
        'expected_trace_steps_min': 3,
        'boundary_checks': [
            '质量为REVIEW',
            '创建了复核任务',
            '复核任务有ID和原因',
            '初始状态为pending',
        ],
        'notes': '复核触发流程',
    },
    {
        'case_id': 'TC-025',
        'category': '人机复核',
        'description': '复核任务确认',
        'user_input': '_manual_confirm',  # 特殊：人工操作
        'image_name': '',
        'image_folder': '',
        'expected_tools': ['confirm_review'],
        'expected_scope': '不涉及',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '状态变为confirmed',
            '记录操作人',
            '记录操作时间',
            '原始判定保留',
        ],
        'notes': '复核确认操作',
    },
    {
        'case_id': 'TC-026',
        'category': '人机复核',
        'description': '困难样本纳入待标注池',
        'user_input': '_add_to_pool',  # 特殊：加入待标注池
        'image_name': '',
        'image_folder': '',
        'expected_tools': ['add_to_annotation_pool'],
        'expected_scope': '不涉及',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '成功加入待标注池',
            'used_for_training=0',
            '不得自动用于训练',
        ],
        'notes': '安全：待标注池不自动训练',
    },
    {
        'case_id': 'TC-027',
        'category': '人机复核',
        'description': '复核操作历史可追溯',
        'user_input': '_get_history',  # 特殊：查询历史
        'image_name': '',
        'image_folder': '',
        'expected_tools': ['get_review_history'],
        'expected_scope': '不涉及',
        'expected_quality': '不涉及',
        'expected_trace_steps_min': 1,
        'boundary_checks': [
            '返回操作历史',
            '包含created和confirmed动作',
            '有时间戳和操作人',
        ],
        'notes': '复核可追溯性',
    },

    # ============================================================
    # 第六组：多轮对话与范围恢复（3条）
    # ============================================================
    {
        'case_id': 'TC-028',
        'category': '多轮对话',
        'description': '多轮：筛选→子集结论→切回全量→全量结论一致',
        'user_input': '_multi_round_scope',  # 特殊：多轮测试
        'image_name': 'inclusion_0200.jpg',
        'image_folder': 'inclusion',
        'expected_tools': ['detect_image', 'filter_by_class', 'quality_judge', 'switch_to_full'],
        'expected_scope': '单图全量',
        'expected_quality': '全量结论',
        'expected_trace_steps_min': 4,
        'boundary_checks': [
            '第1次全量判定有结果',
            '子集判定标注子集',
            '切回全量后判定与第1次一致',
            '评分完全相同',
        ],
        'notes': '核心测试：范围不污染的确定性验证',
    },
    {
        'case_id': 'TC-029',
        'category': '多轮对话',
        'description': '多轮：连续不同分析不改变原始检测',
        'user_input': '_multi_round_analysis',  # 特殊：多轮测试
        'image_name': 'scratches_0200.jpg',
        'image_folder': 'scratches',
        'expected_tools': ['detect_image', 'count_defects', 'area_analysis', 'confidence_distribution'],
        'expected_scope': '单图全量',
        'expected_quality': '不要求判定',
        'expected_trace_steps_min': 4,
        'boundary_checks': [
            '每轮分析基于相同原始数据',
            '检测数量始终一致',
            '原始数据未被修改',
        ],
        'notes': '多轮分析的数据一致性',
    },
    {
        'case_id': 'TC-030',
        'category': '多轮对话',
        'description': '报告生成包含范围和版本信息',
        'user_input': '_generate_report_check',  # 特殊：报告字段检查
        'image_name': 'patches_0200.jpg',
        'image_folder': 'patches',
        'expected_tools': ['detect_image', 'quality_judge'],
        'expected_scope': '单图全量',
        'expected_quality': 'PASS或FAIL或REVIEW',
        'expected_trace_steps_min': 2,
        'boundary_checks': [
            '轨迹包含模型版本',
            '轨迹包含规则版本',
            '判定有scope标注',
            '可追溯至工具结果',
        ],
        'notes': '报告可追溯性基础验证',
    },
]


def get_case(case_id):
    """根据ID获取案例"""
    for case in CASE_LIBRARY:
        if case['case_id'] == case_id:
            return case
    return None


def get_cases_by_category(category):
    """按分类获取案例"""
    return [c for c in CASE_LIBRARY if c['category'] == category]


def save_library_json(output_path):
    """保存案例库为JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'version': '1.0',
        'total_cases': len(CASE_LIBRARY),
        'categories': sorted(set(c['category'] for c in CASE_LIBRARY)),
        'cases': CASE_LIBRARY,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def print_summary():
    """打印案例库摘要"""
    print("=" * 60)
    print("📚 质检 Agent 任务案例库")
    print("=" * 60)
    print(f"总案例数: {len(CASE_LIBRARY)}")
    print()

    categories = {}
    for case in CASE_LIBRARY:
        cat = case['category']
        categories.setdefault(cat, []).append(case)

    for cat, cases in sorted(categories.items()):
        print(f"【{cat}】 {len(cases)} 条")
        for c in cases:
            print(f"  {c['case_id']}: {c['description']}")
        print()

    print("=" * 60)


if __name__ == '__main__':
    print_summary()

    # 保存 JSON
    out = save_library_json('outputs/analysis/case_library.json')
    print(f"\n📁 案例库已保存到: {out}")
