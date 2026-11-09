// -*- coding: utf-8 -*-
/**
 * 生成实验报告（基于代码事实）
 * 
 * 所有数据均来自真实运行结果，不编造任何指标。
 * 未完成项明确标记。
 */

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, ImageRun
} = require('docx');

// ============================================================
// 配置
// ============================================================
const OUTPUT_PATH = path.join(__dirname, '..', '实验报告_Agent升级版.docx');
const SCREENSHOT_DIR = path.join(__dirname, '..', 'outputs', 'screenshots');

const border = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

// ============================================================
// 辅助函数
// ============================================================

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun(text)]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun(text)]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, bold: true, size: 24 })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    ...opts,
    children: [new TextRun({ text, size: 22, ...opts.run })]
  });
}

function boldPara(label, value) {
  return new Paragraph({
    spacing: { after: 100, line: 360 },
    children: [
      new TextRun({ text: label, bold: true, size: 22 }),
      new TextRun({ text: value, size: 22 })
    ]
  });
}

function bulletItem(text, ref = 'bullets') {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 360 },
    children: [new TextRun({ text, size: 22 })]
  });
}

function warnPara(text) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    shading: { fill: 'FFF3CD', type: ShadingType.CLEAR },
    indent: { left: 200, right: 200 },
    children: [
      new TextRun({ text: '⚠️ ', size: 22 }),
      new TextRun({ text, size: 22 })
    ]
  });
}

function infoPara(text) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    shading: { fill: 'D1ECF1', type: ShadingType.CLEAR },
    indent: { left: 200, right: 200 },
    children: [
      new TextRun({ text: 'ℹ️ ', size: 22 }),
      new TextRun({ text, size: 22 })
    ]
  });
}

// 简单表格生成
function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const tableRows = [];

  // 表头
  tableRows.push(new TableRow({
    cantSplit: true,
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: '2E86AB', type: ShadingType.CLEAR },
      margins: cellMargins,
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: h, bold: true, size: 22, color: 'FFFFFF' })]
      })]
    }))
  }));

  // 数据行
  rows.forEach((row, rowIdx) => {
    tableRows.push(new TableRow({
      cantSplit: true,
      children: row.map((cell, i) => new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: rowIdx % 2 === 0 ? 'F8F9FA' : 'FFFFFF', type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({
          children: [new TextRun({ text: String(cell), size: 21 })]
        })]
      }))
    }));
  });

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows
  });
}

function spacer(twips = 200) {
  return new Paragraph({ spacing: { after: twips }, children: [] });
}

// ============================================================
// 封面
// ============================================================

function buildCover() {
  return [
    spacer(2000),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [new TextRun({
        text: 'AI Agent 智能缺陷检测系统',
        bold: true, size: 56,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({
        text: '实验报告（Agent 升级版）',
        size: 36,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' },
        color: '666666'
      })]
    }),
    spacer(600),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [new TextRun({
        text: '基于 YOLOv8 + 质检 Agent 闭环',
        size: 26,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [new TextRun({
        text: '范围隔离 · 安全判定 · 可追溯 · 人机协同',
        size: 24,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' },
        color: '2E86AB'
      })]
    }),
    spacer(2000),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new TextRun({
        text: '版本：v2.0（Agent 增强版）',
        size: 22,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new TextRun({
        text: '日期：2026 年 11 月',
        size: 22,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new TextRun({
        text: '竞赛方向：AI Agent 赋能行业创新 — 制造领域质量检测',
        size: 22,
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' },
        color: '666666'
      })]
    }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第一章：项目概述
// ============================================================

function buildChapter1() {
  return [
    heading1('第一章  项目概述'),

    heading2('1.1 项目背景'),
    para('本项目面向制造领域的表面质量检测场景，以东北大学 NEU-CLS 热轧带钢表面缺陷数据集为基础，构建了一套基于 YOLOv8 的智能缺陷检测系统。本次升级将系统从"检测模型 + 聊天分析界面"进一步升级为"能够理解质检任务、调用工具获取证据、执行受控质量判定、处理异常、组织人工复核并生成可追溯报告的质检 Agent"。'),

    heading2('1.2 升级目标'),
    para('本次 v2 升级的核心目标是完善 Agent 闭环、正确性和实验可信度，具体包括：'),
    bulletItem('实现真正可验证的 Agent 执行（有界循环、工具白名单、执行轨迹）'),
    bulletItem('修复数据范围与状态污染（子集分析不影响全量结论）'),
    bulletItem('重构质量判定的安全边界（图像门控、面积代理指标、演示规则标注）'),
    bulletItem('实现人机复核闭环与可追溯报告'),
    bulletItem('提升实验可复现性（固定种子、自动指标提取）'),
    bulletItem('建立可复现的任务案例库和演示流程'),
    bulletItem('安全加固（文件上传校验、操作授权、SQL 安全）'),

    heading2('1.3 技术栈'),
    makeTable(
      ['模块', '技术方案', '说明'],
      [
        ['检测模型', 'YOLOv8s', '基线模型，基于 NEU-CLS 数据集训练'],
        ['Agent 执行器', '自研 v2', '双路径设计（规则模式 + LLM 预留）'],
        ['质量判定', 'QualityEngineV2', '图像门控 + 评分制 + 安全边界'],
        ['任务上下文', 'TaskContext', '原始数据只读 + 子集视图，防污染'],
        ['执行轨迹', 'ExecutionTrace', '每步工具调用记录，可导出 JSON'],
        ['复核工作流', 'ReviewWorkflow', 'SQLite 持久化，操作历史可追溯'],
        ['前端界面', 'Streamlit', 'v2 页面含执行轨迹面板和安全提示'],
        ['数据增强', 'cDCGAN + Copy-Paste', '保留 v1 实现，本次未改动'],
        ['数据库', 'SQLite', '检测记录 + 复核任务 + 轨迹存储'],
      ],
      [2200, 2800, 4360]
    ),

    heading2('1.4 版本说明'),
    warnPara('本报告中所有质量规则均为演示规则（demo_default_v1.0.0），用于验证系统功能，不构成工业质量判定依据。面积指标为"检测框覆盖率（并集去重）"，是真实缺陷面积的代理指标，非物理测量值。'),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第二章：Agent v2 核心架构
// ============================================================

function buildChapter2() {
  return [
    heading1('第二章  Agent v2 核心架构'),

    heading2('2.1 整体架构'),
    para('Agent v2 采用分层架构，从上到下分为四层：'),
    bulletItem('交互层：Streamlit v2 页面，三栏布局（输入/对话/轨迹）'),
    bulletItem('执行层：AgentExecutorV2，负责任务规划、工具调度、有界循环'),
    bulletItem('能力层：7+ 个工具（检测、统计、判定、筛选、规则查询等）'),
    bulletItem('数据层：TaskContext（任务上下文）+ SQLite（持久化）'),

    heading2('2.2 任务上下文与范围隔离'),
    para('任务上下文（TaskContext）是 Agent v2 的核心数据结构，解决了 v1 中"筛选后判定污染全量"的关键问题。'),
    heading3('设计原则'),
    bulletItem('原始检测数据（raw_detections）只读，任何筛选都不修改原始数据'),
    bulletItem('筛选生成独立的子集视图（SubsetView），仅包含引用而非复制'),
    bulletItem('质量判定必须明确标注 scope（全量/子集）'),
    bulletItem('从子集切回全量时，恢复完整原始数据，判定结果可重现'),

    heading3('安全保证'),
    makeTable(
      ['安全点', '实现方式', '验证状态'],
      [
        ['原始数据不被修改', 'raw_detections 属性只读，筛选返回新视图', '✅ 单元测试通过'],
        ['子集判定标注范围', 'quality_judge 输出 scope_label', '✅ 30 案例验证'],
        ['切回全量一致性', 'switch_to_full() 恢复 raw_detections', '✅ 集成测试通过'],
        ['空检测不 PASS', 'has_raw_detections 检查，强制 REVIEW', '✅ 边界测试通过'],
        ['范围不泄漏', '所有工具读取 current_view，不直接访问 raw', '✅ 架构保证'],
      ],
      [2400, 4200, 2760]
    ),

    heading2('2.3 Agent 执行器 v2'),
    heading3('双路径设计'),
    para('Agent v2 支持两种执行模式，LLM 不可用时自动降级为规则模式：'),
    makeTable(
      ['模式', '触发条件', '规划方式', '特点'],
      [
        ['规则模式', 'llm_provider=none 或 LLM 调用失败', '关键词匹配 + 启发式规划', '确定性高，可复现，无调用成本'],
        ['LLM 模式', '配置了 API Key 且网络正常', '大模型结构化工具调用', '灵活性高，支持复杂任务理解'],
      ],
      [1600, 2600, 2600, 2560]
    ),

    heading3('有界执行循环'),
    para('为防止 Agent 失控，实现了多重安全边界：'),
    bulletItem('最大步骤限制（max_steps，默认 10）'),
    bulletItem('费用预算限制（max_cost_credits，LLM 模式）'),
    bulletItem('工具超时机制（per_tool_timeout_s）'),
    bulletItem('有限重试（max_retries，仅幂等工具）'),
    bulletItem('工具白名单（仅允许注册的工具）'),
    bulletItem('参数校验（ToolSchema 验证所有输入参数）'),

    heading2('2.4 执行轨迹与可观察性'),
    para('Agent 的每一步工具调用都记录在执行轨迹中，确保过程可追溯、可审计。'),
    heading3('轨迹内容'),
    makeTable(
      ['字段', '说明', '示例'],
      [
        ['task_id', '任务唯一标识', 'task_20261105_abc123'],
        ['step', '步骤序号', '1, 2, 3...'],
        ['tool', '调用的工具名', 'detect_image, quality_judge'],
        ['action', '简短行动说明', '图像缺陷检测'],
        ['params', '参数摘要（脱敏）', 'image_path: test.jpg'],
        ['result', '结果摘要', '检测到 5 个缺陷'],
        ['status', '执行状态', 'success / error / timeout'],
        ['time_ms', '执行耗时（毫秒）', '156.3'],
        ['error', '错误信息（如有）', '图像文件不存在'],
        ['model_version', '模型版本', 'yolov8s_baseline-2'],
        ['rule_version', '规则版本', 'demo_default_v1.0.0'],
      ],
      [2000, 3500, 3860]
    ),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第三章：质量判定安全边界
// ============================================================

function buildChapter3() {
  return [
    heading1('第三章  质量判定安全边界'),

    para('质量判定是质检系统的核心输出，必须保证安全、可控、可解释。v2 版本对质量判定引擎进行了全面的安全重构。'),

    heading2('3.1 三重分离建模'),
    para('将三个容易混淆的概念分开建模，避免逻辑混淆：'),
    makeTable(
      ['概念', '定义', '来源', '用途'],
      [
        ['检测置信度', '模型对每个检测框的置信程度', 'YOLO 模型输出', '筛选低置信检测，不直接决定质量'],
        ['缺陷类别与严重程度', '缺陷的类型和对产品质量的影响等级', '规则配置（演示规则）', '质量评分计算的输入之一'],
        ['质量规则', '判定产品合格/复核/不合格的阈值和逻辑', '可配置规则集', '最终质量判定依据'],
      ],
      [2000, 3000, 2200, 2160]
    ),
    warnPara('高置信度不等于缺陷可以接受，小面积也不能自动抵消关键缺陷风险。置信度仅反映模型的确定性，不反映缺陷的严重性。'),

    heading2('3.2 图像质量门控'),
    para('在进行质量判定前，先检查图像质量。低质量图像不得直接给出 PASS 结论，必须进入 REVIEW。'),
    heading3('门控规则'),
    bulletItem('quality_score < 0.5：强制 REVIEW，原因标注"图像质量不足"'),
    bulletItem('存在严重质量问题（模糊/过曝/失焦）：强制 REVIEW'),
    bulletItem('检测失败或无检测数据：强制 REVIEW，原因标注"证据不足"'),
    bulletItem('已有明确 FAIL 证据：不降级，保持 FAIL（即使图像质量有问题）'),

    heading2('3.3 面积计算的诚实表述'),
    para('v2 版本对面积指标的表述进行了严格规范，避免误导：'),
    makeTable(
      ['表述项', 'v1 问题', 'v2 修正'],
      [
        ['指标名称', '缺陷面积', '检测框覆盖率'],
        ['计算方法', '检测框面积求和', '检测框并集面积（扫描线法去重）'],
        ['单位', '像素（易被误解为真实面积）', '像素占比（%），明确标注代理指标'],
        ['物理含义', '暗示为真实缺陷面积', '明确说明是代理指标，不等同于真实缺陷面积'],
        ['重叠处理', '可能重复计算', '并集去重，避免高估'],
      ],
      [1800, 3200, 4360]
    ),
    infoPara('在没有物理标定（像素→实际尺寸换算比例）的情况下，系统不会将像素面积转换为实际面积，也不会输出如"缺陷面积 5.2mm²"之类的表述。'),

    heading2('3.4 演示规则明确标注'),
    para('当前使用的质量规则为演示规则，所有输出均明确标注：'),
    bulletItem('规则 ID：demo_default'),
    bulletItem('版本：v1.0.0'),
    bulletItem('来源：系统内置演示规则'),
    bulletItem('is_demo: True'),
    bulletItem('审批状态：未审批（演示用）'),

    warnPara('演示规则仅供验证系统功能，不构成任何工业质量判定依据。实际应用中，质量规则应由工艺工程师和质量专家共同制定，并经过正式审批流程。'),

    heading2('3.5 质量判定矩阵'),
    makeTable(
      ['场景', '判定结果', '是否强制', '原因说明'],
      [
        ['无检测数据', 'REVIEW', '是', '证据不足，无法判定'],
        ['低质量图像（<0.5）', 'REVIEW', '是', '图像质量不足，结果不可靠'],
        ['有明确 FAIL 证据', 'FAIL', '否', '缺陷数量/严重程度超过阈值'],
        ['有缺陷但不严重', 'PASS', '否', '在规则允许范围内'],
        ['低置信缺陷过多', 'REVIEW', '否', '结果不确定性高，需人工确认'],
        ['空检测 + 高质量图像', 'REVIEW', '是', '未发现缺陷≠产品合格，需人工确认'],
      ],
      [2600, 1400, 1200, 4160]
    ),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第四章：人机复核闭环
// ============================================================

function buildChapter4() {
  return [
    heading1('第四章  人机复核闭环'),

    heading2('4.1 复核工作流设计'),
    para('当质量判定结果为 REVIEW 或 FAIL 时，系统可创建复核任务，进入人工复核流程。'),
    makeTable(
      ['状态', '说明', '可执行操作'],
      [
        ['pending', '待处理', '确认 / 驳回 / 修正'],
        ['confirmed', '已确认（原判定正确）', '查看历史 / 加入待标注池'],
        ['rejected', '已驳回（原判定有误）', '查看历史 / 加入待标注池'],
        ['modified', '已修正（人工修改了结果）', '查看历史 / 加入待标注池'],
        ['closed', '已关闭', '查看历史'],
      ],
      [1800, 3000, 4560]
    ),

    heading2('4.2 可追溯的操作历史'),
    para('每次状态变更都记录完整的操作历史，包括：'),
    bulletItem('操作类型（created / confirmed / rejected / modified / closed）'),
    bulletItem('操作人（actor）'),
    bulletItem('操作时间（精确到秒）'),
    bulletItem('操作备注（note）'),
    bulletItem('修改前后的内容（modified 操作）'),

    heading2('4.3 待标注池与训练安全'),
    para('复核中发现的困难样本可加入待标注池，供后续标注和研究使用。但为保证安全：'),
    makeTable(
      ['安全措施', '实现方式', '目的'],
      [
        ['默认不用于训练', 'used_for_training=0', '防止未经审核的样本进入训练集'],
        ['需人工显式授权', '修改训练状态需确认', '避免污染模型训练数据'],
        ['不在线替换模型', '无自动模型替换接口', '防止生产模型被意外更换'],
        ['保留原始证据', '检测结果不被覆盖', '保证可追溯性和审计能力'],
      ],
      [2400, 3200, 3760]
    ),
    warnPara('系统不会自动将未经审核的预测或复核记录用于训练，也不会在线自动替换生产模型。任何涉及模型更新的操作都需要人工明确授权。'),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第五章：数据集与实验可复现性
// ============================================================

function buildChapter5() {
  return [
    heading1('第五章  数据集与实验可复现性'),

    heading2('5.1 数据集统计'),
    para('使用 analyze_dataset.py 工具对 NEU-CLS 原始数据集进行了统计：'),
    makeTable(
      ['类别', '中文名', '图像数', '占比'],
      [
        ['crazing', '裂纹', '300', '16.67%'],
        ['inclusion', '夹杂', '300', '16.67%'],
        ['patches', '斑块', '300', '16.67%'],
        ['pitted_surface', '麻点', '300', '16.67%'],
        ['rolled-in_scale', '氧化皮', '300', '16.67%'],
        ['scratches', '划痕', '300', '16.67%'],
        ['合计', '-', '1800', '100%'],
      ],
      [2400, 1800, 1800, 1800]
    ),
    para('图像尺寸：全部为 200 × 200 像素（唯一尺寸）。'),
    infoPara('NEU-CLS 数据集每类 300 张，类别数量完全平衡（1:1:1:1:1:1），不存在类别不平衡问题。裂纹属于"困难类别"（检测难度较高），但不是"小样本类别"。'),

    heading2('5.2 数据划分'),
    para('使用 split_dataset.py 工具进行数据划分，保证可复现：'),
    makeTable(
      ['项目', '配置', '说明'],
      [
        ['随机种子', '42', '固定种子，结果可复现'],
        ['划分方式', '分层抽样', '保持各类别比例一致'],
        ['训练集', '1260 张（每类 210 张）', '70%'],
        ['验证集', '270 张（每类 45 张）', '15%'],
        ['测试集', '270 张（每类 45 张）', '15%'],
        ['泄漏检测', '无泄漏', '同一张图不会出现在多个集合'],
      ],
      [2200, 3200, 3960]
    ),
    para('划分清单保存在 data/splits/split_seed42_v1/ 目录下，包含 train.txt、val.txt、test.txt 和 split_config.json。'),

    heading2('5.3 基线实验指标'),
    para('使用 extract_metrics.py 工具从 results.csv 自动提取基线实验指标：'),
    makeTable(
      ['指标', 'yolov8s_baseline', 'yolov8s_baseline-2', '变化（百分点）'],
      [
        ['Precision', '74.38%', '77.08%', '+2.7 pp'],
        ['Recall', '69.15%', '70.13%', '+1.0 pp'],
        ['mAP@50', '75.34%', '75.55%', '+0.2 pp'],
        ['mAP@50-95', '42.48%', '41.45%', '-1.0 pp'],
      ],
      [2000, 2500, 2500, 2360]
    ),
    warnPara('基线对比结果显示：yolov8s_baseline-2 相比 baseline，mAP@50 仅提升 0.2 个百分点，mAP@50-95 反而下降 1.0 个百分点。两次实验的差异较小，可能主要来自训练随机性，不能据此声称"有显著提升"。'),

    heading2('5.4 实验可复现性说明'),
    heading3('已实现的可复现措施'),
    bulletItem('数据划分固定随机种子，清单可复现'),
    bulletItem('指标自动从原始 results.csv 提取，不依赖人工抄写'),
    bulletItem('绝对变化使用"个百分点（pp）"，相对提升单独计算'),
    bulletItem('所有实验配置保存在 args.yaml 中'),

    heading3('待完善项（未执行）'),
    bulletItem('多随机种子实验（建议 ≥3 个种子，报告均值和标准差）'),
    bulletItem('Copy-Paste 增强对比实验'),
    bulletItem('cDCGAN + Copy-Paste 增强对比实验'),
    bulletItem('鲁棒性评估（噪声/模糊/亮度扰动下的 mAP/Recall）'),
    bulletItem('校准评估（置信度可靠性、ECE）'),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第六章：Agent 任务案例库
// ============================================================

function buildChapter6() {
  return [
    heading1('第六章  Agent 任务案例库'),

    heading2('6.1 案例库概述'),
    para('建立了 30 条可复现的任务案例，覆盖 6 大类场景，每条案例都有明确的预期结果和边界检查项。'),
    makeTable(
      ['分类', '案例数', '覆盖场景'],
      [
        ['基础检测', '5', '单图检测、计数、判定、置信度、面积'],
        ['范围控制', '6', '子集筛选、子集判定、切回全量、空子集、多次切换'],
        ['质量判定', '6', '空检测、低质量图像、FAIL不降级、规则版本、面积标注'],
        ['工具与异常', '6', '无图像、最大步骤、轨迹完整性、规则检索、无效指令、轨迹导出'],
        ['人机复核', '4', 'REVIEW触发、复核确认、待标注池、操作历史'],
        ['多轮对话', '3', '范围恢复一致性、多轮分析一致性、报告可追溯'],
      ],
      [2200, 1200, 5960]
    ),

    heading2('6.2 测试结果'),
    para('使用 run_case_library.py 自动运行所有 30 条案例，结果如下：'),
    makeTable(
      ['指标', '数值', '说明'],
      [
        ['总案例数', '30', '6 大分类'],
        ['PASS', '30', '100% 通过'],
        ['FAIL', '0', '-'],
        ['ERROR', '0', '-'],
        ['自动边界检查项', '60+', '由代码架构保证的安全检查'],
        ['待人工验证项', '20+', '标注和表述类（建议抽查）'],
      ],
      [2800, 1400, 5160]
    ),

    heading2('6.3 关键安全案例'),
    heading3('TC-007：子集判定不得代表整体'),
    para('验证：仅筛选裂纹后进行质量判定，结果必须明确标注为子集范围，不得声称整体合格。'),
    bulletItem('预期：scope 标注为"单图子集"'),
    bulletItem('预期：结论中包含子集范围说明'),
    bulletItem('结果：✅ PASS'),

    heading3('TC-008：子集筛选后切回全量，判定一致'),
    para('验证：先筛选裂纹，再切回全量判定，全量判定结果应与直接全量判定完全一致。'),
    bulletItem('预期：切回全量后缺陷数量 = 原始缺陷数量'),
    bulletItem('预期：切回全量后质量等级 = 原始质量等级'),
    bulletItem('结果：✅ PASS'),

    heading3('TC-012：空检测结果不得 PASS'),
    para('验证：没有检测到任何缺陷时，不得输出 PASS，必须进入 REVIEW。'),
    bulletItem('预期：结果为 REVIEW'),
    bulletItem('预期：原因包含"证据不足"'),
    bulletItem('结果：✅ PASS'),

    heading3('TC-013：低质量图像触发门控'),
    para('验证：图像质量差（模糊）时，即使缺陷不多，也必须强制 REVIEW。'),
    bulletItem('预期：结果为 REVIEW'),
    bulletItem('预期：触发强制判定（is_forced=True）'),
    bulletItem('预期：原因包含图像质量问题'),
    bulletItem('结果：✅ PASS'),

    heading3('TC-014：明确 FAIL 证据不降级'),
    para('验证：有明确 FAIL 证据时，不得因其他不确定性降级为 REVIEW。'),
    bulletItem('预期：结果为 FAIL'),
    bulletItem('预期：不触发强制降级'),
    bulletItem('结果：✅ PASS'),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第七章：安全加固
// ============================================================

function buildChapter7() {
  return [
    heading1('第七章  安全加固'),

    heading2('7.1 文件上传安全'),
    makeTable(
      ['安全措施', '实现方式', '目的'],
      [
        ['扩展名白名单', '.jpg/.jpeg/.png/.bmp/.webp/.tiff', '防止上传可执行文件'],
        ['文件大小限制', '最大 20MB', '防止超大文件攻击'],
        ['魔数检查', '校验文件头真实类型', '防止扩展名伪造'],
        ['路径遍历防护', 'basename 清理 + 安全文件名生成', '防止目录穿越'],
        ['文件名哈希', 'MD5 哈希 + 时间戳命名', '防止文件名注入和冲突'],
      ],
      [2200, 3600, 3560]
    ),

    heading2('7.2 SQL 操作安全'),
    makeTable(
      ['安全措施', '实现方式', '目的'],
      [
        ['参数化查询', '所有查询使用 ? 占位符', '防止 SQL 注入'],
        ['标识符白名单', '表名/列名通过白名单验证', '防止标识符注入'],
        ['safe_query 封装', '统一安全查询接口', '规范数据库操作'],
      ],
      [2200, 3600, 3560]
    ),

    heading2('7.3 高危操作授权'),
    para('以下操作需要用户明确确认才能执行：'),
    makeTable(
      ['操作', '风险等级', '说明'],
      [
        ['删除检测记录', '高', '数据不可恢复'],
        ['清空所有记录', '高', '大量数据丢失'],
        ['修改质量规则', '高', '影响所有后续判定'],
        ['触发模型训练', '中', '消耗大量算力'],
        ['替换生产模型', '高', '直接影响检测结果'],
        ['导出全部数据', '中', '数据泄露风险'],
      ],
      [2400, 1400, 5560]
    ),

    heading2('7.4 其他安全措施'),
    bulletItem('输入文本清理：移除控制字符，限制最大长度（2000字符）'),
    bulletItem('会话数据隔离：每个会话独立存储，定期自动清理（24小时）'),
    bulletItem('工具白名单：Agent 只能调用已注册的工具，防止越权'),
    bulletItem('外部内容不可信：文档/检索内容不覆盖系统指令和规则'),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第八章：演示流程
// ============================================================

function buildChapter8() {
  return [
    heading1('第八章  核心演示流程'),

    heading2('演示一：批次检测 → 规则判定 → 异常复核 → 报告'),
    heading3('演示目的'),
    para('展示完整的质检 Agent 工作闭环：从图像检测到质量判定，再到人工复核和报告生成，全过程可追溯。'),
    heading3('演示步骤'),
    bulletItem('步骤1：上传图像，执行缺陷检测'),
    bulletItem('步骤2：调用质量规则，输出判定结果'),
    bulletItem('步骤3：REVIEW/FAIL 结果触发复核任务创建'),
    bulletItem('步骤4：人工复核（确认/驳回/修正），记录操作历史'),
    bulletItem('步骤5：生成可追溯报告（含范围、版本、证据引用）'),
    heading3('验证结果'),
    para('✅ 演示通过 — 完整闭环可执行，每步有轨迹记录，复核操作可追溯。'),

    heading2('演示二：子集筛选 → 切回全量（范围不污染证明）'),
    heading3('演示目的'),
    para('证明子集分析不会污染全量质量结论，相同证据 + 相同规则 = 一致判定。'),
    heading3('演示步骤'),
    bulletItem('步骤1：全量检测 + 质量判定（基线）'),
    bulletItem('步骤2：仅筛选裂纹 → 子集判定（标注子集范围）'),
    bulletItem('步骤3：切回全量 → 全量判定'),
    bulletItem('步骤4：对比第1步和第3步的结果，验证完全一致'),
    heading3('验证结果'),
    para('✅ 演示通过 — 全量缺陷数一致，质量等级一致，子集判定明确标注范围。'),

    heading2('演示三：安全边界 — 低质量/无证据 → REVIEW，不误放行'),
    heading3('演示目的'),
    para('展示系统的安全边界：低质量图像、无检测数据不会被误判为 PASS，明确 FAIL 证据不降级。'),
    heading3('演示步骤'),
    bulletItem('场景1：低质量图像（模糊）→ 强制 REVIEW'),
    bulletItem('场景2：无检测数据 → 强制 REVIEW（证据不足）'),
    bulletItem('场景3：多个严重缺陷 → FAIL（不降级）'),
    heading3('验证结果'),
    para('✅ 演示通过 — 所有安全边界均正确触发，有明确原因说明。'),

    heading2('演示运行方式'),
    para('三条演示流程已脚本化，可一键运行并自动验证：'),
    new Paragraph({
      spacing: { after: 120 },
      shading: { fill: 'F5F5F5', type: ShadingType.CLEAR },
      indent: { left: 360 },
      children: [new TextRun({
        text: 'python tools/demo_flows.py --demo all',
        size: 20,
        font: { ascii: 'Consolas', eastAsia: 'Consolas' }
      })]
    }),

    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ============================================================
// 第九章：结论与未完成项
// ============================================================

function buildChapter9() {
  return [
    heading1('第九章  结论与未完成项'),

    heading2('9.1 已完成工作总结'),
    heading3('P0 核心能力（全部完成）'),
    makeTable(
      ['模块', '功能', '验证状态'],
      [
        ['任务上下文', '原始数据只读 + 子集视图 + 范围恢复', '✅ 单元+集成测试'],
        ['工具Schema', '15 个工具定义 + 参数校验', '✅ 单元测试'],
        ['执行轨迹', '每步记录 + JSON 导出 + SQLite 持久化', '✅ 集成测试'],
        ['质量判定 v2', '图像门控 + 面积并集 + 演示规则标注', '✅ 6 项边界测试'],
        ['Agent 执行器 v2', '双路径 + 有界循环 + 白名单 + 超时重试', '✅ 30 案例验证'],
        ['复核工作流', '任务 CRUD + 操作历史 + 待标注池安全', '✅ 集成测试'],
        ['Streamlit v2 页面', '三栏布局 + 轨迹面板 + 安全提示', '✅ HTTP 验证'],
      ],
      [2200, 4200, 2960]
    ),

    heading3('P1 实验与安全（全部完成）'),
    makeTable(
      ['模块', '功能', '验证状态'],
      [
        ['数据统计工具', '图像数/类别/尺寸/平衡分析', '✅ 真实数据验证'],
        ['数据划分工具', '固定种子 + 分层抽样 + 泄漏检测', '✅ 真实数据验证'],
        ['指标提取工具', '从 results.csv 自动提取 + 对比', '✅ 真实数据验证'],
        ['30 条案例库', '6 大类场景 + 边界检查 + 自动运行', '✅ 30/30 PASS'],
        ['安全加固模块', '文件/SQL/授权/输入/会话 5 类安全', '✅ 代码审计'],
        ['三条演示脚本', '可复现 + 自动验证', '✅ 全部通过'],
      ],
      [2200, 4200, 2960]
    ),

    heading2('9.2 核心结论'),
    para('1. Agent 闭环基本成型：从任务理解到工具调用、质量判定、复核、报告，形成完整链路。'),
    para('2. 安全边界清晰可靠：范围隔离、图像门控、面积诚实表述、演示规则标注，均有代码实现和测试验证。'),
    para('3. 实验可复现性提升：数据划分、指标提取均已脚本化，不依赖人工抄写。'),
    para('4. 价值可量化验证：30 条案例库覆盖核心场景，Agent 相比固定流程的优势在于范围不污染、过程可追溯、异常有处理。'),

    heading2('9.3 未完成事项'),
    makeTable(
      ['事项', '优先级', '状态', '原因 / 阻塞项'],
      [
        ['LLM 真实集成', '高', '🔴 未接入', '缺少 API Key，当前为规则模式降级'],
        ['Word 报告 v2', '中', '🟡 待开发', '现有报告未包含范围/版本/方法备注'],
        ['多随机种子实验', '中', '🔴 未运行', '算力/时间限制（建议 ≥3 个种子）'],
        ['Copy-Paste 对比实验', '中', '🔴 未运行', '需实现增强脚本并训练'],
        ['cDCGAN 对比实验', '中', '🔴 未运行', '需完整训练 pipeline'],
        ['鲁棒性评估', '低', '🔴 未运行', '需扰动测试集 + 标注'],
        ['校准评估', '低', '🔴 未运行', '需标注数据 + 评估脚本'],
        ['工业误放行率验证', '低', '🔴 无法验证', '缺少合格品/不合格品业务标注'],
      ],
      [2400, 1200, 1200, 4560]
    ),

    heading2('9.4 未来工作建议'),
    bulletItem('接入真实 LLM（如 GPT-4 / 通义千问），验证 LLM 模式的工具调用准确率'),
    bulletItem('完成三组对比实验（基线 / Copy-Paste / cDCGAN+Copy-Paste），每种 3 个种子'),
    bulletItem('开发结构化 Word 报告 v2，包含范围、版本、方法备注、证据引用'),
    bulletItem('实现鲁棒性和校准评估，完善模型可靠性分析'),
    bulletItem('积累真实工业数据，验证系统在实际场景中的表现'),

    spacer(400),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 400, after: 200 },
      children: [new TextRun({
        text: '— 报告完 —',
        size: 24,
        color: '999999',
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({
        text: '本报告所有数据均来自真实运行结果，未完成项已明确标记。',
        size: 20,
        color: 'AAAAAA',
        font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
      })]
    }),
  ];
}

// ============================================================
// 主函数
// ============================================================

async function main() {
  console.log('正在生成实验报告...');

  const children = [
    ...buildCover(),
    ...buildChapter1(),
    ...buildChapter2(),
    ...buildChapter3(),
    ...buildChapter4(),
    ...buildChapter5(),
    ...buildChapter6(),
    ...buildChapter7(),
    ...buildChapter8(),
    ...buildChapter9(),
  ];

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: { ascii: 'Arial', hAnsi: 'Arial', eastAsia: 'Microsoft YaHei' },
            size: 22
          }
        }
      },
      paragraphStyles: [
        {
          id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: {
            size: 36, bold: true, color: '2E86AB',
            font: { ascii: 'Arial', hAnsi: 'Arial', eastAsia: 'Microsoft YaHei' }
          },
          paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0, keepNext: false, keepLines: false }
        },
        {
          id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: {
            size: 28, bold: true, color: '333333',
            font: { ascii: 'Arial', hAnsi: 'Arial', eastAsia: 'Microsoft YaHei' }
          },
          paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1, keepNext: false, keepLines: false }
        },
      ]
    },
    numbering: {
      config: [
        {
          reference: 'bullets',
          levels: [{
            level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
          }]
        },
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838 },  // A4
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({
              text: 'AI Agent 智能缺陷检测系统 · 实验报告 v2.0',
              size: 18, color: '999999',
              font: { ascii: 'Arial', eastAsia: 'Microsoft YaHei' }
            })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: '第 ', size: 18, color: '999999' }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '999999' }),
              new TextRun({ text: ' 页', size: 18, color: '999999' }),
            ]
          })]
        })
      },
      children
    }]
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUTPUT_PATH, buffer);
  console.log(`✅ 报告已生成：${OUTPUT_PATH}`);
  console.log(`   文件大小：${(buffer.length / 1024).toFixed(1)} KB`);
}

main().catch(err => {
  console.error('生成失败：', err);
  process.exit(1);
});
