# -*- coding: utf-8 -*-
"""
评估指标自动提取工具
从 YOLOv8 训练/评估结果中自动提取指标，生成结构化报告

支持：
- 从 results.csv 提取训练曲线指标
- 从 validation 结果提取 mAP、Precision、Recall
- 自动计算绝对变化（百分点）和相对提升
- 多实验对比表生成

用法：
    python tools/extract_metrics.py --exp_dir outputs/baseline/yolov8s_baseline-2
    python tools/extract_metrics.py --compare outputs/baseline outputs/gan_mixed --output metrics_report.json
"""
import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def find_results_csv(exp_dir):
    """查找 results.csv"""
    exp_dir = Path(exp_dir)

    # 常见位置
    candidates = [
        exp_dir / 'results.csv',
        exp_dir / 'training_results.csv',
    ]

    for c in candidates:
        if c.exists():
            return c

    # 递归查找
    for f in exp_dir.rglob('results.csv'):
        return f

    return None


def parse_results_csv(csv_path):
    """解析 YOLO results.csv"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 清理列名中的空格
            clean_row = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(clean_row)

    if not rows:
        return None

    # 提取最后一行（最终指标）
    final = rows[-1]

    # 分类提取
    metrics = {
        'epoch': int(float(final.get('epoch', 0))),
        'train': {
            'box_loss': float(final.get('train/box_loss', 0)),
            'cls_loss': float(final.get('train/cls_loss', 0)),
            'dfl_loss': float(final.get('train/dfl_loss', 0)),
        },
        'val': {
            'box_loss': float(final.get('val/box_loss', 0)),
            'cls_loss': float(final.get('val/cls_loss', 0)),
            'dfl_loss': float(final.get('val/dfl_loss', 0)),
        },
        'metrics': {
            'precision': float(final.get('metrics/precision(B)', 0)),
            'recall': float(final.get('metrics/recall(B)', 0)),
            'mAP50': float(final.get('metrics/mAP50(B)', 0)),
            'mAP50_95': float(final.get('metrics/mAP50-95(B)', 0)),
        },
        'lr': {
            'pg0': float(final.get('lr/pg0', 0)),
            'pg1': float(final.get('lr/pg1', 0)),
            'pg2': float(final.get('lr/pg2', 0)),
        },
        'total_epochs': len(rows),
    }

    return metrics


def find_best_metrics(exp_dir):
    """查找最佳权重的指标"""
    exp_dir = Path(exp_dir)

    # 查找 args.yaml 获取配置
    args_file = exp_dir / 'args.yaml'
    config = {}
    if args_file.exists():
        with open(args_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, val = line.split(':', 1)
                    config[key.strip()] = val.strip()

    return config


def extract_experiment(exp_dir):
    """提取单个实验的完整指标"""
    exp_dir = Path(exp_dir)

    result = {
        'exp_name': exp_dir.name,
        'exp_path': str(exp_dir),
        'exists': exp_dir.exists(),
    }

    if not exp_dir.exists():
        return result

    # results.csv
    csv_path = find_results_csv(exp_dir)
    if csv_path:
        metrics = parse_results_csv(csv_path)
        if metrics:
            result['final_metrics'] = metrics
            result['has_results'] = True
        else:
            result['has_results'] = False
    else:
        result['has_results'] = False

    # 配置
    config = find_best_metrics(exp_dir)
    if config:
        result['config'] = {
            'model': config.get('model', 'unknown'),
            'data': config.get('data', 'unknown'),
            'epochs': config.get('epochs', 'unknown'),
            'imgsz': config.get('imgsz', 'unknown'),
            'batch': config.get('batch', 'unknown'),
            'optimizer': config.get('optimizer', 'unknown'),
            'lr0': config.get('lr0', 'unknown'),
            'seed': config.get('seed', 'unknown'),
        }

    # 权重文件
    weights_dir = exp_dir / 'weights'
    if weights_dir.exists():
        weights = [f.name for f in weights_dir.iterdir() if f.suffix == '.pt']
        result['weights'] = weights

    return result


def compare_experiments(exp_dirs):
    """对比多个实验的指标"""
    results = []
    for exp_dir in exp_dirs:
        exp = extract_experiment(exp_dir)
        results.append(exp)

    # 生成对比表
    comparison = {
        'experiments': results,
        'comparison_table': {},
    }

    # 找基线（第一个）
    baseline = results[0] if results else None
    if not baseline or not baseline.get('has_results'):
        return comparison

    baseline_map = baseline['final_metrics']['metrics']

    metric_keys = ['precision', 'recall', 'mAP50', 'mAP50_95']
    table = {}

    for exp in results:
        exp_name = exp['exp_name']
        table[exp_name] = {}

        if not exp.get('has_results'):
            continue

        exp_metrics = exp['final_metrics']['metrics']

        for key in metric_keys:
            val = exp_metrics.get(key, 0)
            base_val = baseline_map.get(key, 0)
            abs_change = val - base_val  # 绝对变化（个百分点）
            rel_change = (abs_change / base_val * 100) if base_val > 0 else 0  # 相对变化（%）

            table[exp_name][key] = {
                'value': round(val, 4),
                'abs_change_pp': round(abs_change * 100, 2),  # 转换为百分点
                'rel_change_pct': round(rel_change, 2),
            }

    comparison['comparison_table'] = table
    return comparison


def print_single_report(exp):
    """打印单个实验报告"""
    print("=" * 70)
    print(f"📊 实验指标: {exp['exp_name']}")
    print("=" * 70)
    print(f"路径: {exp['exp_path']}")

    if 'config' in exp:
        c = exp['config']
        print(f"\n配置:")
        print(f"  模型: {c.get('model', '?')}")
        print(f"  数据: {c.get('data', '?')}")
        print(f"  Epochs: {c.get('epochs', '?')}")
        print(f"  imgsz: {c.get('imgsz', '?')}")
        print(f"  batch: {c.get('batch', '?')}")
        print(f"  optimizer: {c.get('optimizer', '?')}")
        print(f"  lr0: {c.get('lr0', '?')}")
        print(f"  seed: {c.get('seed', '?')}")

    if exp.get('has_results'):
        m = exp['final_metrics']['metrics']
        t = exp['final_metrics']['train']
        v = exp['final_metrics']['val']

        print(f"\n最终指标 (epoch {exp['final_metrics']['epoch']}/{exp['final_metrics']['total_epochs']}):")
        print(f"  Precision:  {m['precision']:.4f}")
        print(f"  Recall:     {m['recall']:.4f}")
        print(f"  mAP@50:     {m['mAP50']:.4f}  ({m['mAP50']*100:.2f}%)")
        print(f"  mAP@50-95:  {m['mAP50_95']:.4f}  ({m['mAP50_95']*100:.2f}%)")

        print(f"\n训练损失:")
        print(f"  box_loss:   {t['box_loss']:.4f}")
        print(f"  cls_loss:   {t['cls_loss']:.4f}")
        print(f"  dfl_loss:   {t['dfl_loss']:.4f}")

        print(f"\n验证损失:")
        print(f"  box_loss:   {v['box_loss']:.4f}")
        print(f"  cls_loss:   {v['cls_loss']:.4f}")
        print(f"  dfl_loss:   {v['dfl_loss']:.4f}")
    else:
        print("\n⚠️ 未找到 results.csv，无法提取指标")

    if 'weights' in exp:
        print(f"\n权重文件: {', '.join(exp['weights'])}")

    print("=" * 70)


def print_comparison_report(comparison):
    """打印对比报告"""
    results = comparison['experiments']
    table = comparison['comparison_table']

    print("=" * 70)
    print("📊 实验对比报告")
    print("=" * 70)
    print(f"基线实验: {results[0]['exp_name'] if results else 'N/A'}")
    print(f"参与对比: {len(results)} 个实验")
    print()

    metric_keys = ['precision', 'recall', 'mAP50', 'mAP50_95']
    metric_labels = {
        'precision': 'Precision',
        'recall': 'Recall',
        'mAP50': 'mAP@50',
        'mAP50_95': 'mAP@50-95',
    }

    # 表格头
    header = f"{'实验':<25}"
    for key in metric_keys:
        header += f" {metric_labels[key]:>14}"
    print(header)
    print("-" * (25 + 14 * len(metric_keys)))

    for exp in results:
        name = exp['exp_name'][:24]
        line = f"{name:<25}"

        if name not in table or not table[name]:
            for _ in metric_keys:
                line += f" {'N/A':>14}"
        else:
            for key in metric_keys:
                info = table[name].get(key, {})
                val = info.get('value', 0)
                abs_pp = info.get('abs_change_pp', 0)

                if abs_pp == 0:
                    line += f" {val*100:>10.2f}%"
                else:
                    sign = '+' if abs_pp > 0 else ''
                    line += f" {val*100:>6.2f}%({sign}{abs_pp:.1f})"
        print(line)

    print()
    print("说明: 括号内为相对基线的绝对变化（百分点）")
    print("     正值用 + 号标记，负值用 - 号标记")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='YOLO 评估指标自动提取')
    parser.add_argument('--exp_dir', type=str, default='', help='单个实验目录')
    parser.add_argument('--compare', nargs='+', default=[], help='多个实验目录对比')
    parser.add_argument('--output', type=str, default='', help='输出 JSON 报告路径')
    args = parser.parse_args()

    if args.compare:
        print(f"对比 {len(args.compare)} 个实验...")
        comparison = compare_experiments(args.compare)
        print_comparison_report(comparison)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            print(f"\n📁 对比报告已保存到: {out_path}")

    elif args.exp_dir:
        exp = extract_experiment(args.exp_dir)
        print_single_report(exp)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(exp, f, ensure_ascii=False, indent=2)
            print(f"\n📁 指标报告已保存到: {out_path}")
    else:
        print("请指定 --exp_dir 或 --compare")
        parser.print_help()


if __name__ == '__main__':
    main()
