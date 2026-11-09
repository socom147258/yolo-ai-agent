# -*- coding: utf-8 -*-
"""
数据划分工具 - 固定随机种子，生成可复现的 train/val/test 清单

特点：
- 固定随机种子（默认 42），结果可复现
- 分层抽样，保持类别比例
- 生成清单文件（train.txt / val.txt / test.txt）
- 保存划分配置（种子、比例、时间戳）

用法：
    python tools/split_dataset.py --data_dir data/raw/NEU-CLS --output_dir data/splits/split_v1
    python tools/split_dataset.py --data_dir data/raw/NEU-CLS --seed 123 --train_ratio 0.7 --val_ratio 0.15
"""
import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from datetime import datetime


CLASS_NAMES = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']


def collect_images(data_dir):
    """收集所有图像，按类别分组（自动检测格式）"""
    data_dir = Path(data_dir)
    images_by_class = defaultdict(list)

    # 检测是否有类别子文件夹
    has_class_dirs = any((data_dir / cls).is_dir() for cls in CLASS_NAMES)

    if has_class_dirs:
        # 分类文件夹格式
        for cls_name in CLASS_NAMES:
            cls_dir = data_dir / cls_name
            if not cls_dir.exists():
                continue

            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    images_by_class[cls_name].append(f)
    else:
        # 平铺格式（文件名前缀为类别）
        for f in sorted(data_dir.iterdir()):
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            name = f.stem
            cls_name = name.rsplit('_', 1)[0] if '_' in name else 'unknown'
            if cls_name in CLASS_NAMES:
                images_by_class[cls_name].append(f)

    return images_by_class


def split_dataset(images_by_class, seed=42, train_ratio=0.7, val_ratio=0.15):
    """
    分层抽样划分数据集

    返回：{'train': [(cls, path), ...], 'val': [...], 'test': [...]}
    """
    random.seed(seed)

    splits = {'train': [], 'val': [], 'test': []}

    for cls_name, images in sorted(images_by_class.items()):
        # 打乱
        shuffled = images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train:n_train + n_val]
        test_imgs = shuffled[n_train + n_val:]

        for img in train_imgs:
            splits['train'].append((cls_name, img))
        for img in val_imgs:
            splits['val'].append((cls_name, img))
        for img in test_imgs:
            splits['test'].append((cls_name, img))

    # 排序，保证输出稳定
    for key in splits:
        splits[key].sort(key=lambda x: (x[0], x[1].name))

    return splits


def check_leakage(splits):
    """检查数据泄漏（同一张图出现在多个集合）"""
    all_paths = []
    for split_name, items in splits.items():
        for cls, path in items:
            all_paths.append((split_name, path.name))

    name_to_splits = defaultdict(set)
    for split_name, filename in all_paths:
        name_to_splits[filename].add(split_name)

    leaks = {name: splits for name, splits in name_to_splits.items() if len(splits) > 1}
    return leaks


def save_splits(splits, output_dir, seed, train_ratio, val_ratio, source_dir):
    """保存划分结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存清单
    for split_name, items in splits.items():
        list_file = output_dir / f"{split_name}.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for cls, path in items:
                f.write(f"{cls}/{path.name}\n")

    # 保存配置
    config = {
        'split_id': output_dir.name,
        'created_at': datetime.now().isoformat(),
        'source_dir': str(source_dir),
        'random_seed': seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': round(1.0 - train_ratio - val_ratio, 4),
        'class_names': CLASS_NAMES,
        'split_counts': {
            split_name: {
                'total': len(items),
                'by_class': _count_by_class(items),
            }
            for split_name, items in splits.items()
        },
    }

    with open(output_dir / 'split_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return config


def _count_by_class(items):
    counts = defaultdict(int)
    for cls, _ in items:
        counts[cls] += 1
    return dict(sorted(counts.items()))


def print_report(config, leaks):
    """打印划分报告"""
    print("=" * 70)
    print("📋 数据划分报告")
    print("=" * 70)
    print(f"随机种子: {config['random_seed']}")
    print(f"划分比例: train={config['train_ratio']}, "
          f"val={config['val_ratio']}, test={config['test_ratio']}")
    print()

    print(f"{'集合':<10} {'总数':>8}", end='')
    for cls in config['class_names']:
        print(f" {cls[:6]:>8}", end='')
    print()
    print("-" * 70)

    for split_name, info in config['split_counts'].items():
        print(f"{split_name:<10} {info['total']:>8}", end='')
        for cls in config['class_names']:
            print(f" {info['by_class'].get(cls, 0):>8}", end='')
        print()

    print()
    if leaks:
        print(f"❌ 数据泄漏检测：发现 {len(leaks)} 个泄漏样本！")
        for name, split_set in list(leaks.items())[:5]:
            print(f"   {name}: {', '.join(split_set)}")
    else:
        print("✅ 数据泄漏检测：无泄漏")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='数据集划分（固定种子，可复现）')
    parser.add_argument('--data_dir', type=str, required=True, help='原始数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    args = parser.parse_args()

    print(f"收集数据: {args.data_dir}")
    images_by_class = collect_images(args.data_dir)

    total = sum(len(v) for v in images_by_class.values())
    print(f"共 {total} 张图像，{len(images_by_class)} 类")

    print(f"\n划分数据集 (seed={args.seed})...")
    splits = split_dataset(images_by_class, args.seed, args.train_ratio, args.val_ratio)

    print("检查数据泄漏...")
    leaks = check_leakage(splits)

    print(f"\n保存到: {args.output_dir}")
    config = save_splits(splits, args.output_dir, args.seed,
                         args.train_ratio, args.val_ratio, args.data_dir)

    print_report(config, leaks)

    if leaks:
        print("\n⚠️ 警告：存在数据泄漏，请检查！")
        exit(1)
    else:
        print("\n✅ 划分完成，结果可复现")


if __name__ == '__main__':
    main()
