# -*- coding: utf-8 -*-
"""
数据集统计分析工具
统计：图像数量、标注实例数、尺寸分布、类别分布

用法：
    python tools/analyze_dataset.py --data_dir data/raw/NEU-CLS
    python tools/analyze_dataset.py --data_dir data/processed/yolo_format --yolo
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image


CLASS_NAMES = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']

CLASS_NAMES_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化皮',
    'scratches': '划痕',
}


def analyze_flat_folder(data_dir):
    """分析平铺文件夹（文件名前缀为类别名，如 crazing_0001.jpg）"""
    data_dir = Path(data_dir)
    stats = {
        'dataset_type': 'flat_folder_by_prefix',
        'root_dir': str(data_dir),
        'classes': {},
        'total_images': 0,
        'size_distribution': {
            'unique_sizes': {},
        },
    }

    all_images = sorted([f for f in data_dir.iterdir()
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])

    # 按文件名前缀分类
    images_by_class = defaultdict(list)
    for img_path in all_images:
        # 提取类别名（下划线前的部分）
        name = img_path.stem
        cls_name = name.rsplit('_', 1)[0] if '_' in name else 'unknown'
        images_by_class[cls_name].append(img_path)

    sizes_global = defaultdict(int)

    for cls_name in sorted(images_by_class.keys()):
        images = images_by_class[cls_name]
        sizes = defaultdict(int)

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    sizes[(w, h)] += 1
                    sizes_global[(w, h)] += 1
            except Exception as e:
                print(f"  ⚠️ 无法读取 {img_path.name}: {e}")

        stats['classes'][cls_name] = {
            'image_count': len(images),
            'cn_name': CLASS_NAMES_CN.get(cls_name, cls_name),
            'size_count': len(sizes),
        }
        stats['total_images'] += len(images)

    stats['size_distribution']['unique_sizes'] = {
        f"{w}x{h}": c for (w, h), c in sizes_global.items()
    }
    stats['size_distribution']['unique_size_count'] = len(sizes_global)

    # 计算尺寸汇总
    if sizes_global:
        widths = []
        heights = []
        for (w, h), count in sizes_global.items():
            widths.extend([w] * count)
            heights.extend([h] * count)
        stats['size_distribution']['summary'] = {
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': round(sum(widths) / len(widths), 1),
            'min_height': min(heights),
            'max_height': max(heights),
            'avg_height': round(sum(heights) / len(heights), 1),
        }

    return stats


def analyze_classification_folder(data_dir):
    """分析分类文件夹结构（每类一个子文件夹）"""
    data_dir = Path(data_dir)
    stats = {
        'dataset_type': 'classification_folder',
        'root_dir': str(data_dir),
        'classes': {},
        'total_images': 0,
        'size_distribution': {
            'widths': [],
            'heights': [],
            'unique_sizes': {},
        },
    }

    for cls_name in CLASS_NAMES:
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue

        images = sorted([f for f in cls_dir.iterdir()
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])

        sizes = defaultdict(int)
        widths = []
        heights = []

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    sizes[(w, h)] += 1
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"  ⚠️ 无法读取 {img_path.name}: {e}")

        stats['classes'][cls_name] = {
            'image_count': len(images),
            'cn_name': CLASS_NAMES_CN.get(cls_name, cls_name),
            'sizes': {f"{w}x{h}": c for (w, h), c in sizes.items()},
            'min_width': min(widths) if widths else 0,
            'max_width': max(widths) if widths else 0,
            'avg_width': round(sum(widths) / len(widths), 1) if widths else 0,
            'min_height': min(heights) if heights else 0,
            'max_height': max(heights) if heights else 0,
            'avg_height': round(sum(heights) / len(heights), 1) if heights else 0,
        }
        stats['total_images'] += len(images)
        stats['size_distribution']['widths'].extend(widths)
        stats['size_distribution']['heights'].extend(heights)
        for size, count in sizes.items():
            key = f"{size[0]}x{size[1]}"
            stats['size_distribution']['unique_sizes'][key] = \
                stats['size_distribution']['unique_sizes'].get(key, 0) + count

    # 汇总尺寸
    widths = stats['size_distribution']['widths']
    heights = stats['size_distribution']['heights']
    if widths:
        stats['size_distribution']['summary'] = {
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': round(sum(widths) / len(widths), 1),
            'min_height': min(heights),
            'max_height': max(heights),
            'avg_height': round(sum(heights) / len(heights), 1),
            'unique_size_count': len(stats['size_distribution']['unique_sizes']),
        }
    # 移除原始列表，减少输出
    del stats['size_distribution']['widths']
    del stats['size_distribution']['heights']

    return stats


def analyze_yolo_format(data_dir):
    """分析 YOLO 格式数据集（图像+标签）"""
    data_dir = Path(data_dir)
    stats = {
        'dataset_type': 'yolo_format',
        'root_dir': str(data_dir),
        'splits': {},
        'total_images': 0,
        'total_annotations': 0,
    }

    for split in ['train', 'val', 'test']:
        img_dir = data_dir / split / 'images'
        lbl_dir = data_dir / split / 'labels'

        if not img_dir.exists():
            continue

        images = sorted([f for f in img_dir.iterdir()
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

        annotations = 0
        class_counts = defaultdict(int)
        size_dist = defaultdict(int)

        for img_path in images:
            # 统计标注
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                with open(lbl_path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                annotations += len(lines)
                for line in lines:
                    cls_id = int(line.split()[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                    class_counts[cls_name] += 1

            # 统计尺寸
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    size_dist[f"{w}x{h}"] += 1
            except:
                pass

        split_stats = {
            'image_count': len(images),
            'annotation_count': annotations,
            'class_distribution': dict(class_counts),
            'size_distribution': dict(size_dist),
        }
        stats['splits'][split] = split_stats
        stats['total_images'] += len(images)
        stats['total_annotations'] += annotations

    return stats


def print_stats_report(stats):
    """打印统计报告"""
    print("=" * 70)
    print("📊 数据集统计报告")
    print("=" * 70)
    print(f"数据集类型: {stats['dataset_type']}")
    print(f"根目录: {stats['root_dir']}")
    print(f"总图像数: {stats['total_images']}")

    if stats['dataset_type'] in ('classification_folder', 'flat_folder_by_prefix'):
        print()
        print("类别分布:")
        print(f"  {'类别':<20} {'中文名':<10} {'图像数':>8} {'占比':>8}")
        print("  " + "-" * 50)
        total = stats['total_images']
        for cls_name, cls_info in sorted(stats['classes'].items()):
            pct = cls_info['image_count'] / total * 100 if total else 0
            print(f"  {cls_name:<20} {cls_info['cn_name']:<10} {cls_info['image_count']:>8} {pct:>7.1f}%")

        if stats.get('size_distribution', {}).get('summary'):
            s = stats['size_distribution']['summary']
            print(f"\n尺寸统计:")
            print(f"  宽度范围: {s['min_width']} - {s['max_width']} (平均 {s['avg_width']})")
            print(f"  高度范围: {s['min_height']} - {s['max_height']} (平均 {s['avg_height']})")
            print(f"  唯一尺寸数: {stats['size_distribution']['unique_size_count']}")

            # 不平衡度分析
            counts = [c['image_count'] for c in stats['classes'].values()]
            if counts:
                max_c, min_c = max(counts), min(counts)
                ratio = max_c / min_c if min_c > 0 else float('inf')
                print(f"\n类别平衡分析:")
                print(f"  最多/最少比: {ratio:.1f}:1")
                if ratio > 2:
                    print(f"  ⚠️ 存在类别不平衡（比例 > 2:1）")
                else:
                    print(f"  ✅ 类别相对平衡（比例 ≤ 2:1）")
                    print(f"  ℹ️ 注意：每类数量相同不代表检测难度相同，"
                          f"困难类别（如小目标、低对比度）仍可能表现较差")

    elif stats['dataset_type'] == 'yolo_format':
        print(f"总标注数: {stats['total_annotations']}")
        print()
        for split, split_stats in stats['splits'].items():
            print(f"【{split.upper()} 集】")
            print(f"  图像数: {split_stats['image_count']}")
            print(f"  标注数: {split_stats['annotation_count']}")
            if split_stats['class_distribution']:
                print(f"  类别分布:")
                for cls, cnt in sorted(split_stats['class_distribution'].items()):
                    cn = CLASS_NAMES_CN.get(cls, cls)
                    pct = cnt / split_stats['annotation_count'] * 100 if split_stats['annotation_count'] else 0
                    print(f"    {cls:<20} ({cn:<6}): {cnt:>6} ({pct:>5.1f}%)")
            print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='数据集统计分析')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--yolo', action='store_true', help='YOLO 格式（有 train/val/test + images/labels）')
    parser.add_argument('--flat', action='store_true', help='平铺格式（文件名前缀为类别）')
    parser.add_argument('--output', type=str, default='', help='输出 JSON 文件路径')
    args = parser.parse_args()

    if args.yolo:
        stats = analyze_yolo_format(args.data_dir)
    elif args.flat:
        stats = analyze_flat_folder(args.data_dir)
    else:
        # 自动检测
        data_dir = Path(args.data_dir)
        # 检查是否有类别子文件夹
        has_class_dirs = any((data_dir / cls).is_dir() for cls in CLASS_NAMES)
        if has_class_dirs:
            stats = analyze_classification_folder(args.data_dir)
        else:
            stats = analyze_flat_folder(args.data_dir)

    print_stats_report(stats)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n📁 详细统计已保存到: {out_path}")


if __name__ == '__main__':
    main()
