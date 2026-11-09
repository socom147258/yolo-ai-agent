# -*- coding: utf-8 -*-
"""
GAN 管理器 - 高层封装，提供统一的训练、生成、评估接口

支持：
- DCGAN / cDCGAN 条件生成
- 预训练模型加载
- 单类/批量图像生成
- 生成质量评估（IS、FID 等）
- 与 YOLO 检测模型联动的数据增强
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import time
from typing import List, Dict, Optional, Tuple, Union
import json

from .dcgan import DCGAN, Generator


# NEU-DET 类别映射（中文名称）
DEFECT_NAMES_CN = {
    0: "裂纹 (crazing)",
    1: "夹杂 (inclusion)",
    2: "斑块 (patches)",
    3: "麻点 (pitted_surface)",
    4: "氧化皮 (rolled-in_scale)",
    5: "划痕 (scratches)",
}

DEFECT_NAMES_EN = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]


class GANManager:
    """
    GAN 模型管理器

    提供缺陷图像生成、模型训练、质量评估等功能。
    基于 DCGAN/cDCGAN 架构，支持 6 类工业表面缺陷的条件生成。
    """

    def __init__(self, config_path: str = None, device: str = None):
        """
        初始化 GAN 管理器

        Args:
            config_path: 配置文件路径，默认使用 configs/gan_config.yaml
            device: 计算设备，自动检测
        """
        # 加载配置
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "configs" / "gan_config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # 模型参数
        data_cfg = self.config.get("data", {})
        dcgan_cfg = self.config.get("dcgan", {})

        self.image_size = data_cfg.get("image_size", 64)
        self.channels = data_cfg.get("channels", 1)
        self.num_classes = data_cfg.get("num_classes", 6)
        self.class_names = data_cfg.get("class_names", DEFECT_NAMES_EN)
        self.latent_dim = dcgan_cfg.get("latent_dim", 100)
        self.g_feature_maps = dcgan_cfg.get("g_feature_maps", 64)
        self.d_feature_maps = dcgan_cfg.get("d_feature_maps", 64)

        # 模型
        self.model = None
        self.generator = None
        self.is_loaded = False

        # 统计信息
        self._stats = {
            "total_generated": 0,
            "last_generate_time": None,
        }

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        加载 GAN 检查点

        Args:
            checkpoint_path: 检查点文件路径 (.pth)

        Returns:
            加载信息字典
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

        # 创建模型
        self.model = DCGAN(
            latent_dim=self.latent_dim,
            g_feature_maps=self.g_feature_maps,
            d_feature_maps=self.d_feature_maps,
            channels=self.channels,
            num_classes=self.num_classes,
            image_size=self.image_size,
            conditional=True,
        )

        # 加载权重
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # 兼容不同的检查点格式
        if "generator_state_dict" in checkpoint:
            self.model.generator.load_state_dict(checkpoint["generator_state_dict"])
            if "discriminator_state_dict" in checkpoint:
                self.model.discriminator.load_state_dict(
                    checkpoint["discriminator_state_dict"]
                )
            epoch = checkpoint.get("epoch", "unknown")
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "unknown")
        else:
            # 尝试直接加载 generator
            try:
                self.model.generator.load_state_dict(checkpoint)
                epoch = "unknown (generator only)"
            except Exception:
                # 尝试另一种格式
                self._load_compatible(checkpoint)
                epoch = "compatible mode"

        self.model.to(self.device)
        self.model.eval()
        self.generator = self.model.generator
        self.is_loaded = True

        info = {
            "checkpoint_path": str(ckpt_path),
            "epoch": epoch,
            "device": str(self.device),
            "model_type": "DCGAN (conditional)",
            "image_size": self.image_size,
            "channels": self.channels,
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim,
        }

        return info

    def _load_compatible(self, checkpoint: dict):
        """兼容加载不同格式的检查点"""
        # 尝试从常见的 key 中提取 generator 权重
        gen_keys = [k for k in checkpoint.keys() if "generator" in k.lower()
                    and "state_dict" in k.lower()]
        if gen_keys:
            self.model.generator.load_state_dict(checkpoint[gen_keys[0]])
        else:
            # 尝试直接作为 generator 权重
            gen_state = {}
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    gen_state[k] = v
            if gen_state:
                self.model.generator.load_state_dict(gen_state, strict=False)

    def generate(self, num_samples: int = 1, class_idx: int = None,
                 class_name: str = None) -> Dict:
        """
        生成缺陷图像

        Args:
            num_samples: 生成数量
            class_idx: 类别索引 (0-5)，为 None 则随机生成
            class_name: 类别名称，与 class_idx 二选一

        Returns:
            包含生成图像和元信息的字典
        """
        if not self.is_loaded:
            raise RuntimeError("GAN 模型未加载，请先调用 load_checkpoint()")

        # 解析类别
        if class_name is not None:
            if class_name in self.class_names:
                class_idx = self.class_names.index(class_name)
            else:
                # 尝试模糊匹配
                for i, name in enumerate(self.class_names):
                    if class_name.lower() in name.lower():
                        class_idx = i
                        break
                if class_idx is None:
                    raise ValueError(
                        f"未知类别: {class_name}, 可用类别: {self.class_names}"
                    )

        start_time = time.time()

        with torch.no_grad():
            images, labels = self.generator.generate(
                num_samples=num_samples,
                class_idx=class_idx,
                device=self.device,
            )

        # 转换为 PIL 图像列表
        pil_images = []
        for i in range(num_samples):
            img_tensor = images[i].cpu()
            # 反归一化 [-1, 1] -> [0, 1]
            img_tensor = (img_tensor + 1) / 2.0
            img_tensor = img_tensor.clamp(0, 1)
            # 转 numpy
            img_np = img_tensor.squeeze().numpy()
            if self.channels == 1:
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='L')
            else:
                img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='RGB')
            pil_images.append(pil_img)

        elapsed = time.time() - start_time
        self._stats["total_generated"] += num_samples
        self._stats["last_generate_time"] = elapsed

        # 类别标签
        class_labels = labels.cpu().tolist() if labels is not None else [-1] * num_samples

        return {
            "images": pil_images,
            "class_indices": class_labels,
            "class_names": [self.class_names[idx] for idx in class_labels],
            "num_samples": num_samples,
            "generate_time": elapsed,
            "device": str(self.device),
        }

    def generate_grid(self, samples_per_class: int = 8,
                      save_path: str = None) -> Dict:
        """
        生成所有类别的网格图像

        Args:
            samples_per_class: 每个类别生成的样本数
            save_path: 保存路径，为 None 则不保存

        Returns:
            包含网格图像的字典
        """
        all_images = []
        all_labels = []

        for class_idx in range(self.num_classes):
            result = self.generate(
                num_samples=samples_per_class,
                class_idx=class_idx,
            )
            all_images.extend(result["images"])
            all_labels.extend(result["class_indices"])

        # 创建网格
        nrow = samples_per_class
        ncol = self.num_classes

        # 计算网格大小
        w, h = all_images[0].size
        grid_w = nrow * w
        grid_h = ncol * h

        if self.channels == 1:
            grid_img = Image.new('L', (grid_w, grid_h), color=255)
        else:
            grid_img = Image.new('RGB', (grid_w, grid_h), color='white')

        for class_idx in range(self.num_classes):
            for sample_idx in range(samples_per_class):
                img_idx = class_idx * samples_per_class + sample_idx
                x = sample_idx * w
                y = class_idx * h
                grid_img.paste(all_images[img_idx], (x, y))

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            grid_img.save(save_path)

        return {
            "grid_image": grid_img,
            "samples_per_class": samples_per_class,
            "num_classes": self.num_classes,
            "total_samples": samples_per_class * self.num_classes,
            "save_path": save_path,
        }

    def generate_and_save(self, num_per_class: int = 100,
                          output_dir: str = None) -> Dict:
        """
        批量生成并保存到指定目录（按类别分子目录）

        Args:
            num_per_class: 每个类别生成数量
            output_dir: 输出目录

        Returns:
            生成统计信息
        """
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            output_dir = base_dir / "data" / "processed" / "gan_generated"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_generated = 0
        class_stats = {}

        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)

            result = self.generate(
                num_samples=num_per_class,
                class_idx=class_idx,
            )

            saved = 0
            for i, img in enumerate(result["images"]):
                filename = f"{class_name}_{i:04d}.jpg"
                filepath = class_dir / filename
                img.save(filepath)
                saved += 1

            class_stats[class_name] = saved
            total_generated += saved

        return {
            "output_dir": str(output_dir),
            "total_generated": total_generated,
            "per_class": class_stats,
            "num_classes": self.num_classes,
        }

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "loaded": self.is_loaded,
            "device": str(self.device),
            "image_size": self.image_size,
            "channels": self.channels,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_names_cn": DEFECT_NAMES_CN,
            "latent_dim": self.latent_dim,
            "g_feature_maps": self.g_feature_maps,
            "d_feature_maps": self.d_feature_maps,
            "architecture": "DCGAN (conditional)",
            "total_generated": self._stats["total_generated"],
        }

    def evaluate_quality(self, real_images_dir: str = None,
                         num_samples: int = 100) -> Dict:
        """
        评估生成图像质量（简化版：基于统计特征的对比）

        实际项目中可使用 FID、IS 等指标，
        这里提供一个轻量级的像素级统计对比。

        Args:
            real_images_dir: 真实图像目录
            num_samples: 采样数量

        Returns:
            质量评估结果
        """
        # 生成样本
        gen_result = self.generate(num_samples=num_samples, class_idx=0)
        gen_images = gen_result["images"]

        # 计算生成图像的统计特征
        gen_pixels = []
        for img in gen_images:
            gen_pixels.append(np.array(img).flatten() / 255.0)
        gen_pixels = np.concatenate(gen_pixels)
        gen_mean = float(np.mean(gen_pixels))
        gen_std = float(np.std(gen_pixels))

        result = {
            "generated_mean": gen_mean,
            "generated_std": gen_std,
            "num_samples_evaluated": num_samples,
            "note": "基础像素统计，完整评估建议使用 FID/IS 指标",
        }

        # 如果有真实图像，计算对比
        if real_images_dir and Path(real_images_dir).exists():
            real_imgs = list(Path(real_images_dir).glob("*.jpg"))[
                        :num_samples]
            if real_imgs:
                real_pixels = []
                for img_path in real_imgs:
                    img = Image.open(img_path).convert('L')
                    real_pixels.append(np.array(img).flatten() / 255.0)
                real_pixels = np.concatenate(real_pixels)
                real_mean = float(np.mean(real_pixels))
                real_std = float(np.std(real_pixels))

                result.update({
                    "real_mean": real_mean,
                    "real_std": real_std,
                    "mean_diff": abs(gen_mean - real_mean),
                    "std_diff": abs(gen_std - real_std),
                    "real_sample_count": len(real_imgs),
                })

        return result


# ============================================================
# 便捷函数
# ============================================================

def create_gan_manager(checkpoint_path: str = None,
                       config_path: str = None,
                       device: str = None) -> GANManager:
    """
    便捷创建并加载 GAN 管理器

    Args:
        checkpoint_path: 检查点路径，默认加载 best_model.pth
        config_path: 配置文件路径
        device: 计算设备

    Returns:
        已加载的 GANManager 实例
    """
    manager = GANManager(config_path=config_path, device=device)

    if checkpoint_path is None:
        # 默认检查点路径
        base_dir = Path(__file__).parent.parent.parent
        checkpoint_path = base_dir / "outputs" / "checkpoints" / "dcgan" / "best_model.pth"

    if Path(checkpoint_path).exists():
        manager.load_checkpoint(str(checkpoint_path))

    return manager


def list_available_checkpoints(checkpoints_dir: str = None) -> List[Dict]:
    """
    列出可用的 GAN 检查点

    Args:
        checkpoints_dir: 检查点目录

    Returns:
        检查点信息列表
    """
    if checkpoints_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        checkpoints_dir = base_dir / "outputs" / "checkpoints" / "dcgan"

    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return []

    checkpoints = []
    for f in sorted(checkpoints_dir.glob("*.pth")):
        size_mb = f.stat().st_size / (1024 * 1024)
        checkpoints.append({
            "name": f.name,
            "path": str(f),
            "size_mb": round(size_mb, 2),
        })

    return checkpoints
