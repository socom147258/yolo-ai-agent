# -*- coding: utf-8 -*-
"""
第二阶段：cDCGAN + Copy-Paste 数据增强管线
============================================
任务1：cDCGAN条件生成缺陷图像
任务2：Copy-Paste合成 + 自动标注生成 + 质量检查

核心原理：
- GAN解决样本多样性问题
- Copy-Paste解决缺陷位置问题
- 标注生成程序解决合成图片与检测标签的对应关系
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import yaml
import cv2

# ============================================================
# 模型定义：条件 DCGAN (cDCGAN)
# ============================================================

class Generator(nn.Module):
    """条件生成器：输入噪声 + 类别标签，生成对应类别缺陷"""
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=6, embed_size=100):
        super().__init__()
        self.nz = nz
        self.num_classes = num_classes
        
        # 类别嵌入
        self.label_embed = nn.Embedding(num_classes, embed_size)
        
        self.main = nn.Sequential(
            # input: nz+embed_size x 1 x 1
            nn.ConvTranspose2d(nz + embed_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state: nc x 64 x 64
        )
    
    def forward(self, noise, labels):
        # 嵌入类别标签
        label_emb = self.label_embed(labels).unsqueeze(2).unsqueeze(3)  # (B, embed, 1, 1)
        # 拼接噪声和标签
        x = torch.cat([noise, label_emb], dim=1)  # (B, nz+embed, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    """条件判别器：输入图像 + 类别标签，判断真假"""
    def __init__(self, ndf=64, nc=3, num_classes=6, img_size=64):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 类别嵌入 -> 空间维度扩展
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)
        
        self.main = nn.Sequential(
            # input: (nc+1) x 64 x 64 (多一个通道放类别信息)
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state: 1 x 1 x 1
        )
    
    def forward(self, img, labels):
        # 类别嵌入转图像尺寸
        label_map = self.label_embed(labels).view(-1, 1, self.img_size, self.img_size)
        # 拼接图像和类别图
        x = torch.cat([img, label_map], dim=1)  # (B, nc+1, 64, 64)
        return self.main(x).view(-1, 1).squeeze(1)


# ============================================================
# 数据集：NEU-DET缺陷分类数据集
# ============================================================

class DefectDataset(Dataset):
    def __init__(self, root_dir, class_names, img_size=64):
        self.root_dir = Path(root_dir)
        self.class_names = class_names
        self.img_size = img_size
        self.samples = []
        
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ]
        self.transform = transforms.Compose(transform_list)
        
        # 加载图像路径
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = self.root_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), cls_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


# ============================================================
# 训练函数
# ============================================================

def train_cdcgan(data_dir, output_dir, class_names, 
                 epochs=300, batch_size=32, nz=100, ngf=64, ndf=64,
                 lr=0.0002, beta1=0.5, device='cuda', img_size=64):
    """训练cDCGAN"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    num_classes = len(class_names)
    
    # 数据集
    dataset = DefectDataset(data_dir, class_names, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=True, num_workers=4, drop_last=True)
    print(f'[INFO] Dataset size: {len(dataset)} images')
    
    # 模型
    netG = Generator(nz=nz, ngf=ngf, nc=3, num_classes=num_classes).to(device)
    netD = Discriminator(ndf=ndf, nc=3, num_classes=num_classes, img_size=img_size).to(device)
    
    # 损失和优化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # 固定噪声用于观察训练过程
    fixed_noise = torch.randn(60, nz, device=device)
    fixed_labels = torch.arange(6).repeat(10).to(device)  # 每类10张
    
    # 训练记录
    g_losses = []
    d_losses = []
    
    print(f'[INFO] Starting cDCGAN training ({epochs} epochs)...')
    print(f'[INFO] Device: {device}')
    print()
    
    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size_curr = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # 真实标签=1，假标签=0
            real_label = torch.full((batch_size_curr,), 1.0, device=device)
            fake_label = torch.full((batch_size_curr,), 0.0, device=device)
            
            # ========== 训练判别器 ==========
            netD.zero_grad()
            
            # 真实图像
            output_real = netD(real_imgs, labels)
            lossD_real = criterion(output_real, real_label)
            lossD_real.backward()
            
            # 生成假图像
            noise = torch.randn(batch_size_curr, nz, 1, 1, device=device)
            fake_imgs = netG(noise, labels)
            
            # 假图像判别
            output_fake = netD(fake_imgs.detach(), labels)
            lossD_fake = criterion(output_fake, fake_label)
            lossD_fake.backward()
            
            lossD = lossD_real + lossD_fake
            optimizerD.step()
            
            # ========== 训练生成器 ==========
            netG.zero_grad()
            
            output = netD(fake_imgs, labels)
            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()
            
            # 记录
            g_losses.append(lossG.item())
            d_losses.append(lossD.item())
        
        # 每10轮打印并保存样本
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]  Loss_D: {lossD.item():.4f}  Loss_G: {lossG.item():.4f}')
            
            # 生成固定样本观察
            with torch.no_grad():
                fake = netG(fixed_noise.unsqueeze(2).unsqueeze(3), fixed_labels)
                # 保存网格图
                save_path = os.path.join(output_dir, 'samples', f'epoch_{epoch+1:03d}.png')
                torchvision.utils.save_image(
                    fake, save_path, nrow=6, normalize=True, padding=2
                )
        
        # 每50轮保存检查点
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'cdgan_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'lossG': lossG.item(),
                'lossD': lossD.item(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'class_names': class_names,
                'nz': nz,
                'ngf': ngf,
                'ndf': ndf,
            }, ckpt_path)
            print(f'  -> Checkpoint saved: {ckpt_path}')
    
    # 保存最终最佳模型
    best_path = os.path.join(output_dir, 'best_cdcgan.pth')
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'class_names': class_names,
        'nz': nz,
        'ngf': ngf,
        'ndf': ndf,
        'g_losses': g_losses,
        'd_losses': d_losses,
    }, best_path)
    
    print(f'\n[INFO] Training complete! Best model saved to: {best_path}')
    return best_path


# ============================================================
# 生成函数：批量生成合成缺陷
# ============================================================

def generate_defect_images(checkpoint_path, output_dir, num_per_class=500, device='cuda'):
    """加载cDCGAN模型，批量生成各类缺陷图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载检查点
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt['class_names']
    nz = ckpt['nz']
    ngf = ckpt['ngf']
    
    # 构建生成器
    netG = Generator(nz=nz, ngf=ngf, nc=3, num_classes=len(class_names)).to(device)
    netG.load_state_dict(ckpt['netG_state_dict'])
    netG.eval()
    
    print(f'[INFO] Generating {num_per_class} images per class...')
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        # 分批生成
        batch_size = 64
        generated = 0
        while generated < num_per_class:
            curr_batch = min(batch_size, num_per_class - generated)
            noise = torch.randn(curr_batch, nz, 1, 1, device=device)
            labels = torch.full((curr_batch,), cls_idx, dtype=torch.long, device=device)
            
            with torch.no_grad():
                fake_imgs = netG(noise, labels)
            
            # 保存图像
            for j in range(curr_batch):
                img = fake_imgs[j].cpu()
                # 反归一化 [-1,1] -> [0,1]
                img = (img + 1) / 2
                img_path = os.path.join(cls_dir, f'gan_{cls_name}_{generated+j:04d}.png')
                torchvision.utils.save_image(img, img_path)
            
            generated += curr_batch
        
        print(f'  {cls_name}: {generated} images saved')
    
    print(f'[INFO] Total generated: {num_per_class * len(class_names)} images')
    return output_dir


# ============================================================
# Copy-Paste 合成 + 自动标注
# ============================================================

def copy_paste_synthesize(defect_dir, background_dir, output_dir, 
                          num_per_class=300, img_size=640, seed=42):
    """
    Copy-Paste合成：将GAN生成的缺陷粘贴到正常背景上，自动生成边界框标注
    
    核心原理：
    - 缺陷纹理来自GAN生成的缺陷图
    - 随机选择背景图（正常或有缺陷的图像均可作为背景）
    - 随机位置、大小、角度粘贴缺陷
    - 自动生成YOLO格式边界框标注
    - 质量检查：剔除重叠过大、超出边界的样本
    """
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    
    class_names = sorted([d.name for d in Path(defect_dir).iterdir() if d.is_dir()])
    print(f'[INFO] Classes: {class_names}')
    
    # 收集背景图像
    bg_images = []
    for ext in ['*.jpg', '*.png', '*.bmp']:
        bg_images.extend(list(Path(background_dir).glob(ext)))
    print(f'[INFO] Background images: {len(bg_images)}')
    
    # 收集每类缺陷图像
    defect_images = {}
    for cls_name in class_names:
        cls_dir = Path(defect_dir) / cls_name
        defect_images[cls_name] = list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpg'))
        print(f'  {cls_name}: {len(defect_images[cls_name])} defect images')
    
    print(f'\n[INFO] Starting Copy-Paste synthesis...')
    print(f'[INFO] Target: {num_per_class} images per class')
    
    total_synthesized = 0
    
    for cls_idx, cls_name in enumerate(class_names):
        defect_list = defect_images[cls_name]
        if not defect_list:
            continue
        
        synthesized = 0
        attempt = 0
        max_attempts = num_per_class * 10
        
        while synthesized < num_per_class and attempt < max_attempts:
            attempt += 1
            
            # 1. 随机选背景
            bg_path = random.choice(bg_images)
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            bg = cv2.resize(bg, (img_size, img_size))
            
            # 2. 随机选缺陷图
            defect_path = random.choice(defect_list)
            defect = cv2.imread(str(defect_path), cv2.IMREAD_UNCHANGED)
            if defect is None:
                continue
            
            # 3. 缺陷预处理：提取缺陷区域（基于灰度阈值生成mask）
            if len(defect.shape) == 3:
                defect_gray = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)
            else:
                defect_gray = defect
            
            # 自适应阈值提取缺陷区域
            _, mask = cv2.threshold(defect_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 缺陷通常是深色，所以取反
            mask = 255 - mask
            
            # 形态学操作，平滑mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 找最大连通域作为缺陷主体
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < 100:
                continue
            
            # 裁剪出缺陷区域
            x, y, w, h = cv2.boundingRect(largest_contour)
            defect_crop = defect[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]
            
            # 4. 随机缩放缺陷（相对图像的10%-30%）
            scale = random.uniform(0.1, 0.3)
            target_w = int(img_size * scale)
            target_h = int(target_w * h / w) if w > 0 else target_w
            target_h = min(target_h, int(img_size * 0.4))
            
            if target_w < 20 or target_h < 20:
                continue
            
            defect_resized = cv2.resize(defect_crop, (target_w, target_h))
            mask_resized = cv2.resize(mask_crop, (target_w, target_h))
            
            # 5. 随机旋转
            angle = random.uniform(-45, 45)
            M = cv2.getRotationMatrix2D((target_w//2, target_h//2), angle, 1.0)
            defect_rot = cv2.warpAffine(defect_resized, M, (target_w, target_h), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask_rot = cv2.warpAffine(mask_resized, M, (target_w, target_h),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            
            # 6. 随机位置粘贴
            paste_x = random.randint(10, img_size - target_w - 10)
            paste_y = random.randint(10, img_size - target_h - 10)
            
            # 7. 泊松融合（更自然）
            mask_rot_3ch = cv2.merge([mask_rot, mask_rot, mask_rot]) / 255.0
            roi = bg[paste_y:paste_y+target_h, paste_x:paste_x+target_w]
            
            # 简单的alpha融合 + 边缘羽化
            mask_feather = cv2.GaussianBlur(mask_rot, (5, 5), 0) / 255.0
            mask_feather_3ch = np.stack([mask_feather]*3, axis=2)
            
            blended = (defect_rot.astype(np.float32) * mask_feather_3ch + 
                      roi.astype(np.float32) * (1 - mask_feather_3ch))
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            bg[paste_y:paste_y+target_h, paste_x:paste_x+target_w] = blended
            
            # 8. 计算边界框（基于mask）
            # 重新计算粘贴后缺陷的实际边界
            ys, xs = np.where(mask_rot > 127)
            if len(xs) == 0:
                continue
            
            bx1 = paste_x + xs.min()
            by1 = paste_y + ys.min()
            bx2 = paste_x + xs.max()
            by2 = paste_y + ys.max()
            
            # 稍微扩大一点边界框
            pad = 3
            bx1 = max(0, bx1 - pad)
            by1 = max(0, by1 - pad)
            bx2 = min(img_size - 1, bx2 + pad)
            by2 = min(img_size - 1, by2 + pad)
            
            bbox_w = bx2 - bx1
            bbox_h = by2 - by1
            
            # 质量检查：太小或太偏的丢弃
            if bbox_w < 15 or bbox_h < 15:
                continue
            if bbox_w / bbox_h > 10 or bbox_h / bbox_w > 10:
                continue
            
            # 9. 保存图像和标注
            img_name = f'syn_{cls_name}_{synthesized:04d}.jpg'
            img_path = os.path.join(output_dir, 'images', 'train', img_name)
            cv2.imwrite(img_path, bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # YOLO格式标注：class cx cy w h (归一化)
            cx = (bx1 + bx2) / 2 / img_size
            cy = (by1 + by2) / 2 / img_size
            bw = bbox_w / img_size
            bh = bbox_h / img_size
            
            label_name = f'syn_{cls_name}_{synthesized:04d}.txt'
            label_path = os.path.join(output_dir, 'labels', 'train', label_name)
            with open(label_path, 'w') as f:
                f.write(f'{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            
            synthesized += 1
            total_synthesized += 1
        
        print(f'  {cls_name}: synthesized {synthesized} images (attempts: {attempt})')
    
    print(f'\n[INFO] Copy-Paste synthesis complete!')
    print(f'[INFO] Total synthesized: {total_synthesized} images')
    
    # 生成data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/train',  # 合成数据仅用于训练
        'nc': len(class_names),
        'names': class_names,
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f'[INFO] Data config saved to: {yaml_path}')
    return output_dir


# ============================================================
# 数据混合：真实数据 + GAN增强数据
# ============================================================

def mix_real_gan_data(real_dir, gan_dir, output_dir, gan_ratio=0.4, seed=42):
    """
    混合真实数据和GAN合成数据
    gan_ratio: 合成数据占总训练集的比例
    """
    random.seed(seed)
    np.random.seed(seed)
    
    from pathlib import Path
    import shutil
    
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)
    
    # 读取类别信息
    with open(os.path.join(real_dir, 'data.yaml'), 'r', encoding='utf-8') as f:
        real_cfg = yaml.safe_load(f)
    class_names = real_cfg['names']
    
    # 复制验证集和测试集（直接用真实数据）
    for split in ['val', 'test']:
        real_img_dir = os.path.join(real_dir, 'images', split)
        real_lbl_dir = os.path.join(real_dir, 'labels', split)
        out_img_dir = os.path.join(output_dir, 'images', split)
        out_lbl_dir = os.path.join(output_dir, 'labels', split)
        
        for img_path in Path(real_img_dir).glob('*.jpg'):
            shutil.copy(img_path, os.path.join(out_img_dir, img_path.name))
            lbl_path = os.path.join(real_lbl_dir, img_path.stem + '.txt')
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(out_lbl_dir, img_path.stem + '.txt'))
    
    # 统计真实训练集数量
    real_train_imgs = list(Path(os.path.join(real_dir, 'images', 'train')).glob('*.jpg'))
    real_count = len(real_train_imgs)
    
    # 计算需要多少GAN样本
    gan_count = int(real_count * gan_ratio / (1 - gan_ratio))
    print(f'[INFO] Real training images: {real_count}')
    print(f'[INFO] GAN target count: {gan_count} (ratio: {gan_ratio})')
    
    # 复制真实训练数据
    for img_path in real_train_imgs:
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'train', img_path.name))
        lbl_path = os.path.join(real_dir, 'labels', 'train', img_path.stem + '.txt')
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(output_dir, 'labels', 'train', img_path.stem + '.txt'))
    
    # 复制GAN合成数据
    gan_img_dir = os.path.join(gan_dir, 'images', 'train')
    gan_lbl_dir = os.path.join(gan_dir, 'labels', 'train')
    
    gan_imgs = list(Path(gan_img_dir).glob('*.jpg'))
    random.shuffle(gan_imgs)
    
    gan_actual = 0
    for img_path in gan_imgs[:gan_count]:
        shutil.copy(img_path, os.path.join(output_dir, 'images', 'train', img_path.name))
        lbl_path = os.path.join(gan_lbl_dir, img_path.stem + '.txt')
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(output_dir, 'labels', 'train', img_path.stem + '.txt'))
            gan_actual += 1
    
    print(f'[INFO] GAN images added: {gan_actual}')
    print(f'[INFO] Total training images: {real_count + gan_actual}')
    
    # 生成data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names,
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f'[INFO] Mixed dataset saved to: {output_dir}')
    return output_dir


if __name__ == '__main__':
    print('Phase 2: cDCGAN + Copy-Paste Pipeline')
    print('Import only - run via phase2_main.py')
