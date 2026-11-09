"""
Crazing (裂纹) Defect Enhancement Module.

Specialized augmentation strategies for crazing defect detection.
Crazing is one of the most challenging defect types due to:
- Fine, web-like texture patterns
- Low contrast with background
- High intra-class variability
- Often confused with rolled-in_scale and pitted_surface

Enhancement strategies:
1. Texture-aware augmentations (CLAHE, edge enhancement)
2. Elastic deformations (simulate crack morphology variations)
3. Copy-Paste augmentation (paste real cracks onto diverse backgrounds)
4. Noise and blur robustness training
5. Class-balanced sampling
"""
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Optional


class CrazingTextureEnhance:
    """
    Texture enhancement specifically designed for crazing defects.
    
    Enhances fine crack structures and improves contrast between
    crack patterns and the background.
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: int = 8,
                 edge_enhance_strength: float = 0.3,
                 probability: float = 0.5):
        """
        Args:
            clahe_clip_limit: CLAHE clip limit
            clahe_grid_size: CLAHE grid size
            edge_enhance_strength: Edge enhancement strength (0-1)
            probability: Probability of applying this augmentation
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.edge_enhance_strength = edge_enhance_strength
        self.probability = probability
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply texture enhancement.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Enhanced image
        """
        if random.random() > self.probability:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast, making fine cracks more visible
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit * random.uniform(0.5, 1.5),
            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
        )
        enhanced = clahe.apply(gray)
        
        # Edge enhancement using unsharp masking
        if random.random() < self.edge_enhance_strength:
            blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
            sharpened = cv2.addWeighted(
                enhanced, 1 + self.edge_enhance_strength,
                blurred, -self.edge_enhance_strength, 0
            )
            enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Convert back to BGR if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced


class ElasticDeformation:
    """
    Elastic deformation augmentation for crazing defects.
    
    Simulates natural variations in crack patterns by applying
    smooth elastic deformations to the image.
    Particularly effective for web-like structures such as crazing.
    
    Reference:
        Best practices for convolutional neural networks applied to
        visual document analysis (Simard et al., 2003)
    """
    
    def __init__(self,
                 alpha: float = 30.0,
                 sigma: float = 5.0,
                 probability: float = 0.4):
        """
        Args:
            alpha: Deformation strength
            sigma: Smoothness of deformation
            probability: Probability of applying this augmentation
        """
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, image: np.ndarray, 
                 bboxes: Optional[List] = None) -> Tuple[np.ndarray, Optional[List]]:
        """
        Apply elastic deformation.
        
        Args:
            image: Input image (H, W, C)
            bboxes: List of bounding boxes [class, x_center, y_center, w, h] (normalized)
            
        Returns:
            Deformed image and updated bounding boxes
        """
        if random.random() > self.probability:
            return image, bboxes
        
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.rand(h, w) * 2 - 1
        dy = np.random.rand(h, w) * 2 - 1
        
        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=self.sigma) * self.alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap image
        deformed = cv2.remap(image, map_x, map_y, 
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        
        # Update bounding boxes (approximate)
        if bboxes is not None and len(bboxes) > 0:
            updated_bboxes = []
            for bbox in bboxes:
                cls, cx, cy, bw, bh = bbox
                # Convert to pixel coordinates
                cx_px = cx * w
                cy_px = cy * h
                bw_px = bw * w
                bh_px = bh * h
                
                # Get four corners
                corners = np.array([
                    [cx_px - bw_px/2, cy_px - bh_px/2],
                    [cx_px + bw_px/2, cy_px - bh_px/2],
                    [cx_px - bw_px/2, cy_px + bh_px/2],
                    [cx_px + bw_px/2, cy_px + bh_px/2]
                ])
                
                # Apply displacement to corners (approximate)
                deformed_corners = []
                for px, py in corners:
                    px_clamped = int(np.clip(px, 0, w-1))
                    py_clamped = int(np.clip(py, 0, h-1))
                    new_px = px + dx[py_clamped, px_clamped]
                    new_py = py + dy[py_clamped, px_clamped]
                    deformed_corners.append([new_px, new_py])
                
                deformed_corners = np.array(deformed_corners)
                
                # Compute new bounding box
                x_min = np.clip(deformed_corners[:, 0].min(), 0, w-1)
                x_max = np.clip(deformed_corners[:, 0].max(), 0, w-1)
                y_min = np.clip(deformed_corners[:, 1].min(), 0, h-1)
                y_max = np.clip(deformed_corners[:, 1].max(), 0, h-1)
                
                # Convert back to normalized coordinates
                new_cx = (x_min + x_max) / 2 / w
                new_cy = (y_min + y_max) / 2 / h
                new_bw = (x_max - x_min) / w
                new_bh = (y_max - y_min) / h
                
                updated_bboxes.append([cls, new_cx, new_cy, new_bw, new_bh])
            
            return deformed, updated_bboxes
        
        return deformed, bboxes


class CrazingCopyPaste:
    """
    Copy-Paste augmentation specifically for crazing defects.
    
    Extracts real crazing defect regions from images and pastes them
    onto other images (including non-crazing backgrounds).
    This significantly increases the diversity of crazing samples
    and helps the model learn to detect cracks in various contexts.
    
    Reference:
        Simple Copy-Paste is a Strong Data Augmentation Method
        for Instance Segmentation (Ghiasi et al., 2021)
    """
    
    def __init__(self,
                 data_dir: str,
                 class_name: str = "crazing",
                 paste_probability: float = 0.5,
                 max_pastes: int = 3,
                 min_scale: float = 0.5,
                 max_scale: float = 1.5):
        """
        Args:
            data_dir: Path to dataset directory (YOLO format)
            class_name: Class name to extract and paste
            paste_probability: Probability of applying paste
            max_pastes: Maximum number of pastes per image
            min_scale: Minimum scale factor for pasted objects
            max_scale: Maximum scale factor for pasted objects
        """
        self.class_name = class_name
        self.paste_probability = paste_probability
        self.max_pastes = max_pastes
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Collect crazing defect patches
        self.crack_patches = []
        self._collect_crack_patches(data_dir)
    
    def _collect_crack_patches(self, data_dir: str):
        """Collect crazing defect patches from training set."""
        from pathlib import Path
        import yaml
        
        data_path = Path(data_dir)
        yaml_file = data_path / "data.yaml"
        
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get('names', [])
            train_dir = data_path / data_cfg.get('train', 'train/images')
        else:
            class_names = ["crazing", "inclusion", "patches", 
                          "pitted_surface", "rolled-in_scale", "scratches"]
            train_dir = data_path / "train" / "images"
        
        if self.class_name not in class_names:
            print(f"Warning: {self.class_name} not found in class names")
            return
        
        class_idx = class_names.index(self.class_name)
        label_dir = train_dir.parent / "labels"
        
        image_files = sorted(train_dir.glob("*.jpg")) + sorted(train_dir.glob("*.png"))
        
        print(f"Collecting {self.class_name} patches from {len(image_files)} images...")
        
        for img_path in image_files[:200]:  # Limit for efficiency
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            
            h, w = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    if cls != class_idx:
                        continue
                    
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    # Add padding
                    pad_x = int(bw * w * 0.1)
                    pad_y = int(bh * h * 0.1)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    # Extract patch
                    patch = img[y1:y2, x1:x2].copy()
                    
                    if patch.size > 0 and patch.shape[0] > 10 and patch.shape[1] > 10:
                        self.crack_patches.append(patch)
        
        print(f"Collected {len(self.crack_patches)} {self.class_name} patches")
    
    def __call__(self, image: np.ndarray, 
                 bboxes: List) -> Tuple[np.ndarray, List]:
        """
        Apply copy-paste augmentation.
        
        Args:
            image: Input image (H, W, C)
            bboxes: List of bounding boxes [class, x_center, y_center, w, h]
            
        Returns:
            Augmented image and updated bounding boxes
        """
        if random.random() > self.paste_probability:
            return image, bboxes
        
        if len(self.crack_patches) == 0:
            return image, bboxes
        
        h, w = image.shape[:2]
        num_pastes = random.randint(1, self.max_pastes)
        new_bboxes = list(bboxes)
        
        for _ in range(num_pastes):
            # Randomly select a crack patch
            patch = random.choice(self.crack_patches)
            
            # Random scale
            scale = random.uniform(self.min_scale, self.max_scale)
            new_pw = int(patch.shape[1] * scale)
            new_ph = int(patch.shape[0] * scale)
            
            if new_pw < 10 or new_ph < 10 or new_pw > w // 2 or new_ph > h // 2:
                continue
            
            resized_patch = cv2.resize(patch, (new_pw, new_ph))
            
            # Random position
            x = random.randint(0, w - new_pw)
            y = random.randint(0, h - new_ph)
            
            # Random rotation
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((new_pw//2, new_ph//2), angle, 1.0)
            rotated = cv2.warpAffine(resized_patch, M, (new_pw, new_ph),
                                    borderMode=cv2.BORDER_REFLECT)
            
            # Create blend mask (feathered edges)
            mask = np.ones((new_ph, new_pw), dtype=np.float32)
            # Feather edges
            border_size = min(new_pw, new_ph) // 5
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=border_size/3)
            mask = mask[..., np.newaxis]
            
            # Blend patch into image
            roi = image[y:y+new_ph, x:x+new_pw]
            blended = (rotated * mask + roi * (1 - mask)).astype(np.uint8)
            image[y:y+new_ph, x:x+new_pw] = blended
            
            # Add new bounding box (class 0 = crazing)
            cx_norm = (x + new_pw / 2) / w
            cy_norm = (y + new_ph / 2) / h
            bw_norm = new_pw / w
            bh_norm = new_ph / h
            
            new_bboxes.append([0, cx_norm, cy_norm, bw_norm, bh_norm])
        
        return image, new_bboxes


class NoiseRobustnessAugment:
    """
    Noise and blur augmentation to improve robustness.
    
    Adds various types of noise and blur during training to improve
    the model's robustness against real-world image quality variations.
    """
    
    def __init__(self,
                 noise_probability: float = 0.3,
                 blur_probability: float = 0.2,
                 max_noise_std: float = 0.05,
                 max_blur_kernel: int = 5):
        """
        Args:
            noise_probability: Probability of adding noise
            blur_probability: Probability of adding blur
            max_noise_std: Maximum noise standard deviation (as fraction of 255)
            max_blur_kernel: Maximum blur kernel size
        """
        self.noise_probability = noise_probability
        self.blur_probability = blur_probability
        self.max_noise_std = max_noise_std
        self.max_blur_kernel = max_blur_kernel
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply noise and blur augmentation."""
        # Add Gaussian noise
        if random.random() < self.noise_probability:
            noise_std = random.uniform(0.005, self.max_noise_std) * 255
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Add Gaussian blur
        if random.random() < self.blur_probability:
            kernel_size = random.choice(range(3, self.max_blur_kernel + 1, 2))
            sigma = random.uniform(0.5, kernel_size / 2)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
        
        return image


class CrazingAugmentationPipeline:
    """
    Complete augmentation pipeline for crazing defect detection.
    
    Combines all crazing-specific augmentations in a single pipeline.
    """
    
    def __init__(self,
                 data_dir: str = None,
                 use_texture_enhance: bool = True,
                 use_elastic_deform: bool = True,
                 use_copy_paste: bool = True,
                 use_noise_robustness: bool = True,
                 texture_prob: float = 0.5,
                 elastic_prob: float = 0.4,
                 copypaste_prob: float = 0.3,
                 noise_prob: float = 0.3,
                 blur_prob: float = 0.2):
        """
        Args:
            data_dir: Path to dataset (needed for copy-paste)
            use_texture_enhance: Enable texture enhancement
            use_elastic_deform: Enable elastic deformation
            use_copy_paste: Enable copy-paste augmentation
            use_noise_robustness: Enable noise/blur robustness
            texture_prob: Probability for texture enhancement
            elastic_prob: Probability for elastic deformation
            copypaste_prob: Probability for copy-paste
            noise_prob: Probability for noise addition
            blur_prob: Probability for blur addition
        """
        self.augmentations = []
        
        if use_texture_enhance:
            self.augmentations.append(
                ('texture', CrazingTextureEnhance(probability=texture_prob))
            )
        
        if use_elastic_deform:
            self.augmentations.append(
                ('elastic', ElasticDeformation(probability=elastic_prob))
            )
        
        if use_copy_paste and data_dir is not None:
            try:
                self.augmentations.append(
                    ('copypaste', CrazingCopyPaste(
                        data_dir=data_dir,
                        paste_probability=copypaste_prob
                    ))
                )
            except Exception as e:
                print(f"Warning: CopyPaste initialization failed: {e}")
        
        if use_noise_robustness:
            self.augmentations.append(
                ('noise', NoiseRobustnessAugment(
                    noise_probability=noise_prob,
                    blur_probability=blur_prob
                ))
            )
    
    def __call__(self, image: np.ndarray, 
                 bboxes: List = None) -> Tuple[np.ndarray, List]:
        """
        Apply full augmentation pipeline.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            
        Returns:
            Augmented image and bounding boxes
        """
        for name, aug in self.augmentations:
            if name in ['texture', 'noise']:
                image = aug(image)
            elif name in ['elastic', 'copypaste']:
                image, bboxes = aug(image, bboxes)
        
        return image, bboxes


def compute_class_weights(num_per_class: List[int], 
                          mode: str = "inverse") -> List[float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        num_per_class: Number of samples per class
        mode: Weighting mode ("inverse", "sqrt", "effective")
        
    Returns:
        List of class weights
    """
    total = sum(num_per_class)
    num_classes = len(num_per_class)
    
    if mode == "inverse":
        weights = [total / (num_classes * n) for n in num_per_class]
    elif mode == "sqrt":
        weights = [np.sqrt(total / (num_classes * n)) for n in num_per_class]
    elif mode == "effective":
        beta = 0.9999
        weights = [(1 - beta) / (1 - beta ** max(n, 1)) for n in num_per_class]
    else:
        weights = [1.0] * num_classes
    
    # Normalize to have mean of 1
    weights = [w / np.mean(weights) for w in weights]
    
    return weights


if __name__ == "__main__":
    print("Crazing Enhancement Module")
    print("=" * 50)
    print("\nAvailable augmentations:")
    print("  1. CrazingTextureEnhance - CLAHE + edge enhancement")
    print("  2. ElasticDeformation - Elastic deformations for crack patterns")
    print("  3. CrazingCopyPaste - Copy-paste real cracks onto backgrounds")
    print("  4. NoiseRobustnessAugment - Noise and blur robustness")
    print("  5. CrazingAugmentationPipeline - Full pipeline")
    print("\nUse with YOLO training to improve crazing detection!")
