"""
CDGAN (Contextual Discriminator GAN) implementation
for high-fidelity industrial defect image generation.

CDGAN uses a dual discriminator architecture:
- Global Discriminator: Evaluates the overall image realism
- Local Discriminator: Focuses on defect regions, ensuring fine-grained
  texture and detail authenticity

This is crucial for industrial defect detection where the defect
regions need to be highly realistic to improve detection model performance.

Reference:
    Contextual Discriminator for Generative Adversarial Networks
    (Inspired by PatchGAN and progressive growing approaches)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dcgan import weights_init, Generator


class GlobalDiscriminator(nn.Module):
    """
    Global Discriminator - evaluates the entire image.

    Same architecture as standard DCGAN discriminator,
    assesses overall image realism and coherence.
    """

    def __init__(self, d_feature_maps: int = 64, channels: int = 1,
                 num_classes: int = 6, image_size: int = 64,
                 conditional: bool = True):
        super(GlobalDiscriminator, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.conditional = conditional
        self.image_size = image_size

        num_layers = int(np.log2(image_size / 4))

        layers = []

        # First layer
        layers.append(nn.Conv2d(
            channels, d_feature_maps,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        for i in range(1, num_layers):
            in_ch = d_feature_maps * (2 ** (i - 1))
            out_ch = d_feature_maps * (2 ** i)
            layers.append(nn.Conv2d(
                in_ch, out_ch,
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        final_channels = d_feature_maps * (2 ** (num_layers - 1))
        self.feature_extractor = nn.Sequential(*layers)

        # Conditional projection
        if conditional:
            self.label_emb = nn.Embedding(num_classes, final_channels * 4 * 4)

        self.final_conv = nn.Conv2d(
            final_channels, 1,
            kernel_size=4, stride=1, padding=0, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, labels=None):
        """
        Args:
            x: Input images (B, C, H, W)
            labels: Class labels (B,)

        Returns:
            Real/fake probability (B,)
        """
        features = self.feature_extractor(x)

        if self.conditional and labels is not None:
            label_emb = self.label_emb(labels)
            label_emb = label_emb.view(features.shape)
            features = features + label_emb

        output = self.final_conv(features)
        output = self.sigmoid(output)

        return output.view(-1, 1).squeeze(1)


class LocalDiscriminator(nn.Module):
    """
    Local Discriminator (PatchGAN-style) - evaluates local image patches.

    Focuses on fine-grained details and texture authenticity of defect regions.
    Outputs a matrix of patch-level real/fake decisions.

    This is critical for industrial defects because:
    1. Defect textures need to be highly realistic at the pixel level
    2. Global discriminators can miss local texture artifacts
    3. Patch-level feedback helps generator learn detailed defect patterns
    """

    def __init__(self, d_feature_maps: int = 64, channels: int = 1,
                 num_classes: int = 6, image_size: int = 64,
                 patch_size: int = 32, conditional: bool = True):
        """
        Args:
            d_feature_maps: Base feature map size
            channels: Image channels
            num_classes: Number of classes
            image_size: Input image size
            patch_size: Receptive field size of local discriminator
            conditional: Whether to use conditional GAN
        """
        super(LocalDiscriminator, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.conditional = conditional
        self.image_size = image_size
        self.patch_size = patch_size

        # Calculate output patch grid size
        # Each layer halves the spatial dimensions
        num_layers = 3  # 3 conv layers to get patch-level output
        self.output_size = image_size // (2 ** num_layers)

        layers = []

        # Layer 1: (image_size) -> (image_size/2)
        layers.append(nn.Conv2d(
            channels, d_feature_maps,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 2: (image_size/2) -> (image_size/4)
        layers.append(nn.Conv2d(
            d_feature_maps, d_feature_maps * 2,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.BatchNorm2d(d_feature_maps * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 3: (image_size/4) -> (image_size/8)
        layers.append(nn.Conv2d(
            d_feature_maps * 2, d_feature_maps * 4,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.BatchNorm2d(d_feature_maps * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final 1x1 conv to get patch-level scores
        layers.append(nn.Conv2d(
            d_feature_maps * 4, 1,
            kernel_size=3, stride=1, padding=1, bias=False
        ))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

        # Conditional: class embedding projected to feature space
        if conditional:
            # Project class label to match intermediate feature dimensions
            self.label_embedding = nn.Sequential(
                nn.Embedding(num_classes, d_feature_maps * 4),
                nn.ReLU(True)
            )
            # Additional conv after conditioning
            self.cond_conv = nn.Conv2d(
                d_feature_maps * 4 + 1, d_feature_maps * 4,
                kernel_size=3, stride=1, padding=1, bias=False
            )
            self.cond_bn = nn.BatchNorm2d(d_feature_maps * 4)
            self.cond_relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(weights_init)

    def forward(self, x, labels=None):
        """
        Args:
            x: Input images (B, C, H, W)
            labels: Class labels (B,)

        Returns:
            Patch-level real/fake map (B, 1, H/8, W/8)
        """
        # Get intermediate features (before final layer)
        features = self.main[:-2](x)  # All layers except last conv + sigmoid

        if self.conditional and labels is not None:
            # Project label to spatial feature map
            label_emb = self.label_embedding(labels)  # (B, d*4)
            # Expand to spatial dimensions
            label_map = label_emb.unsqueeze(2).unsqueeze(3)
            label_map = label_map.expand(-1, -1,
                                         features.shape[2],
                                         features.shape[3])
            # Add as additional channel
            features = torch.cat([features, label_map[:, :1, :, :]], dim=1)
            features = self.cond_conv(features)
            features = self.cond_bn(features)
            features = self.cond_relu(features)

        # Final layers
        output = self.main[-2:](features)

        return output


class CDGAN(nn.Module):
    """
    CDGAN: Contextual Discriminator GAN

    Architecture:
        - Generator: Same as DCGAN (with conditional support)
        - Global Discriminator: Assesses overall image realism
        - Local Discriminator: Evaluates patch-level texture/detail quality

    Loss components:
        - Global adversarial loss: Full image realism
        - Local adversarial loss: Patch-level texture fidelity
        - Reconstruction loss (optional): L1 loss on image content
    """

    def __init__(self, latent_dim: int = 100, g_feature_maps: int = 64,
                 global_d_feature_maps: int = 64,
                 local_d_feature_maps: int = 64,
                 channels: int = 1, num_classes: int = 6,
                 image_size: int = 64, local_patch_size: int = 32,
                 conditional: bool = True,
                 lambda_global: float = 1.0,
                 lambda_local: float = 1.0,
                 lambda_recon: float = 0.1):
        super(CDGAN, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.lambda_recon = lambda_recon

        # Generator
        self.generator = Generator(
            latent_dim=latent_dim,
            g_feature_maps=g_feature_maps,
            channels=channels,
            num_classes=num_classes,
            image_size=image_size,
            conditional=conditional
        )

        # Global discriminator
        self.global_discriminator = GlobalDiscriminator(
            d_feature_maps=global_d_feature_maps,
            channels=channels,
            num_classes=num_classes,
            image_size=image_size,
            conditional=conditional
        )

        # Local discriminator
        self.local_discriminator = LocalDiscriminator(
            d_feature_maps=local_d_feature_maps,
            channels=channels,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=local_patch_size,
            conditional=conditional
        )

    def forward(self, real_images, labels, z=None):
        """
        Forward pass for training.

        Args:
            real_images: Real images (B, C, H, W)
            labels: Class labels (B,)
            z: Latent vector (B, latent_dim), optional

        Returns:
            Dictionary with all outputs
        """
        batch_size = real_images.size(0)
        device = real_images.device

        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)

        # Generate fake images
        fake_images = self.generator(z, labels)

        # Global discriminator outputs
        global_real = self.global_discriminator(real_images, labels)
        global_fake = self.global_discriminator(fake_images.detach(), labels)
        global_fake_for_g = self.global_discriminator(fake_images, labels)

        # Local discriminator outputs (patch-level)
        local_real = self.local_discriminator(real_images, labels)
        local_fake = self.local_discriminator(fake_images.detach(), labels)
        local_fake_for_g = self.local_discriminator(fake_images, labels)

        return {
            'fake_images': fake_images,
            'global_real': global_real,
            'global_fake': global_fake,
            'global_fake_for_g': global_fake_for_g,
            'local_real': local_real,
            'local_fake': local_fake,
            'local_fake_for_g': local_fake_for_g,
        }

    def compute_discriminator_loss(self, outputs, real_labels, fake_labels,
                                   criterion):
        """
        Compute total discriminator loss (global + local).

        Args:
            outputs: Forward pass outputs
            real_labels: Labels for real images (usually 1s with label smoothing)
            fake_labels: Labels for fake images (usually 0s)
            criterion: Loss function (BCELoss)

        Returns:
            Total discriminator loss
        """
        # Global loss
        global_loss_real = criterion(outputs['global_real'], real_labels)
        global_loss_fake = criterion(outputs['global_fake'], fake_labels)
        global_loss = (global_loss_real + global_loss_fake) * 0.5

        # Local loss (patch-level BCE)
        # Expand labels to match patch grid
        batch_size = outputs['local_real'].size(0)
        grid_size = outputs['local_real'].shape[2:]

        local_real_labels = real_labels.view(-1, 1, 1, 1).expand(
            -1, 1, grid_size[0], grid_size[1]
        )
        local_fake_labels = fake_labels.view(-1, 1, 1, 1).expand(
            -1, 1, grid_size[0], grid_size[1]
        )

        local_loss_real = criterion(outputs['local_real'], local_real_labels)
        local_loss_fake = criterion(outputs['local_fake'], local_fake_labels)
        local_loss = (local_loss_real + local_loss_fake) * 0.5

        # Total loss
        total_loss = (self.lambda_global * global_loss +
                      self.lambda_local * local_loss)

        return {
            'total': total_loss,
            'global': global_loss,
            'local': local_loss,
            'global_real': global_loss_real,
            'global_fake': global_loss_fake,
            'local_real': local_loss_real,
            'local_fake': local_loss_fake,
        }

    def compute_generator_loss(self, outputs, real_labels, criterion,
                               real_images=None):
        """
        Compute total generator loss.

        Args:
            outputs: Forward pass outputs
            real_labels: Labels for "real" (used for generator loss)
            criterion: Loss function
            real_images: Real images for reconstruction loss (optional)

        Returns:
            Total generator loss
        """
        # Global adversarial loss
        global_loss = criterion(outputs['global_fake_for_g'], real_labels)

        # Local adversarial loss
        batch_size = outputs['local_fake_for_g'].size(0)
        grid_size = outputs['local_fake_for_g'].shape[2:]
        local_labels = real_labels.view(-1, 1, 1, 1).expand(
            -1, 1, grid_size[0], grid_size[1]
        )
        local_loss = criterion(outputs['local_fake_for_g'], local_labels)

        total_loss = (self.lambda_global * global_loss +
                      self.lambda_local * local_loss)

        # Reconstruction loss (optional)
        recon_loss = torch.tensor(0.0, device=outputs['fake_images'].device)
        if self.lambda_recon > 0 and real_images is not None:
            recon_loss = F.l1_loss(outputs['fake_images'], real_images)
            total_loss = total_loss + self.lambda_recon * recon_loss

        return {
            'total': total_loss,
            'global': global_loss,
            'local': local_loss,
            'recon': recon_loss,
        }

    def generate_samples(self, num_samples: int, class_idx: int = None,
                         device: torch.device = None):
        """Generate samples from generator."""
        return self.generator.generate(num_samples, class_idx, device)
