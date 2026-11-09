"""
DCGAN (Deep Convolutional Generative Adversarial Network) implementation
for industrial defect image generation.

Supports conditional generation (class-conditional DCGAN) to generate
specific defect types.

Reference:
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks (Radford et al., 2015)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m):
    """
    Custom weight initialization for DCGAN.
    From the DCGAN paper, all weights shall be randomly initialized
    from a Normal distribution with mean=0, stdev=0.02.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator.

    Takes a latent vector z and class label (for conditional GAN) as input,
    outputs a generated image.

    Architecture (for 64x64 output):
        Input: (z_dim + num_classes) x 1 x 1
        -> ConvTranspose2d(4x4, stride=1, padding=0) -> BatchNorm -> ReLU
        -> ConvTranspose2d(8x8, stride=2, padding=1) -> BatchNorm -> ReLU
        -> ConvTranspose2d(16x16, stride=2, padding=1) -> BatchNorm -> ReLU
        -> ConvTranspose2d(32x32, stride=2, padding=1) -> BatchNorm -> ReLU
        -> ConvTranspose2d(64x64, stride=2, padding=1) -> Tanh
    """

    def __init__(self, latent_dim: int = 100, g_feature_maps: int = 64,
                 channels: int = 1, num_classes: int = 6,
                 image_size: int = 64, conditional: bool = True):
        """
        Args:
            latent_dim: Dimension of latent vector z
            g_feature_maps: Size of feature maps in generator
            channels: Number of image channels (1 for grayscale, 3 for RGB)
            num_classes: Number of classes (for conditional GAN)
            image_size: Output image size (must be power of 2, >= 64)
            conditional: Whether to use conditional GAN
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional
        self.image_size = image_size

        # Calculate number of upsampling layers needed
        # Starting from 4x4, each layer doubles the size
        num_layers = int(np.log2(image_size / 4))

        # Input dimension: latent + class embedding (if conditional)
        input_dim = latent_dim + (num_classes if conditional else 0)

        # Build main network
        layers = []

        # First layer: project and reshape to 4x4
        layers.append(nn.ConvTranspose2d(
            input_dim, g_feature_maps * (2 ** (num_layers - 1)),
            kernel_size=4, stride=1, padding=0, bias=False
        ))
        layers.append(nn.BatchNorm2d(g_feature_maps * (2 ** (num_layers - 1))))
        layers.append(nn.ReLU(True))

        # Intermediate layers
        for i in range(num_layers - 1, 0, -1):
            in_channels = g_feature_maps * (2 ** i)
            out_channels = g_feature_maps * (2 ** (i - 1))
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))

        # Final layer: output image
        layers.append(nn.ConvTranspose2d(
            g_feature_maps, channels,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.main = nn.Sequential(*layers)

        # Class embedding for conditional GAN
        if conditional:
            self.label_emb = nn.Embedding(num_classes, num_classes)

        # Initialize weights
        self.apply(weights_init)

    def forward(self, z, labels=None):
        """
        Forward pass.

        Args:
            z: Latent vector, shape (batch_size, latent_dim)
            labels: Class labels, shape (batch_size,) for conditional GAN

        Returns:
            Generated images, shape (batch_size, channels, H, W)
        """
        if self.conditional and labels is not None:
            # Embed labels and concatenate with latent vector
            label_embedding = self.label_emb(labels)
            z = torch.cat([z, label_embedding], dim=1)

        # Reshape to (batch, C, 1, 1) for conv layers
        z = z.unsqueeze(2).unsqueeze(3)

        return self.main(z)

    def generate(self, num_samples: int, class_idx: int = None,
                 device: torch.device = None):
        """
        Generate samples.

        Args:
            num_samples: Number of samples to generate
            class_idx: Class index for conditional generation (None for random)
            device: Device to use

        Returns:
            Generated images tensor
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)

        if self.conditional:
            if class_idx is None:
                labels = torch.randint(0, self.num_classes, (num_samples,),
                                       device=device)
            else:
                labels = torch.full((num_samples,), class_idx,
                                    dtype=torch.long, device=device)
            return self.forward(z, labels), labels
        else:
            return self.forward(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator.

    Takes an image and class label (for conditional GAN) as input,
    outputs a probability of the image being real.

    Architecture (for 64x64 input):
        Input: channels x 64 x 64
        -> Conv2d(32x32) -> LeakyReLU
        -> Conv2d(16x16) -> BatchNorm -> LeakyReLU
        -> Conv2d(8x8) -> BatchNorm -> LeakyReLU
        -> Conv2d(4x4) -> BatchNorm -> LeakyReLU
        -> Conv2d(1x1) -> Sigmoid
    """

    def __init__(self, d_feature_maps: int = 64, channels: int = 1,
                 num_classes: int = 6, image_size: int = 64,
                 conditional: bool = True):
        """
        Args:
            d_feature_maps: Size of feature maps in discriminator
            channels: Number of image channels
            num_classes: Number of classes (for conditional GAN)
            image_size: Input image size
            conditional: Whether to use conditional GAN
        """
        super(Discriminator, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.conditional = conditional
        self.image_size = image_size

        # Calculate number of downsampling layers
        num_layers = int(np.log2(image_size / 4))

        # Input channels: image channels + (num_classes if conditional, projected to spatial)
        # For conditional, we use projection-based conditioning
        input_channels = channels

        layers = []

        # First layer
        layers.append(nn.Conv2d(
            input_channels, d_feature_maps,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        for i in range(1, num_layers):
            in_channels = d_feature_maps * (2 ** (i - 1))
            out_channels = d_feature_maps * (2 ** i)
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final feature extraction (to 4x4)
        final_channels = d_feature_maps * (2 ** (num_layers - 1))
        self.feature_extractor = nn.Sequential(*layers)

        # Classification head
        # For conditional GAN: project label and add to features
        if conditional:
            self.label_emb = nn.Embedding(num_classes, final_channels * 4 * 4)
            self.final_conv = nn.Conv2d(
                final_channels, 1,
                kernel_size=4, stride=1, padding=0, bias=False
            )
        else:
            self.final_conv = nn.Conv2d(
                final_channels, 1,
                kernel_size=4, stride=1, padding=0, bias=False
            )

        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.apply(weights_init)

    def forward(self, x, labels=None):
        """
        Forward pass.

        Args:
            x: Input images, shape (batch_size, channels, H, W)
            labels: Class labels, shape (batch_size,) for conditional GAN

        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        features = self.feature_extractor(x)

        if self.conditional and labels is not None:
            # Project label to feature map size and add
            label_emb = self.label_emb(labels)
            label_emb = label_emb.view(features.shape)
            features = features + label_emb

        output = self.final_conv(features)
        output = self.sigmoid(output)

        return output.view(-1, 1).squeeze(1)


class DCGAN(nn.Module):
    """
    Complete DCGAN model with generator and discriminator.
    """

    def __init__(self, latent_dim: int = 100, g_feature_maps: int = 64,
                 d_feature_maps: int = 64, channels: int = 1,
                 num_classes: int = 6, image_size: int = 64,
                 conditional: bool = True):
        super(DCGAN, self).__init__()

        self.generator = Generator(
            latent_dim=latent_dim,
            g_feature_maps=g_feature_maps,
            channels=channels,
            num_classes=num_classes,
            image_size=image_size,
            conditional=conditional
        )

        self.discriminator = Discriminator(
            d_feature_maps=d_feature_maps,
            channels=channels,
            num_classes=num_classes,
            image_size=image_size,
            conditional=conditional
        )

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional

    def forward(self, real_images, labels, z=None):
        """
        Forward pass for training.

        Args:
            real_images: Real images batch
            labels: Class labels
            z: Optional latent vector (generated if not provided)

        Returns:
            Tuple of (real_output, fake_output, fake_images)
        """
        batch_size = real_images.size(0)
        device = real_images.device

        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)

        # Generate fake images
        fake_images = self.generator(z, labels)

        # Discriminator on real and fake
        real_output = self.discriminator(real_images, labels)
        fake_output = self.discriminator(fake_images.detach(), labels)

        return real_output, fake_output, fake_images

    def generate_samples(self, num_samples: int, class_idx: int = None,
                         device: torch.device = None):
        """Generate samples from generator."""
        return self.generator.generate(num_samples, class_idx, device)
