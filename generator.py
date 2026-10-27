"""Pure-PyTorch StyleGAN2 Generator Architecture and High-Level Inference.

This module provides:
1. Seed validation and reproducible Gaussian latent sampling z ~ N(0, I).
2. The pure-PyTorch StyleGAN2 Generator architecture (Mapping + Synthesis networks).
3. The high-level generate_image() inference pipeline:
   - Uses actual model inference with torch.inference_mode().
   - Connects to model_loader.load_model() for cached pretrained weights.
   - Converts raw output tensors [-1.0, 1.0] to displayable PIL RGB images.
   - Returns the image alongside execution metadata (seed, device, resolution, norms).
"""

import math
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from backend.preprocessing import validate_seed, tensor_to_image


# ---------------------------------------------------------------------------
# Seed Validation & Latent Sampling
# ---------------------------------------------------------------------------

def sample_latent(seed: int, latent_dim: int = 512, device: str = "cpu") -> torch.Tensor:
    """Sample a reproducible standard normal latent vector z ~ N(0, I) from a seed.

    Uses an isolated local torch.Generator to ensure exact reproducibility across runs
    without mutating the global random number generator state.

    Args:
        seed: Random integer seed.
        latent_dim: Dimension of latent space (512 for StyleGAN2).
        device: Target compute device.

    Returns:
        torch.Tensor: Latent tensor of shape (1, latent_dim).
    """
    valid_seed = validate_seed(seed)

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for latent vector sampling.")

    # Isolated CPU generator ensures cross-platform bit-exact seed sampling
    rng = torch.Generator(device="cpu").manual_seed(valid_seed)
    z = torch.randn(1, latent_dim, generator=rng, device="cpu").to(device)
    return z


# Backward-compatible alias
sample_z = sample_latent


def calculate_num_layers(resolution: int = 1024) -> int:

    """Calculate the number of style layers in w+ space for a given resolution.

    Formula: 2 * (log2(resolution) - 1)
    - 256x256   -> 14 layers
    - 512x512   -> 16 layers
    - 1024x1024 -> 18 layers
    """
    log_res = int(math.log2(resolution))
    return 2 * (log_res - 1)


# ---------------------------------------------------------------------------
# Pure-PyTorch Layers (No CUDA / Ninja Required)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class PixelNorm(nn.Module):
        """Pixelwise normalization used in the mapping network."""
        def __init__(self, eps: float = 1e-8):
            super().__init__()
            self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

    def make_kernel(k):
        """Create a 2D separable filter kernel from 1D coefficients."""
        k = torch.tensor(k, dtype=torch.float32)
        if k.ndim == 1:
            k = k[None, :] * k[:, None]
        return k / k.sum()

    def upfirdn2d_native(input_t, kernel, up_x=1, up_y=1, down_x=1, down_y=1,
                         pad_x0=0, pad_x1=0, pad_y0=0, pad_y1=0):
        """Pure-PyTorch implementation of upsample -> FIR filter -> downsample."""
        _, channel, in_h, in_w = input_t.shape
        input_t = input_t.permute(0, 2, 3, 1)

        _, in_h, in_w, minor = input_t.shape
        kernel_h, kernel_w = kernel.shape

        out = input_t.view(-1, in_h, 1, in_w, 1, minor)
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
        out = out.view(-1, in_h * up_y, in_w * up_x, minor)

        out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
        out = out[
            :,
            max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
            max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
            :,
        ]
        out = out.permute(0, 3, 1, 2)
        out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w).to(out.device, out.dtype)
        out = F.conv2d(out, w)
        out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                          in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
        out = out.permute(0, 2, 3, 1)
        out = out[:, ::down_y, ::down_x, :]
        return out.permute(0, 3, 1, 2)

    def upfirdn2d(input_t, kernel, up=1, down=1, pad=(0, 0)):
        return upfirdn2d_native(input_t, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

    class Upsample(nn.Module):
        """Upsample feature maps by factor 2 with binomial anti-aliasing."""
        def __init__(self, kernel, factor=2):
            super().__init__()
            self.factor = factor
            kernel = make_kernel(kernel) * (factor ** 2)
            self.register_buffer("kernel", kernel)

            p = kernel.shape[0] - factor
            self.pad = ((p + 1) // 2 + factor - 1, p // 2)

        def forward(self, input_t):
            return upfirdn2d(input_t, self.kernel, up=self.factor, down=1, pad=self.pad)

    class Blur(nn.Module):
        """Apply anti-aliasing binomial blur filter."""
        def __init__(self, kernel, pad, upsample_factor=1):
            super().__init__()
            kernel = make_kernel(kernel)
            if upsample_factor > 1:
                kernel = kernel * (upsample_factor ** 2)
            self.register_buffer("kernel", kernel)
            self.pad = pad

        def forward(self, input_t):
            return upfirdn2d(input_t, self.kernel, pad=self.pad)

    class EqualLinear(nn.Module):
        """Linear layer with equalized learning rate (He initialization dynamic scaling)."""
        def __init__(self, in_dim: int, out_dim: int, bias: bool = True, bias_init: float = 0.0,
                     lr_mul: float = 1.0, activation: Optional[str] = None):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
            self.activation = activation
            self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
            self.lr_mul = lr_mul

        def forward(self, input_t):
            bias = self.bias * self.lr_mul if self.bias is not None else None
            out = F.linear(input_t, self.weight * self.scale, bias=bias)
            if self.activation == "fused_lrelu":
                out = F.leaky_relu(out, 0.2) * (2 ** 0.5)
            return out

    class ModulatedConv2d(nn.Module):
        """Weight-modulated and demodulated 2D convolution for StyleGAN2."""
        def __init__(self, in_channel: int, out_channel: int, kernel_size: int, style_dim: int,
                     demodulate: bool = True, upsample: bool = False, downsample: bool = False,
                     blur_kernel: List[int] = [1, 3, 3, 1]):
            super().__init__()
            self.eps = 1e-8
            self.kernel_size = kernel_size
            self.in_channel = in_channel
            self.out_channel = out_channel
            self.upsample = upsample
            self.downsample = downsample

            if upsample:
                factor = 2
                p = (len(blur_kernel) - factor) - (kernel_size - 1)
                pad0 = (p + 1) // 2 + factor - 1
                pad1 = p // 2 + 1
                self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

            fan_in = in_channel * kernel_size ** 2
            self.scale = 1.0 / math.sqrt(fan_in)
            self.padding = kernel_size // 2

            self.weight = nn.Parameter(
                torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
            )
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1.0)
            self.demodulate = demodulate

        def forward(self, input_t, style):
            batch, in_channel, height, width = input_t.shape

            style_mod = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            weight = self.scale * self.weight * style_mod

            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

            weight = weight.view(
                batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )

            if self.upsample:
                input_t = input_t.view(1, batch * in_channel, height, width)
                weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
                weight = weight.transpose(1, 2).reshape(
                    batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
                )
                out = F.conv_transpose2d(input_t, weight, padding=0, stride=2, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
                out = self.blur(out)
            else:
                input_t = input_t.view(1, batch * in_channel, height, width)
                out = F.conv2d(input_t, weight, padding=self.padding, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)

            return out

    class NoiseInjection(nn.Module):
        """Add learned per-pixel stochastic noise to feature maps."""
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, image, noise=None):
            if noise is None:
                batch, _, height, width = image.shape
                noise = image.new_empty(batch, 1, height, width).normal_()
            return image + self.weight * noise

    class ConstantInput(nn.Module):
        """Learned constant 4x4 tensor that initiates synthesis."""
        def __init__(self, channel: int, size: int = 4):
            super().__init__()
            self.input = nn.Parameter(torch.randn(1, channel, size, size))

        def forward(self, input_t):
            batch = input_t.shape[0]
            return self.input.repeat(batch, 1, 1, 1)

    class ScaledLeakyReLU(nn.Module):
        """LeakyReLU with bias addition and sqrt(2) scaling."""
        def __init__(self, channels: int, negative_slope: float = 0.2):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(channels))
            self.negative_slope = negative_slope
            self.scale = 2 ** 0.5

        def forward(self, x):
            bias_view = self.bias.view(1, -1, 1, 1)
            return F.leaky_relu(x + bias_view, self.negative_slope) * self.scale

    class StyledConv(nn.Module):
        """Synthesis convolutional block combining ModulatedConv2d, Noise, and LeakyReLU."""
        def __init__(self, in_channel: int, out_channel: int, kernel_size: int, style_dim: int,
                     upsample: bool = False, blur_kernel: List[int] = [1, 3, 3, 1],
                     demodulate: bool = True):
            super().__init__()
            self.conv = ModulatedConv2d(
                in_channel, out_channel, kernel_size, style_dim,
                upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate
            )
            self.noise = NoiseInjection()
            self.activate = ScaledLeakyReLU(out_channel)

        def forward(self, input_t, style, noise=None):
            out = self.conv(input_t, style)
            out = self.noise(out, noise=noise)
            out = self.activate(out)
            return out

    class ToRGB(nn.Module):
        """Projects high-dimensional feature maps down to 3 RGB channels."""
        def __init__(self, in_channel: int, style_dim: int, upsample: bool = True,
                     blur_kernel: List[int] = [1, 3, 3, 1]):
            super().__init__()
            self.upsample = Upsample(blur_kernel) if upsample else None
            self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
            self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        def forward(self, input_t, style, skip=None):
            out = self.conv(input_t, style)
            out = out + self.bias
            if skip is not None:
                if self.upsample is not None:
                    skip = self.upsample(skip)
                out = out + skip
            return out

    class Generator(nn.Module):
        """Full StyleGAN2 Generator network (Mapping Network + Synthesis Network)."""
        def __init__(self, size: int = 1024, style_dim: int = 512, n_mlp: int = 8,
                     channel_multiplier: int = 2, blur_kernel: List[int] = [1, 3, 3, 1],
                     lr_mlp: float = 0.01):
            super().__init__()
            self.size = size
            self.style_dim = style_dim

            # 1. Mapping Network (z -> w)
            layers = [PixelNorm()]
            for _ in range(n_mlp):
                layers.append(
                    EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")
                )
            self.style = nn.Sequential(*layers)

            # 2. Channel distribution across resolution layers
            self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }

            self.input = ConstantInput(self.channels[4])
            self.conv1 = StyledConv(
                self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
            )
            self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

            self.log_size = int(math.log(size, 2))
            self.num_layers = (self.log_size - 2) * 2 + 1

            self.convs = nn.ModuleList()
            self.to_rgbs = nn.ModuleList()
            self.noises = nn.Module()

            in_channel = self.channels[4]

            for layer_idx in range(self.num_layers):
                res = (layer_idx + 5) // 2
                shape = [1, 1, 2 ** res, 2 ** res]
                self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

            for i in range(3, self.log_size + 1):
                out_channel = self.channels[2 ** i]

                self.convs.append(
                    StyledConv(
                        in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel
                    )
                )
                self.convs.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.to_rgbs.append(ToRGB(out_channel, style_dim))
                in_channel = out_channel

            self.n_latent = self.log_size * 2 - 2  # Total style layers in w+ (18 for 1024x1024)

        def get_latent(self, z):
            """Pass z through mapping network to obtain w."""
            return self.style(z)

        def forward(self, styles, input_is_latent: bool = False, truncation: float = 1.0,
                    truncation_latent: Optional[torch.Tensor] = None, randomize_noise: bool = False):
            """Synthesize image from latent representations.

            Args:
                styles: List of tensors or single tensor for z or w/w+.
                input_is_latent: If True, inputs are in w or w+ space; if False, inputs are z.
                truncation: Truncation factor psi.
                truncation_latent: Mean latent vector w_avg.
                randomize_noise: If True, uses random noise per pass; if False, uses registered buffers.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Generated image tensor and w+ latent tensor.
            """
            if not isinstance(styles, list):
                styles = [styles]

            if not input_is_latent:
                styles = [self.style(s) for s in styles]

            # Stochastic noise buffers (fixed noise guarantees reproducible images)
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

            # Truncation trick in w-space
            if truncation < 1.0 and truncation_latent is not None:
                styles = [
                    truncation_latent + truncation * (s - truncation_latent)
                    for s in styles
                ]

            # Broadcast or assemble into w+ space (batch, n_latent, 512)
            if len(styles) == 1:
                if styles[0].ndim < 3:
                    latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
                else:
                    latent = styles[0]
            else:
                inject_index = self.n_latent // 2
                l1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                l2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
                latent = torch.cat([l1, l2], dim=1)

            # Progressive synthesis from learned constant 4x4 input
            out = self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])
            skip = self.to_rgb1(out, latent[:, 1])

            idx = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out = conv1(out, latent[:, idx], noise=noise1)
                out = conv2(out, latent[:, idx + 1], noise=noise2)
                skip = to_rgb(out, latent[:, idx + 2], skip=skip)
                idx += 2

            return skip, latent


# ---------------------------------------------------------------------------
# High-Level Inference Functions
# ---------------------------------------------------------------------------

def generate_image(
    seed: int = 42,
    truncation: float = 0.7,
    generator: Optional[Any] = None,
    w_avg: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
    output_size: Optional[int] = None,
    randomize_noise: bool = False,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Synthesize a reproducible image from an integer seed using real model inference.

    Requirements addressed:
    - Validates integer seed cleanly.
    - Samples reproducible standard normal z ~ N(0, I).
    - Runs in inference mode (torch.inference_mode) without autograd graph tracking.
    - Uses documented StyleGAN2 mapping network and truncation trick.
    - Converts output to a displayable PIL Image.
    - Returns image alongside rich metadata.

    Args:
        seed: Random integer seed.
        truncation: Truncation cutoff psi (0.0 = average face, 1.0 = full diversity).
        generator: Preloaded Generator instance (if None, loaded via model_loader).
        w_avg: Precomputed average latent tensor.
        device: Compute device ('mps', 'cuda', 'cpu').
        output_size: Optional target size (e.g. 512) for faster display.
        randomize_noise: If False, uses fixed noise buffers for 100% bit-exact reproducibility.

    Returns:
        Tuple[Image.Image, Dict[str, Any]]: The generated PIL image and execution metadata.
    """
    valid_seed = validate_seed(seed)

    # Lazy-load model if not supplied by caller
    if generator is None or w_avg is None:
        from backend.model_loader import load_model, get_device
        target_device = device or get_device()
        generator, w_avg = load_model(device=target_device)
    else:
        target_device = device or next(generator.parameters()).device.type

    # Sample z from valid seed
    z = sample_latent(valid_seed, latent_dim=512, device=target_device)

    # Inference mode: Disables gradient tape tracking and view tracking for maximum efficiency
    with torch.inference_mode():
        w = generator.get_latent(z)
        raw_output, latent_w_plus = generator(
            [z],
            input_is_latent=False,
            truncation=truncation,
            truncation_latent=w_avg,
            randomize_noise=randomize_noise,
        )

    # Convert tensor [-1, 1] to displayable PIL image
    pil_image = tensor_to_image(raw_output, target_size=output_size)

    # Assemble metadata
    metadata: Dict[str, Any] = {
        "seed": valid_seed,
        "device": str(target_device),
        "dimensions": (pil_image.width, pil_image.height),
        "native_resolution": (generator.size, generator.size),
        "latent_dimension": generator.style_dim,
        "truncation": truncation,
        "z_norm": float(torch.norm(z).cpu()),
        "w_norm": float(torch.norm(w).cpu()),
        "raw_tensor_min": float(raw_output.min().cpu()),
        "raw_tensor_max": float(raw_output.max().cpu()),
        "randomize_noise": randomize_noise,
    }

    return pil_image, metadata
