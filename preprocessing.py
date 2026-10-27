"""Image preprocessing, tensor conversion, and seed validation utilities.

This module is responsible for:
- Validating integer random seeds with clean error reporting.
- Rescaling raw StyleGAN2 generator output tensors [-1.0, 1.0] to [0, 255] RGB.
- Converting tensors to displayable PIL Images with correct channel ordering and dimensions.
- Converting PIL images to raw bytes for saving and downloading.
- Validating the 512-dimensional latent vector shapes.
"""

import io
from typing import Any, Optional, Union
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


def validate_seed(seed: Any) -> int:
    """Validate that the supplied seed is a valid non-negative integer.

    Args:
        seed: The random seed to validate (integer or int-compatible).

    Returns:
        int: The validated integer seed.

    Raises:
        TypeError: If seed is None, a float, or not an integer type.
        ValueError: If seed is negative or exceeds standard 32-bit limits.
    """
    if seed is None:
        raise TypeError("Seed cannot be None. Please supply a valid integer seed.")

    # Strictly reject floating point numbers even if integer-valued (e.g. 42.0)
    if isinstance(seed, float):
        raise TypeError(f"Seed must be an integer, got float: {seed!r}")

    # Attempt conversion if given string or int-like object
    try:
        int_seed = int(seed)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Cannot parse seed '{seed}' as an integer.") from exc

    # Non-negative check
    if int_seed < 0:
        raise ValueError(f"Seed must be non-negative (>= 0), got: {int_seed}")

    # Standard 32-bit unsigned integer maximum
    max_seed = 2**32 - 1
    if int_seed > max_seed:
        raise ValueError(f"Seed exceeds maximum allowed 32-bit value ({max_seed}), got: {int_seed}")

    return int_seed


def tensor_to_image(tensor, target_size: Optional[int] = None) -> Image.Image:
    """Convert a PyTorch generator output tensor into a displayable RGB PIL Image.

    StyleGAN2 generator outputs images in range [-1.0, 1.0] with channel-first order:
    (3, H, W) or (1, 3, H, W).

    This function:
    1. Clamps values to [-1.0, 1.0] to remove any minor numerical overshoots.
    2. Rescales from [-1.0, 1.0] to [0, 255].
    3. Permutes channel dimensions from (C, H, W) to (H, W, C) for standard RGB.
    4. Converts to a PIL Image, with optional high-fidelity Lanczos downsampling.

    Args:
        tensor: PyTorch tensor or NumPy array of shape (3, H, W) or (1, 3, H, W).
        target_size: Optional spatial resolution to resize to (e.g. 256 or 512).

    Returns:
        PIL.Image.Image: Formatted RGB image ready for display.
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        # Detach from graph and move to CPU
        img_t = tensor.detach().cpu()
        if img_t.dim() == 4:
            img_t = img_t.squeeze(0)  # Remove batch dim: (3, H, W)

        # Clamp and scale: [-1, 1] -> [0, 255]
        img_t = (img_t.clamp(-1.0, 1.0) + 1.0) * 127.5
        # Permute (C, H, W) -> (H, W, C)
        img_np = img_t.permute(1, 2, 0).to(torch.uint8).numpy()

    elif NUMPY_AVAILABLE and isinstance(tensor, np.ndarray):
        img_np = tensor.copy()
        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip((img_np + 1.0) * 127.5, 0, 255).astype(np.uint8)

    else:
        raise TypeError("Expected a PyTorch Tensor or NumPy array for tensor_to_image.")

    pil_img = Image.fromarray(img_np, mode="RGB")

    # Optional resize (e.g. downscale 1024 to 256 or 512 for responsive UI previews)
    if target_size is not None and (pil_img.width != target_size or pil_img.height != target_size):
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return pil_img


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Encode a PIL Image into in-memory bytes for export or download.

    Args:
        image: PIL Image instance.
        format: Image format string ('PNG', 'JPEG').

    Returns:
        bytes: Encoded image data.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def validate_latent_shape(latent, expected_dim: int = 512) -> bool:
    """Validate that a latent vector has the expected 512-dimensional shape.

    Args:
        latent: Tensor or array representing z or w.
        expected_dim: Expected feature dimension (512 for StyleGAN2).

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If the last dimension is not expected_dim.
        TypeError: If the object has no shape attribute.
    """
    if hasattr(latent, "shape"):
        if latent.shape[-1] != expected_dim:
            raise ValueError(
                f"Invalid latent vector dimension: expected last dimension {expected_dim}, "
                f"got {latent.shape[-1]} (full shape: {latent.shape})"
            )
        return True
    raise TypeError("Latent must be a PyTorch Tensor or NumPy array with a .shape attribute.")
