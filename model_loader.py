"""Model loader and checkpoint management for pretrained StyleGAN2.

This module is responsible for:
- Detecting compatible hardware (Apple Silicon MPS, NVIDIA CUDA, or CPU).
- Downloading and locally caching trusted PyTorch checkpoints from Hugging Face.
- Verifying checkpoint integrity and preventing arbitrary untrusted loads.
- Instantiating and initializing the Generator once in evaluation mode (`eval()`).
- Reusing the loaded model across interactive UI slider updates.
"""

import os
from typing import Tuple, Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

from backend.generator import Generator

# ---------------------------------------------------------------------------
# Trusted Model Specifications
# ---------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Any] = {
    "name": "StyleGAN2 FFHQ Config-F",
    "description": "Pretrained on Flickr-Faces-HQ (human faces) at 1024x1024.",
    "resolution": 1024,
    "latent_dim": 512,
    "num_layers": 18,  # Total layers in w+ extended latent space
    "trusted_url": "https://huggingface.co/ZeqiangLai/StyleGAN2/resolve/main/stylegan2-ffhq-config-f.pt",
    "filename": "stylegan2-ffhq-config-f.pt",
    "expected_size_bytes": 132717355,
}

# Module-level model cache to prevent reloading the 126MB model on every slider update
_CACHED_GENERATOR: Optional[Generator] = None
_CACHED_W_AVG: Optional[torch.Tensor] = None
_CACHED_DEVICE: Optional[str] = None


def get_device() -> str:
    """Detect and return the best available compute device for inference.

    Supports:
    1. 'mps' - Apple Silicon GPU acceleration (Metal Performance Shaders).
    2. 'cuda' - NVIDIA GPU acceleration.
    3. 'cpu' - High-compatibility fallback device.
    """
    if not TORCH_AVAILABLE:
        return "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_info() -> Dict[str, Any]:
    """Return documented model metadata and configuration."""
    return MODEL_SPECS.copy()


def download_checkpoint(url: str, destination_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Download checkpoint weights from a trusted URL and cache them locally.

    Args:
        url: Direct HTTP download URL.
        destination_path: Local filesystem destination.
        chunk_size: Stream chunk size in bytes (default 1 MB).

    Returns:
        str: Path to the cached checkpoint.
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("The 'requests' library is required to download model weights.")

    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)

    # If the file already exists and is non-empty, reuse it
    if os.path.isfile(destination_path) and os.path.getsize(destination_path) > 10 * 1024 * 1024:
        return destination_path

    print(f"Downloading pretrained weights from {url} to {destination_path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Write to a temporary file first to avoid corrupted partial downloads
        temp_dest = destination_path + ".tmp"
        with open(temp_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        os.rename(temp_dest, destination_path)
        print(f"Weights downloaded successfully ({os.path.getsize(destination_path)} bytes).")
        return destination_path
    except Exception as exc:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        raise RuntimeError(
            f"Failed to download pretrained StyleGAN2 checkpoint from {url}. Error: {exc}"
        ) from exc


def verify_and_load_checkpoint_file(checkpoint_path: str) -> Dict[str, Any]:
    """Verify that the checkpoint is a trusted, well-formed PyTorch state dictionary.

    Security & Validation:
    - Ensures the file exists and has sufficient size.
    - Inspects the loaded structure to verify 'g_ema' (generator state_dict) is present.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    file_size = os.path.getsize(checkpoint_path)
    if file_size < 10 * 1024 * 1024:  # StyleGAN2 weights must be > 10MB
        raise ValueError(f"Checkpoint file at {checkpoint_path} is unusually small ({file_size} bytes).")

    # Load weights onto CPU first to decouple from target device memory
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint_data, dict) or "g_ema" not in checkpoint_data:
        raise ValueError(
            f"Invalid checkpoint format in {checkpoint_path}. "
            f"Expected a dictionary containing 'g_ema', got {type(checkpoint_data)}."
        )

    return checkpoint_data


def load_model(
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[Generator, torch.Tensor]:
    """Load the pretrained StyleGAN2 generator and the average w latent vector.

    This function implements:
    1. Caching: Reuses the existing model in memory to keep interactive UI sliders fast.
    2. Evaluation Mode: Calls generator.eval() to turn off dropout/batchnorm training behavior.
    3. Multi-device support: Automatically places model on MPS (Mac GPU), CUDA, or CPU.

    Args:
        device: Hardware device to place the model on ('mps', 'cuda', 'cpu').
        checkpoint_path: Optional explicit local path to weights.
        force_reload: If True, forces a fresh model instantiation.

    Returns:
        Tuple[Generator, torch.Tensor]: The loaded generator and the w_avg tensor.
    """
    global _CACHED_GENERATOR, _CACHED_W_AVG, _CACHED_DEVICE

    target_device = device or get_device()

    # Reuse cached model instance if already loaded on the requested device
    if (
        not force_reload
        and _CACHED_GENERATOR is not None
        and _CACHED_W_AVG is not None
        and _CACHED_DEVICE == target_device
    ):
        return _CACHED_GENERATOR, _CACHED_W_AVG

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to instantiate and run StyleGAN2.")

    # Resolve checkpoint path
    if checkpoint_path is None:
        default_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        checkpoint_path = os.path.join(default_dir, MODEL_SPECS["filename"])

    # Download from trusted source if not locally present
    if not os.path.isfile(checkpoint_path):
        download_checkpoint(MODEL_SPECS["trusted_url"], checkpoint_path)

    # Verify and load dictionary
    checkpoint_data = verify_and_load_checkpoint_file(checkpoint_path)

    # 1. Instantiate the generator architecture matching Config-F
    generator = Generator(
        size=MODEL_SPECS["resolution"],  # 1024
        style_dim=MODEL_SPECS["latent_dim"],  # 512
        n_mlp=8,  # Standard 8-layer mapping network
    )

    # 2. Load pre-trained weights into the network
    # strict=False allows deterministic non-learned blur kernels to remain initialized
    generator.load_state_dict(checkpoint_data["g_ema"], strict=False)

    # 3. Set Evaluation Mode
    # In PyTorch, model.eval() switches off stochastic training behavior like dropout
    # and ensures layers compute deterministic forward evaluations.
    generator.eval()

    # 4. Move model parameters to target hardware device
    generator.to(target_device)

    # 5. Extract precomputed w_avg tensor (average latent code for truncation trick)
    if "latent_avg" in checkpoint_data and checkpoint_data["latent_avg"] is not None:
        w_avg = checkpoint_data["latent_avg"].to(target_device)
    else:
        # Fallback: compute empirical average from random samples if missing
        with torch.no_grad():
            sample_z = torch.randn(4096, 512, device=target_device)
            w_avg = generator.style(sample_z).mean(dim=0)

    # Update cache
    _CACHED_GENERATOR = generator
    _CACHED_W_AVG = w_avg
    _CACHED_DEVICE = target_device

    return generator, w_avg
