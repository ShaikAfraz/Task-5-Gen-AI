"""Latent space vector math, interpolation, and morphing generation.

This module provides:
1. get_latent_from_seed(): Reproducible latent representations in Z, W, or W+ space.
2. interpolate_latents(): Linear interpolation (LERP) between two latent representations.
3. generate_interpolation(): High-level pipeline synthesizing images from interpolated latents.
4. Classical operations: SLERP (spherical LERP), truncation trick, and hierarchical style mixing.

Academic Latent Space Distinction:
----------------------------------
StyleGAN2 features three distinct latent representations:
1. Z-Space (z in R^512):
   - The initial isotropic Gaussian distribution: z ~ N(0, I).
   - High probability mass lies on a hypersphere shell of radius sqrt(512) approx 22.6.
   - Entangled: features like gender, pose, and hair length are entangled.
2. W-Space (w in R^512):
   - Disentangled space obtained by passing z through an 8-layer MLP Mapping Network: w = f(z).
   - Linear trajectories in W-space correspond to smooth, perceptually linear attribute morphing.
   - Default and recommended space for linear interpolation.
3. W+ Space (w+ in R^(18 x 512)):
   - Extended latent space where different 512-dim vectors feed each resolution layer.
"""

from typing import Optional, Tuple, Dict, Any, Union
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

from backend.preprocessing import validate_seed, tensor_to_image


# ---------------------------------------------------------------------------
# Latent Extraction & Validation
# ---------------------------------------------------------------------------

def get_latent_from_seed(
    seed: int,
    space: str = "w",
    generator: Optional[Any] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Obtain a reproducible latent vector in Z, W, or W+ space for a given seed.

    Args:
        seed: Valid non-negative integer seed.
        space: Latent space representation:
            - 'z' : Raw Gaussian noise vector z in R^(1 x 512).
            - 'w' : Disentangled intermediate latent vector w = f(z) in R^(1 x 512) [Default].
            - 'w+': Extended latent tensor broadcast across 18 layers in R^(1 x 18 x 512).
        generator: Preloaded StyleGAN2 Generator (loaded automatically if None).
        device: Target compute device ('mps', 'cuda', 'cpu').

    Returns:
        torch.Tensor: The latent tensor in the requested space.
    """
    from backend.generator import sample_latent

    valid_seed = validate_seed(seed)
    space_lower = space.lower()

    if space_lower not in ("z", "w", "w+"):
        raise ValueError(f"Unknown latent space '{space}'. Supported spaces are: 'z', 'w', 'w+'.")

    # Sample standard normal z in R^(1 x 512)
    z = sample_latent(valid_seed, latent_dim=512, device=device or "cpu")

    # If Z-space was requested, return z directly
    if space_lower == "z":
        return z

    # For W and W+ spaces, pass z through the 8-layer MLP Mapping Network
    if generator is None:
        from backend.model_loader import load_model, get_device
        target_device = device or get_device()
        generator, _ = load_model(device=target_device)
    else:
        target_device = device or next(generator.parameters()).device.type

    z = z.to(target_device)

    # Inference mode: no gradient tracking
    with torch.inference_mode():
        w = generator.get_latent(z)  # Shape: (1, 512)

    if space_lower == "w":
        return w

    # space == 'w+': broadcast w across all 18 resolution layers
    # Shape: (1, 18, 512)
    num_layers = getattr(generator, "n_latent", 18)
    w_plus = w.unsqueeze(1).repeat(1, num_layers, 1)
    return w_plus


# ---------------------------------------------------------------------------
# Latent Interpolation
# ---------------------------------------------------------------------------

def interpolate_latents(latent_a: Any, latent_b: Any, alpha: float) -> Any:
    """Compute an intermediate latent vector using linear interpolation (LERP).

    Mathematical Formula:
        latent(alpha) = (1.0 - alpha) * latent_a + alpha * latent_b

    Properties:
    - At alpha = 0.0: Returns latent_a exactly.
    - At alpha = 1.0: Returns latent_b exactly.
    - At alpha = 0.5: Returns the arithmetic midpoint (latent_a + latent_b) / 2.0.

    Args:
        latent_a: Starting latent vector/tensor.
        latent_b: Ending latent vector/tensor.
        alpha: Interpolation parameter strictly in range [0.0, 1.0].

    Returns:
        Interpolated vector/tensor of the same type and shape.

    Raises:
        TypeError: If alpha is non-numeric or latents are incompatible types.
        ValueError: If alpha < 0.0, alpha > 1.0, or shapes do not match.
    """
    # 1. Validate alpha type and bounds
    if alpha is None or isinstance(alpha, str):
        raise TypeError(f"alpha must be a numeric value, got: {type(alpha).__name__}")

    try:
        alpha_val = float(alpha)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Cannot parse alpha '{alpha}' as a float.") from exc

    if alpha_val < 0.0 or alpha_val > 1.0:
        raise ValueError(f"alpha must be within the closed interval [0.0, 1.0], got: {alpha_val}")

    # 2. Validate vector shape compatibility
    if not hasattr(latent_a, "shape") or not hasattr(latent_b, "shape"):
        raise TypeError("latent_a and latent_b must be PyTorch Tensors or NumPy arrays with a .shape attribute.")

    if latent_a.shape != latent_b.shape:
        raise ValueError(
            f"Incompatible vector shapes for interpolation: "
            f"latent_a has shape {latent_a.shape}, while latent_b has shape {latent_b.shape}."
        )

    # 3. Exact endpoint guarantees
    if alpha_val == 0.0:
        return latent_a.clone() if hasattr(latent_a, "clone") else latent_a.copy()
    if alpha_val == 1.0:
        return latent_b.clone() if hasattr(latent_b, "clone") else latent_b.copy()

    # 4. Standard linear interpolation formula
    return (1.0 - alpha_val) * latent_a + alpha_val * latent_b


# ---------------------------------------------------------------------------
# High-Level Interpolation Image Synthesis
# ---------------------------------------------------------------------------

def generate_interpolation(
    seed_a: int,
    seed_b: int,
    alpha: float,
    space: str = "w",
    truncation: float = 0.7,
    generator: Optional[Any] = None,
    w_avg: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
    output_size: Optional[int] = None,
    randomize_noise: bool = False,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Generate an image from the interpolated latent state between Seed A and Seed B.

    Coordinates:
    - Obtains reproducible latent representations for seed_a and seed_b in the specified space.
    - Computes the intermediate latent vector via linear interpolation at alpha.
    - Runs actual forward synthesis using the pretrained StyleGAN2 generator.
    - Returns the displayable PIL Image alongside comprehensive metadata.

    Args:
        seed_a: Starting identity seed (alpha = 0.0).
        seed_b: Ending identity seed (alpha = 1.0).
        alpha: Interpolation parameter in [0.0, 1.0].
        space: Latent space to interpolate in ('w' recommended, 'z', or 'w+').
        truncation: Truncation trick factor psi.
        generator: Preloaded Generator instance.
        w_avg: Precomputed average latent code.
        device: Target compute device.
        output_size: Optional spatial resolution resize (e.g. 512).
        randomize_noise: If False, uses fixed noise for smooth, flicker-free morphing.

    Returns:
        Tuple[Image.Image, Dict[str, Any]]: Synthesized image and metadata.
    """
    valid_seed_a = validate_seed(seed_a)
    valid_seed_b = validate_seed(seed_b)

    # Lazy-load model if needed
    if generator is None or w_avg is None:
        from backend.model_loader import load_model, get_device
        target_device = device or get_device()
        generator, w_avg = load_model(device=target_device)
    else:
        target_device = device or next(generator.parameters()).device.type

    # 1. Obtain endpoint latents in the requested space
    latent_a = get_latent_from_seed(valid_seed_a, space=space, generator=generator, device=target_device)
    latent_b = get_latent_from_seed(valid_seed_b, space=space, generator=generator, device=target_device)

    # 2. Interpolate in the specified latent space
    interp_latent = interpolate_latents(latent_a, latent_b, alpha)

    # 3. Synthesize image via generator
    # For 'w' or 'w+' spaces, inputs are already in style space: input_is_latent = True
    input_is_latent = space.lower() in ("w", "w+")

    with torch.inference_mode():
        raw_output, _ = generator(
            [interp_latent],
            input_is_latent=input_is_latent,
            truncation=truncation,
            truncation_latent=w_avg,
            randomize_noise=randomize_noise,
        )

    # 4. Format PIL Image
    pil_image = tensor_to_image(raw_output, target_size=output_size)

    # 5. Compute Euclidean distances from actual tensors
    dist_total = float(torch.norm(latent_a - latent_b).cpu())
    dist_from_start = float(torch.norm(interp_latent - latent_a).cpu())
    dist_to_end = float(torch.norm(latent_b - interp_latent).cpu())

    # 6. Compile metadata
    metadata: Dict[str, Any] = {
        "seed_a": valid_seed_a,
        "seed_b": valid_seed_b,
        "alpha": float(alpha),
        "space": space.upper(),
        "device": str(target_device),
        "dimensions": (pil_image.width, pil_image.height),
        "native_resolution": (generator.size, generator.size),
        "truncation": truncation,
        "latent_norm": float(torch.norm(interp_latent).cpu()),
        "dist_total": dist_total,
        "dist_from_start": dist_from_start,
        "dist_to_end": dist_to_end,
        "raw_tensor_min": float(raw_output.min().cpu()),
        "raw_tensor_max": float(raw_output.max().cpu()),
    }

    return pil_image, metadata


def generate_interpolation_sequence(
    seed_a: int,
    seed_b: int,
    alphas: Optional[list] = None,
    steps: int = 5,
    space: str = "w",
    truncation: float = 0.7,
    generator: Optional[Any] = None,
    w_avg: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
    output_size: Optional[int] = 512,
    randomize_noise: bool = False,
) -> list:
    """Generate a sequence of images transitioning between Seed A and Seed B.

    Requirements addressed:
    1. Generates 5 interpolation samples by default (at 0.0, 0.25, 0.50, 0.75, 1.0).
    2. Includes the starting (alpha=0.0) and ending (alpha=1.0) images in the sequence.
    3. Reuses the endpoint latent vectors (latent_a and latent_b) across all steps
       rather than recalculating them in the mapping network for every frame.
    4. Synthesizes each intermediate image using the actual pretrained StyleGAN2 generator.
    5. Avoids storing intermediate tensors indefinitely; converts each tensor to PIL
       and releases raw tensor buffers immediately to prevent memory growth.

    Args:
        seed_a: Starting integer seed.
        seed_b: Ending integer seed.
        alphas: Explicit list of alpha values in [0.0, 1.0]. If None, generates
                uniform steps (default: [0.0, 0.25, 0.50, 0.75, 1.0] for steps=5).
        steps: Number of steps if alphas is None (default: 5).
        space: Latent space ('w' recommended, 'z', or 'w+').
        truncation: Truncation trick factor psi.
        generator: Preloaded Generator instance.
        w_avg: Precomputed average latent code.
        device: Target compute device ('mps', 'cuda', 'cpu').
        output_size: Display resolution (default: 512).
        randomize_noise: If False, uses fixed noise for reproducible morphing.

    Returns:
        List of tuples: [(alpha, PIL.Image, metadata_dict), ...]
    """
    valid_seed_a = validate_seed(seed_a)
    valid_seed_b = validate_seed(seed_b)

    if generator is None or w_avg is None:
        from backend.model_loader import load_model, get_device
        target_device = device or get_device()
        generator, w_avg = load_model(device=target_device)
    else:
        target_device = device or next(generator.parameters()).device.type

    # 1. Compute endpoint latents ONCE and reuse them across all intermediate steps
    latent_a = get_latent_from_seed(valid_seed_a, space=space, generator=generator, device=target_device)
    latent_b = get_latent_from_seed(valid_seed_b, space=space, generator=generator, device=target_device)

    # 2. Determine alpha values (default: 5 samples at 0.0, 0.25, 0.50, 0.75, 1.0)
    if alphas is None:
        if steps < 2:
            raise ValueError(f"Interpolation sequence requires at least 2 steps, got: {steps}")
        alphas = [i / float(steps - 1) for i in range(steps)]

    input_is_latent = space.lower() in ("w", "w+")
    sequence_results = []

    # 3. Generate each intermediate frame using the real generator
    for alpha in alphas:
        # Interpolate between the precalculated endpoint latents
        interp_latent = interpolate_latents(latent_a, latent_b, alpha)

        with torch.inference_mode():
            raw_output, _ = generator(
                [interp_latent],
                input_is_latent=input_is_latent,
                truncation=truncation,
                truncation_latent=w_avg,
                randomize_noise=randomize_noise,
            )

        # Convert to displayable image and release raw tensor memory immediately
        img = tensor_to_image(raw_output, target_size=output_size)
        del raw_output  # Immediate memory cleanup to avoid accumulating tensors

        meta: Dict[str, Any] = {
            "seed_a": valid_seed_a,
            "seed_b": valid_seed_b,
            "alpha": float(alpha),
            "percentage": int(round(float(alpha) * 100)),
            "space": space.upper(),
            "device": str(target_device),
            "dimensions": (img.width, img.height),
            "latent_norm": float(torch.norm(interp_latent).cpu()),
        }
        sequence_results.append((float(alpha), img, meta))

    return sequence_results


# ---------------------------------------------------------------------------
# ML Diagnostics & Distance Metrics
# ---------------------------------------------------------------------------

def compute_latent_distances(latent_a: Any, latent_b: Any, latent_curr: Any) -> Dict[str, float]:
    """Calculate Euclidean (L2) distances between endpoint and current latent vectors.

    Formulas:
    - Total Distance: D_total = ||latent_a - latent_b||_2
    - Distance from Start: D_start = ||latent_curr - latent_a||_2
    - Distance to End: D_end = ||latent_b - latent_curr||_2

    Handles equal endpoints (latent_a == latent_b) cleanly without zero-division.

    Args:
        latent_a: Starting latent vector (PyTorch tensor or NumPy array).
        latent_b: Ending latent vector.
        latent_curr: Current interpolated latent vector.

    Returns:
        dict: Containing 'dist_total', 'dist_from_start', and 'dist_to_end'.
    """
    if TORCH_AVAILABLE and isinstance(latent_a, torch.Tensor):
        dist_total = float(torch.norm(latent_a - latent_b).cpu())
        dist_from_start = float(torch.norm(latent_curr - latent_a).cpu())
        dist_to_end = float(torch.norm(latent_b - latent_curr).cpu())
    elif NUMPY_AVAILABLE and isinstance(latent_a, np.ndarray):
        dist_total = float(np.linalg.norm(latent_a - latent_b))
        dist_from_start = float(np.linalg.norm(latent_curr - latent_a))
        dist_to_end = float(np.linalg.norm(latent_b - latent_curr))
    else:
        raise TypeError("Inputs must be PyTorch tensors or NumPy arrays.")

    return {
        "dist_total": dist_total,
        "dist_from_start": dist_from_start,
        "dist_to_end": dist_to_end,
    }


def create_distance_plot(dist_total: float, current_alpha: float):
    """Plot interpolation alpha vs. Euclidean latent distances using Matplotlib.

    Visualizes how the linear trajectory moves through latent space:
    - Distance from Start = alpha * D_total (increases linearly from 0 to D_total)
    - Distance to End = (1 - alpha) * D_total (decreases linearly from D_total to 0)
    - Current alpha position is marked with a vertical dashed line and point markers.
    - Zero-distance case (Seed A == Seed B) is handled gracefully without errors.

    Args:
        dist_total: Euclidean distance between latent_a and latent_b.
        current_alpha: Current interpolation factor in [0.0, 1.0].

    Returns:
        matplotlib.figure.Figure: Formatted figure object.
    """
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    alphas = np.linspace(0.0, 1.0, 100)
    d_start = alphas * dist_total
    d_end = (1.0 - alphas) * dist_total

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=100)

    # Plot linear distance lines
    ax.plot(alphas, d_start, label="Distance from Start (Seed A)", color="#1f77b4", linewidth=2.5)
    ax.plot(alphas, d_end, label="Distance to End (Seed B)", color="#ff7f0e", linewidth=2.5, linestyle="--")

    # Mark current alpha position
    curr_d_start = current_alpha * dist_total
    curr_d_end = (1.0 - current_alpha) * dist_total
    ax.axvline(x=current_alpha, color="#2ca02c", linestyle=":", alpha=0.8, linewidth=1.8, label=f"Current α = {current_alpha:.2f}")
    ax.scatter([current_alpha, current_alpha], [curr_d_start, curr_d_end], color=["#1f77b4", "#ff7f0e"], s=60, zorder=5)

    ax.set_xlabel("Interpolation Coefficient (α)", fontsize=11)
    ax.set_ylabel("Euclidean Distance (L2 Norm)", fontsize=11)
    ax.set_title(f"Latent Trajectory Distances (Total L2 Distance = {dist_total:.3f})", fontsize=12, pad=10)
    ax.set_xlim(-0.02, 1.02)

    # Handle zero-distance edge case without negative axis crashes
    if dist_total < 1e-6:
        ax.set_ylim(-0.2, 1.0)
    else:
        ax.set_ylim(-0.05 * dist_total, 1.10 * dist_total)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", framealpha=0.9, fontsize=9, ncol=3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classical Helper Operations (Preserved for compatibility)
# ---------------------------------------------------------------------------

def lerp(v1, v2, alpha: float):
    """Linear interpolation alias."""
    return interpolate_latents(v1, v2, alpha)


def slerp(v1, v2, alpha: float, eps: float = 1e-7):
    """Spherical linear interpolation along the hypersphere surface."""
    if TORCH_AVAILABLE and isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
        v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)

        dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

        if torch.all(dot > 0.9995):
            return interpolate_latents(v1, v2, alpha)

        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        weight1 = torch.sin((1.0 - alpha) * omega) / sin_omega
        weight2 = torch.sin(alpha * omega) / sin_omega
        return weight1 * v1 + weight2 * v2

    elif NUMPY_AVAILABLE and isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        v1_norm = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps)
        v2_norm = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + eps)

        dot = np.sum(v1_norm * v2_norm, axis=-1, keepdims=True)
        dot = np.clip(dot, -1.0 + eps, 1.0 - eps)

        if np.all(dot > 0.9995):
            return interpolate_latents(v1, v2, alpha)

        omega = np.arccos(dot)
        sin_omega = np.sin(omega)

        weight1 = np.sin((1.0 - alpha) * omega) / sin_omega
        weight2 = np.sin(alpha * omega) / sin_omega
        return weight1 * v1 + weight2 * v2

    else:
        raise TypeError("Inputs must both be PyTorch Tensors or both NumPy arrays.")


def apply_truncation(w, w_avg, psi: float = 0.7):
    """Apply the StyleGAN truncation trick to a latent vector in w-space."""
    return w_avg + psi * (w - w_avg)


def mix_styles(w1, w2, crossover_layer: int = 8, num_layers: int = 18):
    """Mix two latent representations across layer resolutions in w+ space."""
    if crossover_layer < 0 or crossover_layer > num_layers:
        raise ValueError(f"crossover_layer must be between 0 and {num_layers}, got {crossover_layer}")

    if TORCH_AVAILABLE and isinstance(w1, torch.Tensor) and isinstance(w2, torch.Tensor):
        mixed = w1.clone()
        mixed[:, crossover_layer:num_layers, :] = w2[:, crossover_layer:num_layers, :]
        return mixed
    elif NUMPY_AVAILABLE and isinstance(w1, np.ndarray) and isinstance(w2, np.ndarray):
        mixed = np.copy(w1)
        mixed[:, crossover_layer:num_layers, :] = w2[:, crossover_layer:num_layers, :]
        return mixed
    else:
        raise TypeError("Inputs must both be PyTorch Tensors or both NumPy arrays.")


def generate_interpolation_series(v1, v2, steps: int = 10, method: str = "lerp") -> list:
    """Generate a series of interpolated vectors between v1 and v2."""
    if steps < 2:
        raise ValueError("Interpolation series requires at least 2 steps (start and end).")

    series = []
    for step_idx in range(steps):
        alpha = step_idx / float(steps - 1)
        if method.lower() == "slerp":
            vector = slerp(v1, v2, alpha)
        else:
            vector = interpolate_latents(v1, v2, alpha)
        series.append(vector)
    return series
