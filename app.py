"""StyleGAN2 Latent Space Explorer - Streamlit Frontend Application.

An interactive academic tool for exploring generative latent space interpolation
between two facial identities generated from Seed A and Seed B.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import streamlit as st
import numpy as np
from PIL import Image

# Import verified backend functions
from backend.model_loader import load_model, get_device, get_model_info
from backend.preprocessing import validate_seed, image_to_bytes
from backend.interpolation import (
    generate_interpolation,
    generate_interpolation_sequence,
    create_distance_plot,
)


# ---------------------------------------------------------------------------
# 1. Model & Sequence Caching Strategy
# ---------------------------------------------------------------------------
# Load model into memory only ONCE on application startup
@st.cache_resource(show_spinner="Initializing StyleGAN2 Pretrained Model...")
def get_cached_model():
    """Load and cache the generator network and mean latent vector."""
    device = get_device()
    generator, w_avg = load_model(device=device)
    model_info = get_model_info()
    return generator, w_avg, device, model_info


# Cache sequence generation with LRU bounds (max_entries=8) to avoid memory accumulation
@st.cache_data(max_entries=8, show_spinner=False)
def get_cached_sequence(seed_a: int, seed_b: int, steps: int = 5, output_size: int = 512):
    """Compute and cache the sequence of interpolation samples from Seed A to Seed B.

    Reuses endpoint latents across all frames and bounds memory footprint.
    """
    generator, w_avg, device, _ = get_cached_model()
    return generate_interpolation_sequence(
        seed_a=seed_a,
        seed_b=seed_b,
        steps=steps,
        space="w",  # W-space ensures clean, disentangled morphing
        truncation=0.7,
        generator=generator,
        w_avg=w_avg,
        device=device,
        output_size=output_size,
        randomize_noise=False,  # Fixed noise for smooth, flicker-free morphing
    )


# Cache arbitrary slider positions with bounded cache size (max_entries=16)
@st.cache_data(max_entries=16, show_spinner=False)
def get_cached_slider_interpolation(seed_a: int, seed_b: int, alpha: float, output_size: int = 512):
    """Compute and cache the specific image selected by the slider."""
    generator, w_avg, device, _ = get_cached_model()
    return generate_interpolation(
        seed_a=seed_a,
        seed_b=seed_b,
        alpha=alpha,
        space="w",
        truncation=0.7,
        generator=generator,
        w_avg=w_avg,
        device=device,
        output_size=output_size,
        randomize_noise=False,
    )


# ---------------------------------------------------------------------------
# 2. Main Streamlit User Interface
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="StyleGAN2 Latent Space Explorer",
        page_icon="🎨",
        layout="wide",
    )

    st.title("StyleGAN2 Latent Space Explorer")
    st.markdown(
        r"""
        Explore continuous identity morphing in the **disentangled latent space ($\mathcal{W}$)**
        of a pretrained StyleGAN2 model. Enter two random seeds and explore the linear transition sequence between them.
        """
    )

    # 1. Load the pretrained model once
    try:
        generator, w_avg, device, model_info = get_cached_model()
    except Exception as err:
        st.error(f"❌ Failed to load pretrained StyleGAN2 model: {err}")
        st.stop()

    device_name = {
        "mps": "Apple Silicon GPU (Metal Performance Shaders)",
        "cuda": "NVIDIA GPU (CUDA)",
        "cpu": "CPU (Standard)",
    }.get(device, device)

    # -----------------------------------------------------------------------
    # Sidebar: Model and Hardware Specifications
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Model & Hardware Status")
        st.write(f"**Model Name:** {model_info['name']}")
        st.write(f"**Inference Device:** `{device_name}`")
        st.write(f"**Native Resolution:** `{model_info['resolution']} × {model_info['resolution']}` px")
        st.write(f"**Latent Dimension:** `{model_info['latent_dim']}D` ($\\mathcal{{W}}$-space)")
        st.markdown("---")

        st.subheader("💡 Academic Concept Guide")
        st.markdown(
            r"""
            * **Linear Interpolation in $\mathcal{W}$:**
              $$\mathbf{w}(\alpha) = (1 - \alpha) \mathbf{w}_A + \alpha \mathbf{w}_B$$
            * **Default Sequence (5 Steps):**
              $$\alpha \in [0.00, 0.25, 0.50, 0.75, 1.00]$$
            * **Endpoint Reuse:**
              Latent vectors $\mathbf{w}_A$ and $\mathbf{w}_B$ are calculated **once**
              and reused for all intermediate samples, eliminating redundant mapping passes.
            """
        )

    # -----------------------------------------------------------------------
    # Controls: Seed Inputs, Steps, and Trigger Button
    # -----------------------------------------------------------------------
    st.subheader("1. Experiment Setup")

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 2, 2, 2])

    with col_ctrl1:
        seed_a_input = st.number_input(
            "Seed A (Start Face)",
            min_value=0,
            max_value=2**32 - 1,
            value=st.session_state.get("seed_a", 42),
            step=1,
            help="Integer seed determining starting identity A in latent space.",
        )

    with col_ctrl2:
        seed_b_input = st.number_input(
            "Seed B (End Face)",
            min_value=0,
            max_value=2**32 - 1,
            value=st.session_state.get("seed_b", 77),
            step=1,
            help="Integer seed determining ending identity B in latent space.",
        )

    with col_ctrl3:
        num_steps_input = st.number_input(
            "Sequence Samples",
            min_value=3,
            max_value=9,
            value=st.session_state.get("num_steps", 5),
            step=1,
            help="Number of sequence samples to generate (default: 5 at α = 0.0, 0.25, 0.50, 0.75, 1.0).",
        )

    with col_ctrl4:
        st.write("")  # Vertical spacing
        st.write("")
        generate_btn = st.button("🚀 Generate Images", type="primary", use_container_width=True)

    # Session State: Initialize or update inputs
    if generate_btn or "has_generated" not in st.session_state:
        try:
            seed_a = validate_seed(seed_a_input)
            seed_b = validate_seed(seed_b_input)
            st.session_state["seed_a"] = seed_a
            st.session_state["seed_b"] = seed_b
            st.session_state["num_steps"] = int(num_steps_input)
            st.session_state["has_generated"] = True
        except (ValueError, TypeError) as val_err:
            st.error(f"⚠️ Invalid Seed Input: {val_err}")
            st.stop()

    active_seed_a = st.session_state["seed_a"]
    active_seed_b = st.session_state["seed_b"]
    active_steps = st.session_state["num_steps"]

    # -----------------------------------------------------------------------
    # Sequence Precomputation & Endpoint Extraction
    # -----------------------------------------------------------------------
    with st.spinner(f"Synthesizing sequence ({active_steps} samples) between Seeds {active_seed_a} and {active_seed_b}..."):
        try:
            # Generate the sequence of samples (endpoint latents are calculated once and reused)
            sequence_data = get_cached_sequence(active_seed_a, active_seed_b, steps=active_steps)
        except Exception as gen_err:
            st.error(f"❌ Error during image synthesis: {gen_err}")
            st.stop()

    # Extract endpoints directly from the sequence
    start_alpha, img_start, meta_start = sequence_data[0]
    end_alpha, img_end, meta_end = sequence_data[-1]

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Interactive Slider Control
    # -----------------------------------------------------------------------
    st.subheader("2. Interactive Latent Morphing Slider")

    interp_pct = st.slider(
        "Slide to preview any continuous position between Seed A and Seed B (0% to 100%):",
        min_value=0,
        max_value=100,
        value=st.session_state.get("slider_pct", 50),
        step=1,
        help="0% corresponds strictly to Seed A; 100% corresponds strictly to Seed B.",
    )
    st.session_state["slider_pct"] = interp_pct
    alpha_float = interp_pct / 100.0

    # Retrieve or generate current slider-selected image
    img_current, meta_current = get_cached_slider_interpolation(active_seed_a, active_seed_b, alpha=alpha_float)

    # -----------------------------------------------------------------------
    # Display: Comparison Columns (Start, Current Slider, End)
    # -----------------------------------------------------------------------
    col_start, col_curr, col_end = st.columns([1, 1.2, 1])

    with col_start:
        st.markdown(f"### 🅰️ Start Image (Seed {active_seed_a})")
        st.image(img_start, caption=f"Start: α = 0.0 (0%) | Seed {active_seed_a}", use_container_width=True)
        st.download_button(
            "Download Seed A (PNG)",
            data=image_to_bytes(img_start),
            file_name=f"stylegan2_seed_{active_seed_a}_start.png",
            mime="image/png",
            use_container_width=True,
        )

    with col_curr:
        st.markdown(f"### 🔀 Selected Interpolation ({interp_pct}%)")
        st.image(
            img_current,
            caption=f"Selected: α = {alpha_float:.2f} ({interp_pct}%) | w(α)",
            use_container_width=True,
        )
        st.download_button(
            f"Download Interpolated Image at {interp_pct}% (PNG)",
            data=image_to_bytes(img_current),
            file_name=f"stylegan2_interp_{active_seed_a}_{active_seed_b}_pct_{interp_pct}.png",
            mime="image/png",
            use_container_width=True,
        )

    with col_end:
        st.markdown(f"### 🅱️ End Image (Seed {active_seed_b})")
        st.image(img_end, caption=f"End: α = 1.0 (100%) | Seed {active_seed_b}", use_container_width=True)
        st.download_button(
            "Download Seed B (PNG)",
            data=image_to_bytes(img_end),
            file_name=f"stylegan2_seed_{active_seed_b}_end.png",
            mime="image/png",
            use_container_width=True,
        )

    # Endpoint exactness verification
    matches_start = np.array_equal(np.array(img_current), np.array(img_start)) if interp_pct == 0 else "N/A (Slider not at 0%)"
    matches_end = np.array_equal(np.array(img_current), np.array(img_end)) if interp_pct == 100 else "N/A (Slider not at 100%)"

    st.markdown("---")

    # -----------------------------------------------------------------------
    # ML Diagnostics: Distances, Space Convention, and Trajectory Plot
    # -----------------------------------------------------------------------
    st.subheader("3. 📊 Machine Learning Diagnostics & Latent Metrics")
    st.markdown(
        r"""
        **Active Latent-Space Convention:**
        **Disentangled $\mathcal{W}$-space ($\mathbb{R}^{512}$)** — The 512-dimensional vector space produced by passing
        standard Gaussian noise $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ through the 8-layer MLP Mapping Network.
        """
    )

    # Display 4 real metric cards calculated from actual running tensors
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Interpolation Factor (α)",
            value=f"{alpha_float:.2f}",
            delta=f"{interp_pct}% of path",
            help="Interpolation coefficient alpha along the linear trajectory from 0.0 to 1.0.",
        )

    with metric_col2:
        st.metric(
            label="Total Distance ||w_A - w_B||",
            value=f"{meta_current['dist_total']:.3f}",
            help="Total Euclidean (L2) distance between the two endpoint vectors in W-space.",
        )

    with metric_col3:
        st.metric(
            label="Distance from Start ||w(α) - w_A||",
            value=f"{meta_current['dist_from_start']:.3f}",
            delta=f"{meta_current['dist_from_start']:.2f} L2 units",
            help="Euclidean distance from Seed A to the current interpolated vector.",
        )

    with metric_col4:
        st.metric(
            label="Distance to End ||w_B - w(α)||",
            value=f"{meta_current['dist_to_end']:.3f}",
            delta=f"-{meta_current['dist_to_end']:.2f} remaining",
            help="Euclidean distance from the current interpolated vector to Seed B.",
        )

    # Matplotlib Distance Plot & Perceptual Distance Explanation
    plot_col, theory_col = st.columns([1.1, 1.0])

    with plot_col:
        # Generates a real Matplotlib plot of alpha vs Euclidean distances
        fig = create_distance_plot(meta_current["dist_total"], alpha_float)
        st.pyplot(fig)

    with theory_col:
        st.markdown(
            r"""
            #### 🧠 Latent Distance $\neq$ Perceptual Similarity

            In machine learning and generative vision, **Euclidean distance in latent space is not identical to human visual perception**:

            1. **Linear Geometric Metric ($L_2$):**
               Euclidean distance $\|\mathbf{w}_1 - \mathbf{w}_2\|_2$ measures purely geometric displacement in $\mathbb{R}^{512}$.
               Because we use linear interpolation, distance from the origin increases strictly linearly with $\alpha$.

            2. **Non-Linear Perceptual Manifold:**
               Human vision evaluates facial identity, semantic traits (e.g. smile, gaze, gender), and lighting non-linearly:
               * A **small Euclidean shift** across a sensitive semantic boundary (e.g., eyeglasses hyperplane) produces a dramatic visual change.
               * A **large Euclidean shift** along an insensitive null direction (e.g., subtle hair micro-texture) may look virtually identical.

            3. **Modern Evaluation:**
               Academic vision research uses deep feature metrics like **LPIPS (Learned Perceptual Image Patch Similarity)**
               to quantify true visual difference rather than raw geometric distance.
            """
        )

    # -----------------------------------------------------------------------
    # Display: Transition Sequence Row (Start to End)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"4. 🖼️ Transition Sequence Gallery ({len(sequence_data)} Samples)")
    st.markdown(
        f"Row of generated images illustrating the smooth transition from **Seed A ({active_seed_a})** "
        f"to **Seed B ({active_seed_b})** using the actual pretrained StyleGAN2 generator:"
    )

    seq_cols = st.columns(len(sequence_data))
    for idx, (alpha_val, seq_img, seq_meta) in enumerate(sequence_data):
        pct = seq_meta["percentage"]
        with seq_cols[idx]:
            if idx == 0:
                label = f"**Start (0%)**<br>Seed A: {active_seed_a}"
            elif idx == len(sequence_data) - 1:
                label = f"**End (100%)**<br>Seed B: {active_seed_b}"
            elif abs(alpha_val - 0.5) < 1e-4:
                label = f"**Midpoint ({pct}%)**<br>α = {alpha_val:.2f}"
            else:
                label = f"**{pct}%**<br>α = {alpha_val:.2f}"

            st.markdown(f"<div style='text-align: center; font-size: 0.9em; margin-bottom: 5px;'>{label}</div>", unsafe_allow_html=True)
            st.image(seq_img, use_container_width=True)


if __name__ == "__main__":
    main()
