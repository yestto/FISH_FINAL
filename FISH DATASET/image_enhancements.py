"""
IMAGE ENHANCEMENT MODULE FOR DISTURBED UNDERWATER ENVIRONMENTS
==============================================================
6 proven Computer Vision enhancement techniques designed to rescue
fish measurements from severely degraded, low-visibility video frames.

Publication Reference:
    These techniques are standard in underwater image restoration literature.
    - CLAHE: Zuiderveld (1994), "Contrast Limited Adaptive Histogram Equalization"
    - Non-Local Means: Buades et al. (2005), "A Non-Local Algorithm for Image Denoising"
    - Bilateral Filter: Tomasi & Manduchi (1998), "Bilateral Filtering for Gray and Color Images"
    - Retinex: Jobson et al. (1997), "A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes"
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. AUTO GAMMA CORRECTION (Exposure Normalization)
# ─────────────────────────────────────────────────────────────
def auto_gamma_correction(frame_bgr: np.ndarray, target_mean: float = 127.0) -> np.ndarray:
    """
    Automatically adjusts the gamma of the frame so that the mean luminance
    hits a target value. This normalizes overexposed (washed out) and
    underexposed (too dark) frames to a consistent baseline.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    if mean_lum <= 1.0:
        mean_lum = 1.0
    # gamma = log(target) / log(current_mean), clamped to [0.3, 3.0]
    gamma = np.log(target_mean / 255.0) / np.log(mean_lum / 255.0 + 1e-9)
    gamma = float(np.clip(gamma, 0.3, 3.0))

    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame_bgr, table)


# ─────────────────────────────────────────────────────────────
# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ─────────────────────────────────────────────────────────────
def apply_clahe(frame_bgr: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE on the L-channel of LAB color space.
    Unlike global histogram equalization, CLAHE operates on local tiles,
    dramatically improving local contrast in unevenly lit underwater scenes
    without blowing out highlights.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
# 3. NON-LOCAL MEANS DENOISING
# ─────────────────────────────────────────────────────────────
def denoise_nlm(frame_bgr: np.ndarray, h: float = 10.0, h_color: float = 10.0,
                template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """
    Fast denoising using Gaussian blur in LAB color space.
    Smooths luminance noise while preserving color edges.
    Replaces the slow cv2.fastNlMeansDenoisingColored for real-time processing.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    # Denoise luminance channel only (preserves color)
    ksize = max(3, int(h) | 1)  # Ensure odd kernel
    l_ch = cv2.GaussianBlur(l_ch, (ksize, ksize), 0)
    denoised_lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(denoised_lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
# 4. BILATERAL FILTER (Edge-Preserving Smoothing)
# ─────────────────────────────────────────────────────────────
def apply_bilateral_filter(frame_bgr: np.ndarray, d: int = 5,
                           sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Bilateral filter smooths flat regions (water background) while
    keeping sharp edges (fish outline) perfectly intact.
    Replaces the standard GaussianBlur which blurs everything equally.
    """
    return cv2.bilateralFilter(frame_bgr, d, sigma_color, sigma_space)


# ─────────────────────────────────────────────────────────────
# 5. ADAPTIVE MORPHOLOGICAL REFINEMENT
# ─────────────────────────────────────────────────────────────
def refine_mask_enhanced(mask: np.ndarray) -> np.ndarray:
    """
    Advanced morphological refinement for broken/fragmented masks.
    Uses larger kernels than the baseline pipeline and applies
    convex hull gap-filling to eliminate internal holes.
    """
    # Step 1: Large close to bridge gaps in fragmented masks
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=2
    )
    # Step 2: Open to remove small noise blobs
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1
    )
    # Step 3: Convex hull fill — eliminates internal holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull_mask = np.zeros_like(mask)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        mask = hull_mask

    # Step 4: Final gentle smoothing of edges
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    return mask


# ─────────────────────────────────────────────────────────────
# 6. MULTI-SCALE RETINEX (Visibility Rescue for Low-Visibility Frames)
# ─────────────────────────────────────────────────────────────
def multi_scale_retinex(frame_bgr: np.ndarray, scales: list = None) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) for underwater haze/turbidity removal.
    Estimates the illumination at multiple scales and removes it,
    dramatically boosting the visibility of a barely-visible fish
    against a murky, turbid background.
    
    Particularly effective for front-view cameras with severe turbidity.
    """
    if scales is None:
        scales = [15, 80, 250]

    frame_float = frame_bgr.astype(np.float64) + 1.0
    retinex = np.zeros_like(frame_float)

    for sigma in scales:
        blur = cv2.GaussianBlur(frame_float, (0, 0), sigma)
        retinex += np.log10(frame_float) - np.log10(blur + 1.0)

    retinex = retinex / len(scales)

    # Normalize to [0, 255]
    for i in range(3):
        channel = retinex[:, :, i]
        min_val = float(np.percentile(channel, 1))
        max_val = float(np.percentile(channel, 99))
        if max_val - min_val < 1e-6:
            retinex[:, :, i] = 128
        else:
            retinex[:, :, i] = np.clip((channel - min_val) / (max_val - min_val) * 255.0, 0, 255)

    return retinex.astype(np.uint8)


def unsharp_mask(frame_bgr: np.ndarray, sigma: float = 3.0, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking — aggressively sharpens edges in haze/fog-like
    conditions where the fish outline is nearly invisible.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# VISIBILITY QUALITY SCORE
# ─────────────────────────────────────────────────────────────
def visibility_score(frame_bgr: np.ndarray) -> float:
    """
    Computes a 0-100 visibility quality score for a frame.
    Combines contrast, sharpness, and entropy metrics.
    
    0  = pitch black / completely invisible
    100 = crystal clear, maximum contrast
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Metric 1: RMS Contrast (0-100 scale)
    contrast = float(np.std(gray.astype(np.float64)))
    contrast_score = min(contrast / 60.0, 1.0) * 100.0

    # Metric 2: Laplacian Sharpness (0-100 scale)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.std(lap))
    sharpness_score = min(sharpness / 30.0, 1.0) * 100.0

    # Metric 3: Entropy (information content, 0-100 scale)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-9))
    entropy_score = min(entropy / 7.5, 1.0) * 100.0

    # Weighted combination
    score = 0.4 * contrast_score + 0.35 * sharpness_score + 0.25 * entropy_score
    return round(float(np.clip(score, 0, 100)), 1)


# ─────────────────────────────────────────────────────────────
# FULL ENHANCEMENT PIPELINE
# ─────────────────────────────────────────────────────────────
def enhance_frame(frame_bgr: np.ndarray, low_visibility: bool = False, apply_gamma: bool = True) -> np.ndarray:
    """
    Applies the full enhancement pipeline to a single video frame.
    
    Pipeline order (carefully chosen for optimal interaction):
        1. Gamma Correction (optional)
        2. CLAHE
        3. Multi-Scale Retinex (optional)
        4. NLM Denoising
        5. Unsharp Mask (optional)
    """
    # Step 1: Normalize exposure
    if apply_gamma:
        out = auto_gamma_correction(frame_bgr)
    else:
        out = frame_bgr.copy()

    # Step 2: Local contrast enhancement
    out = apply_clahe(out, clip_limit=3.0)

    # Step 3: Retinex for severely degraded frames
    if low_visibility:
        out = multi_scale_retinex(out)

    # Step 4: Denoise (must come after contrast enhancement)
    out = denoise_nlm(out, h=5.0, h_color=5.0)

    # Step 5: Sharpen if low visibility
    if low_visibility:
        out = unsharp_mask(out, sigma=2.0, strength=1.0)

    return out


def motion_mask_enhanced(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
                         exclude: np.ndarray, diff_thresh: int) -> np.ndarray:
    """
    Enhanced motion mask using median blur (edge-preserving, fast)
    instead of the baseline GaussianBlur, plus improved morphological refinement.
    """
    # Median blur instead of Gaussian — preserves fish edges, removes salt-pepper noise
    frame_smooth = cv2.medianBlur(frame_bgr, 7)
    bg_smooth = cv2.medianBlur(bg_bgr, 7)

    d = cv2.absdiff(frame_smooth, bg_smooth)
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, diff_thresh, 255, cv2.THRESH_BINARY)

    if exclude is not None:
        m = cv2.bitwise_and(m, cv2.bitwise_not(exclude))

    # Enhanced morphological refinement (technique #5)
    m = refine_mask_enhanced(m)

    return m
