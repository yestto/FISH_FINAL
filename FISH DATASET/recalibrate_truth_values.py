"""
FIXED Recalibration: Use the original estimate_grid_period_px function
but WITHOUT the biased choose_px_per_cm scoring.

ROOT CAUSE: The original choose_px_per_cm() used score = abs(L - 25.0)
which forces ALL fish to appear ~25cm long regardless of actual size.

FIX: Determine the grid period from the graph paper, then calibrate
using the known ruler markings visible in each image.
"""
import cv2
import numpy as np
import pandas as pd
import os
import glob
import sys

# Add the pipeline directory to path
sys.path.insert(0, r'C:\Users\shain\Downloads\FISH DATASET\Git_Fish_Pipeline')


def _autocorr_period_1d(signal, min_lag, max_lag):
    """Autocorrelation-based period detection (from original pipeline)."""
    signal = signal.astype(np.float64)
    signal = signal - np.mean(signal)
    n = len(signal)
    if n < max_lag + 1:
        return float("nan")
    
    # FFT-based autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    f = np.fft.rfft(signal, n=fft_size)
    acf = np.fft.irfft(f * np.conj(f), n=fft_size)[:n]
    acf = acf / (acf[0] + 1e-9)
    
    # Find first significant peak
    segment = acf[min_lag:max_lag+1]
    peaks = []
    for i in range(1, len(segment) - 1):
        if segment[i] > segment[i-1] and segment[i] > segment[i+1] and segment[i] > 0.05:
            peaks.append((i + min_lag, segment[i]))
    
    if not peaks:
        return float("nan")
    
    # Return the lag with highest correlation
    best = max(peaks, key=lambda x: x[1])
    return float(best[0])


def estimate_grid_period_px(img_bgr):
    """Detect dominant grid spacing using Sobel + autocorrelation (from original pipeline)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    px = np.mean(np.abs(gx), axis=0)
    py = np.mean(np.abs(gy), axis=1)
    
    min_lag = max(8, int(min(gray.shape[:2]) * 0.01))
    max_lag = int(min(gray.shape[:2]) * 0.25)
    if max_lag <= min_lag:
        return float("nan")
    
    dx = _autocorr_period_1d(px, min_lag=min_lag, max_lag=max_lag)
    dy = _autocorr_period_1d(py, min_lag=min_lag, max_lag=max_lag)
    vals = [v for v in [dx, dy] if np.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def segment_fish_on_grid(img_bgr):
    """Segment fish on graph paper (from original pipeline)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    bg = cv2.medianBlur(gray_blur, 51)
    diff = cv2.subtract(bg, gray_blur)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, bin_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt((a * a + b * b).astype(np.float32))
    chroma = cv2.GaussianBlur(chroma, (7, 7), 0)
    chroma_u8 = np.clip((chroma / (np.percentile(chroma, 99.5) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    _, bin_chroma = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    bin_mask = cv2.bitwise_or(bin_mask, bin_chroma)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return bin_mask, {"area_px": float("nan"), "perimeter_px": float("nan"), 
                         "length_px": float("nan"), "width_px": float("nan")}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    idx = int(np.argmax(areas)) + 1
    mask = (labels == idx).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, {"area_px": float("nan"), "perimeter_px": float("nan"),
                     "length_px": float("nan"), "width_px": float("nan")}
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    area_px = float(cv2.contourArea(hull))
    perimeter_px = float(cv2.arcLength(hull, True))
    rect = cv2.minAreaRect(hull)
    w, h = rect[1]
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    return mask, {"area_px": area_px, "perimeter_px": perimeter_px, 
                  "length_px": length_px, "width_px": width_px}


def determine_grid_size_cm(period_px, img_w, img_h, ruler_range_cm):
    """
    Given the detected grid period in pixels, determine what physical size
    each grid square represents.
    
    The graph paper has:
    - Fine lines at 1mm (0.1 cm)
    - Medium lines at 5mm (0.5 cm)  
    - Bold lines at 1cm
    
    We use the known ruler range to determine the correct mapping.
    """
    # The ruler shows the total cm span of the graph paper
    # For most images, the bottom ruler goes from 0 to ~20-28 cm
    # The image width (minus ruler) covers this span
    
    # Estimate expected PxPerCm from the ruler range
    # The graph paper typically fills ~90% of image width (rest is ruler)
    graph_paper_width_px = img_w * 0.92  # approx
    expected_px_per_cm = graph_paper_width_px / ruler_range_cm
    
    # Now check which interpretation of the grid period matches
    # If period = 1cm line spacing: px_per_cm = period_px
    # If period = 5mm line spacing: px_per_cm = period_px * 2
    # If period = 1mm line spacing: px_per_cm = period_px * 10
    
    candidates = {
        '1cm': period_px,
        '5mm': period_px * 2,
        '2mm': period_px * 5,
        '1mm': period_px * 10,
    }
    
    # Pick the one closest to expected_px_per_cm
    best_name = None
    best_diff = float('inf')
    best_ppc = None
    for name, ppc in candidates.items():
        diff = abs(ppc - expected_px_per_cm)
        if diff < best_diff:
            best_diff = diff
            best_name = name
            best_ppc = ppc
    
    return best_ppc, best_name


def main():
    root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'
    old_truth = pd.read_csv(os.path.join(root, 'truth_values.csv'))
    out_dir = os.path.join(root, 'calibration_verification_v2')
    os.makedirs(out_dir, exist_ok=True)
    
    fish_dirs = ['fish01','fish2','fish3','fish4','fish5','fish6','fish7',
                 'fish8','fish9','fish10','fish11','fish12','fish13','fish14','fish15']
    
    # Known ruler ranges from visual inspection of images (bottom ruler max cm)
    # Read from the images: bottom ruler 0→max_cm
    ruler_ranges = {
        'fish01': 22, 'fish2': 28, 'fish3': 30, 'fish4': 28,
        'fish5': 22, 'fish6': 23, 'fish7': 25, 'fish8': 20,
        'fish9': 22, 'fish10': 20, 'fish11': 17, 'fish12': 20,
        'fish13': 22, 'fish14': 19, 'fish15': 21,
    }
    
    results = []
    
    for fish_dir in fish_dirs:
        fish_id = fish_dir
        img_folder = os.path.join(root, fish_dir, 'single image')
        if not os.path.isdir(img_folder):
            continue
        
        imgs = glob.glob(os.path.join(img_folder, '*.jpg'))
        if not imgs:
            continue
        
        img = cv2.imread(imgs[0])
        if img is None:
            continue
        
        h, w = img.shape[:2]
        print(f"\n{'='*60}")
        print(f"{fish_dir}: {w}x{h}")
        
        # Step 1: Detect grid period (using original pipeline logic)
        period_px = estimate_grid_period_px(img)
        print(f"  Grid period (px): {period_px:.1f}")
        
        # Step 2: Determine what the grid represents using ruler range
        ruler_cm = ruler_ranges.get(fish_dir, 22)
        px_per_cm, grid_type = determine_grid_size_cm(period_px, w, h, ruler_cm)
        print(f"  Ruler range: 0-{ruler_cm} cm")
        print(f"  Grid type detected: {grid_type}")
        print(f"  NEW PxPerCm: {px_per_cm:.1f}")
        
        # Step 3: Segment fish (using original pipeline logic)
        mask, m = segment_fish_on_grid(img)
        length_px = m['length_px']
        width_px = m['width_px']
        area_px = m['area_px']
        perimeter_px = m['perimeter_px']
        print(f"  Fish pixels: L={length_px:.0f} W={width_px:.0f}")
        
        # Step 4: Compute measurements
        if np.isfinite(px_per_cm) and px_per_cm > 0:
            L_cm = length_px / px_per_cm
            W_cm = width_px / px_per_cm
            A_cm2 = area_px / (px_per_cm ** 2)
            P_cm = perimeter_px / px_per_cm
        else:
            L_cm = W_cm = A_cm2 = P_cm = float('nan')
        
        # Get old values
        old_row = old_truth[old_truth['FishID'] == fish_id]
        old_ppc = float(old_row['PxPerCm'].iloc[0]) if not old_row.empty else None
        old_len = float(old_row['Length_truth (cm)'].iloc[0]) if not old_row.empty else None
        old_wid = float(old_row['Width_truth (cm)'].iloc[0]) if not old_row.empty else None
        
        print(f"  OLD: PPC={old_ppc:.0f}  L={old_len:.2f}cm  W={old_wid:.2f}cm")
        print(f"  NEW: PPC={px_per_cm:.1f}  L={L_cm:.2f}cm  W={W_cm:.2f}cm")
        
        # Step 5: Create verification image
        vis = img.copy()
        overlay = vis.copy()
        if mask is not None:
            overlay[mask > 0] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(c)
                rect = cv2.minAreaRect(hull)
                box = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
                cv2.drawContours(vis, [hull], 0, (0, 255, 0), 2)
        
        text1 = f"OLD: L={old_len:.1f}cm W={old_wid:.1f}cm (PPC={old_ppc:.0f})"
        text2 = f"NEW: L={L_cm:.1f}cm W={W_cm:.1f}cm (PPC={px_per_cm:.0f}, grid={grid_type})"
        cv2.putText(vis, text1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
        cv2.putText(vis, text1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
        cv2.putText(vis, text2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
        cv2.putText(vis, text2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        
        vis_path = os.path.join(out_dir, f'{fish_id}_fixed.jpg')
        cv2.imwrite(vis_path, vis)
        
        results.append({
            'FishID': fish_id,
            'PxPerCm_OLD': old_ppc,
            'PxPerCm_NEW': round(px_per_cm, 1),
            'Grid_Period_px': round(period_px, 1),
            'Grid_Type': grid_type,
            'Length_OLD_cm': round(old_len, 2) if old_len else None,
            'Length_NEW_cm': round(L_cm, 2),
            'Width_OLD_cm': round(old_wid, 2) if old_wid else None,
            'Width_NEW_cm': round(W_cm, 2),
            'Area_NEW_cm2': round(A_cm2, 2),
            'Perimeter_NEW_cm': round(P_cm, 2),
        })
    
    # Print summary
    df = pd.DataFrame(results)
    print("\n\n" + "="*110)
    print("FINAL COMPARISON: BIASED (OLD) vs FIXED (NEW)")
    print("="*110)
    print(f"{'Fish':<8} {'PPC_OLD':>8} {'PPC_NEW':>8} {'Grid':>6} {'Len_OLD':>10} {'Len_NEW':>10} {'Wid_OLD':>10} {'Wid_NEW':>10}")
    print("-"*110)
    for _, r in df.iterrows():
        print(f"{r['FishID']:<8} {r['PxPerCm_OLD']:>8.0f} {r['PxPerCm_NEW']:>8.1f} "
              f"{r['Grid_Type']:>6} {r['Length_OLD_cm']:>10.2f} {r['Length_NEW_cm']:>10.2f} "
              f"{r['Width_OLD_cm']:>10.2f} {r['Width_NEW_cm']:>10.2f}")
    
    # Save
    out_path = os.path.join(root, 'truth_values_FIXED.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved FIXED truth values to: {out_path}")
    print(f"Verification images saved to: {out_dir}")


if __name__ == '__main__':
    main()
