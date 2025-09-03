import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_otsu, gabor
from scipy import ndimage
import matplotlib.pyplot as plt


def complete_fingerprint_feature_extraction(image_path):
    # -------------------- 1. Read & Enhanced Pre-processing --------------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Multi-stage enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Advanced background removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bg = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    enhanced = cv2.subtract(enhanced, bg)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # -------------------- 2. Create Fingerprint Region Mask --------------------
    def create_fingerprint_mask(img):
        """Create a mask to identify the actual fingerprint region"""
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mask)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255

        boundary_margin = 20
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (boundary_margin * 2, boundary_margin * 2))
        inner_mask = cv2.erode(mask, kernel_erode)

        return mask, inner_mask

    fingerprint_mask, inner_mask = create_fingerprint_mask(enhanced)

    # ====================== LEVEL 1 FEATURES ======================

    def extract_level1_features(img, mask):
        """Extract Level 1 features: Ridge Flow and Ridge Frequency"""

        # ---- Ridge Flow (Orientation) Extraction ----
        def compute_ridge_orientation(img, block_size=16):
            """Compute ridge orientation using gradient-based method"""
            h, w = img.shape
            orientation_map = np.zeros((h, w), dtype=np.float32)
            coherence_map = np.zeros((h, w), dtype=np.float32)

            # Compute gradients
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            # Process in blocks
            for i in range(0, h - block_size, block_size // 2):
                for j in range(0, w - block_size, block_size // 2):
                    if np.mean(mask[i:i + block_size, j:j + block_size]) > 127:
                        # Extract block
                        block_gx = sobelx[i:i + block_size, j:j + block_size]
                        block_gy = sobely[i:i + block_size, j:j + block_size]

                        # Compute orientation tensor components
                        gxx = np.sum(block_gx * block_gx)
                        gxy = np.sum(block_gx * block_gy)
                        gyy = np.sum(block_gy * block_gy)

                        # Ridge orientation (perpendicular to gradient)
                        if gxx - gyy != 0:
                            theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                        else:
                            theta = 0

                        # Ridge orientation is perpendicular to gradient direction
                        ridge_orientation = theta + np.pi / 2

                        # Coherence measure
                        coherence = np.sqrt((gxx - gyy) * 2 + 4 * gxy * 2) / (gxx + gyy + 1e-5)

                        # Fill block
                        orientation_map[i:i + block_size, j:j + block_size] = ridge_orientation
                        coherence_map[i:i + block_size, j:j + block_size] = coherence

            return orientation_map, coherence_map

        # ---- Ridge Frequency Extraction ----
        def compute_ridge_frequency(img, orientation_map, mask, block_size=16):
            """Compute ridge frequency using spectral analysis"""
            h, w = img.shape
            frequency_map = np.zeros((h, w), dtype=np.float32)

            for i in range(0, h - block_size, block_size // 2):
                for j in range(0, w - block_size, block_size // 2):
                    if np.mean(mask[i:i + block_size, j:j + block_size]) > 127:
                        block = img[i:i + block_size, j:j + block_size]
                        orientation = orientation_map[i + block_size // 2, j + block_size // 2]

                        # Project block along ridge direction
                        projected_signal = []
                        for k in range(block_size):
                            x = int(block_size // 2 + k * np.cos(orientation))
                            y = int(block_size // 2 + k * np.sin(orientation))
                            if 0 <= x < block_size and 0 <= y < block_size:
                                projected_signal.append(block[y, x])

                        if len(projected_signal) > 8:
                            # Find dominant frequency using autocorrelation
                            signal = np.array(projected_signal)
                            autocorr = np.correlate(signal, signal, mode='full')
                            autocorr = autocorr[len(autocorr) // 2:]

                            # Find peaks in autocorrelation
                            peaks = []
                            for p in range(3, len(autocorr) - 3):
                                if (autocorr[p] > autocorr[p - 1] and autocorr[p] > autocorr[p + 1] and
                                        autocorr[p] > autocorr[p - 2] and autocorr[p] > autocorr[p + 2]):
                                    peaks.append(p)

                            if peaks:
                                ridge_period = peaks[0] if peaks[0] > 0 else 8
                                frequency = 1.0 / ridge_period
                            else:
                                frequency = 0.1  # Default frequency
                        else:
                            frequency = 0.1

                        frequency_map[i:i + block_size, j:j + block_size] = frequency

            return frequency_map

        # Extract Level 1 features
        orientation_map, coherence_map = compute_ridge_orientation(img, block_size=16)
        frequency_map = compute_ridge_frequency(img, orientation_map, mask, block_size=16)

        return orientation_map, coherence_map, frequency_map

    # Extract Level 1 features
    orientation_map, coherence_map, frequency_map = extract_level1_features(enhanced, fingerprint_mask)

    # ---- Create Level 1 Visualization ----
    def create_level1_visualization(img, orientation_map, coherence_map, mask):
        """Create Level 1 visualization with ridge flow directions"""
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape

        # Draw orientation vectors
        step = 12  # Grid spacing
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                if mask[y, x] > 0 and coherence_map[y, x] > 0.3:  # Only high coherence areas
                    angle = orientation_map[y, x]
                    length = 8

                    # Calculate line endpoints
                    dx = int(length * np.cos(angle))
                    dy = int(length * np.sin(angle))

                    # Draw orientation line
                    cv2.line(vis, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 0), 1)

                    # Draw small circle at center
                    cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

        return vis

    level1_vis = create_level1_visualization(enhanced, orientation_map, coherence_map, fingerprint_mask)

    # ====================== LEVEL 2 FEATURES ======================

    # Binarization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    if np.mean(binary[fingerprint_mask > 0]) > 127:
        binary = 255 - binary

    binary = cv2.bitwise_and(binary, fingerprint_mask)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    binary_cleaned = remove_small_objects(binary > 0, min_size=50)
    binary = (binary_cleaned * 255).astype(np.uint8)

    # Skeletonization
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)

    def clean_skeleton(skel):
        """Remove short spurs and artifacts from skeleton"""
        cleaned = skel.copy()
        iterations = 3

        for _ in range(iterations):
            endpoints = []
            h, w = cleaned.shape
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if cleaned[i, j]:
                        neighbors = np.sum(cleaned[i - 1:i + 2, j - 1:j + 2]) - cleaned[i, j]
                        if neighbors == 1:
                            endpoints.append((i, j))

            for y, x in endpoints:
                length = 0
                curr_y, curr_x = y, x
                visited = set()

                while length < 8:
                    visited.add((curr_y, curr_x))
                    neighbors = []

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = curr_y + dy, curr_x + dx
                            if (0 <= ny < h and 0 <= nx < w and
                                    cleaned[ny, nx] and (ny, nx) not in visited):
                                neighbors.append((ny, nx))

                    if len(neighbors) == 1:
                        cleaned[curr_y, curr_x] = 0
                        curr_y, curr_x = neighbors[0]
                        length += 1
                    else:
                        break

        return cleaned

    skeleton_cleaned = clean_skeleton(skeleton)
    skeleton_img = (skeleton_cleaned * 255).astype(np.uint8)

    # Minutiae Detection (same as before)
    def detect_minutiae_with_boundary_filter(skel, inner_mask):
        minutiae_end = []
        minutiae_bif = []
        rows, cols = skel.shape

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        for i in range(3, rows - 3):
            for j in range(3, cols - 3):
                if skel[i, j] and inner_mask[i, j] > 0:
                    neighbors = []
                    for dy, dx in offsets:
                        neighbors.append(1 if skel[i + dy, j + dx] else 0)

                    cn = 0
                    for k in range(8):
                        cn += abs(neighbors[k] - neighbors[(k + 1) % 8])
                    cn = cn // 2

                    if cn == 1:
                        if is_valid_minutiae(skel, i, j, inner_mask, 'ending'):
                            minutiae_end.append((j, i))
                    elif cn == 3:
                        if is_valid_minutiae(skel, i, j, inner_mask, 'bifurcation'):
                            minutiae_bif.append((j, i))

        return minutiae_end, minutiae_bif

    def is_valid_minutiae(skel, y, x, inner_mask, minutiae_type):
        margin = 15
        h, w = skel.shape

        if (y < margin or y >= h - margin or x < margin or x >= w - margin):
            return False

        local_region = skel[y - 5:y + 6, x - 5:x + 6]
        mask_region = inner_mask[y - 5:y + 6, x - 5:x + 6]

        ridge_density = np.sum(local_region) / np.sum(mask_region > 0) if np.sum(mask_region > 0) > 0 else 0

        if ridge_density < 0.1:
            return False

        if minutiae_type == 'ending':
            extended_region = skel[y - 8:y + 9, x - 8:x + 9]
            if np.sum(extended_region) < 5:
                return False

        return True

    endings, bifurcations = detect_minutiae_with_boundary_filter(skeleton_cleaned, inner_mask)

    def remove_close_minutiae(endings, bifurcations, min_distance=12):
        def filter_close_points(points, min_dist):
            if len(points) < 2:
                return points

            filtered = []
            for i, p1 in enumerate(points):
                too_close = False
                for j, p2 in enumerate(filtered):
                    dist = np.sqrt((p1[0] - p2[0]) * 2 + (p1[1] - p2[1]) * 2)
                    if dist < min_dist:
                        too_close = True
                        break
                if not too_close:
                    filtered.append(p1)
            return filtered

        filtered_endings = filter_close_points(endings, min_distance)
        filtered_bifurcations = filter_close_points(bifurcations, min_distance)

        return filtered_endings, filtered_bifurcations

    endings_filtered, bifurcations_filtered = remove_close_minutiae(endings, bifurcations)

    # Create Level 2 visualization
    level2_vis = np.zeros((skeleton_img.shape[0], skeleton_img.shape[1], 3), dtype=np.uint8)
    level2_vis[skeleton_cleaned] = [255, 255, 255]

    for (x, y) in endings_filtered:
        cv2.circle(level2_vis, (x, y), 4, (0, 0, 255), 2)
        cv2.circle(level2_vis, (x, y), 2, (0, 0, 255), -1)

    for (x, y) in bifurcations_filtered:
        cv2.rectangle(level2_vis, (x - 3, y - 3), (x + 3, y + 3), (0, 255, 0), 2)
        cv2.rectangle(level2_vis, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)

    # -------------------- Display All Results --------------------
    plt.figure(figsize=(24, 12))

    # Row 1: Preprocessing and Level 1
    plt.subplot(3, 4, 1)
    plt.title("Original Enhanced")
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.title("Level 1: Ridge Flow")
    plt.imshow(level1_vis)
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.title("Ridge Orientation Map")
    plt.imshow(orientation_map, cmap='hsv')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.title("Ridge Coherence Map")
    plt.imshow(coherence_map, cmap='hot')
    plt.colorbar()
    plt.axis('off')

    # Row 2: Level 1 continued and Level 2 prep
    plt.subplot(3, 4, 5)
    plt.title("Ridge Frequency Map")
    plt.imshow(frequency_map, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.title("Fingerprint Mask")
    plt.imshow(fingerprint_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.title("Binary Image")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.title("Cleaned Skeleton")
    plt.imshow(skeleton_img, cmap='gray')
    plt.axis('off')

    # Row 3: Level 2 final results
    plt.subplot(3, 4, 9)
    plt.title("Level 2: Complete Result")
    plt.imshow(level2_vis)
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.title("Combined Level 1 + 2")
    combined_vis = level1_vis.copy()
    for (x, y) in endings_filtered:
        cv2.circle(combined_vis, (x, y), 6, (0, 0, 255), 2)
    for (x, y) in bifurcations_filtered:
        cv2.rectangle(combined_vis, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 0), 2)
    plt.imshow(combined_vis)
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.title("Ridge Flow Detail")
    detail_vis = enhanced.copy()
    # Add enhanced ridge flow visualization
    plt.imshow(detail_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 12)
    plt.title("Feature Summary")
    summary_text = f"""
Level 1 Features:
- Ridge Flow: Extracted
- Ridge Frequency: Computed
- Coherence: Measured

Level 2 Features:
- Ridge Endings: {len(endings_filtered)}
- Bifurcations: {len(bifurcations_filtered)}
- Total Minutiae: {len(endings_filtered) + len(bifurcations_filtered)}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("output.png")

    print("=" * 60)
    print("COMPLETE FINGERPRINT FEATURE EXTRACTION RESULTS")
    print("=" * 60)
    print(f"📊 LEVEL 1 FEATURES:")
    print(f"   ✓ Ridge Flow (Orientation): Extracted")
    print(f"   ✓ Ridge Frequency: Computed")
    print(f"   ✓ Ridge Coherence: Measured")
    print(f"   ✓ Quality Assessment: Available")

    print(f"\n📍 LEVEL 2 FEATURES:")
    print(f"   ✓ Ridge Endings: {len(endings_filtered)} detected")
    print(f"   ✓ Bifurcations: {len(bifurcations_filtered)} detected")
    print(f"   ✓ Total Minutiae: {len(endings_filtered) + len(bifurcations_filtered)}")
    print(f"   ✓ Boundary Filtering: Applied")
    print("=" * 60)

    return {
        'enhanced': enhanced,
        'level1': {
            'orientation_map': orientation_map,
            'coherence_map': coherence_map,
            'frequency_map': frequency_map,
            'flow_visualization': level1_vis
        },
        'level2': {
            'binary': binary,
            'skeleton': skeleton_cleaned,
            'minutiae_endings': endings_filtered,
            'minutiae_bifurcations': bifurcations_filtered,
            'skeleton_visualization': level2_vis
        },
        'masks': {
            'fingerprint_mask': fingerprint_mask,
            'inner_mask': inner_mask
        }
    }
# Helper: Gabor Filtering
def apply_gabor_enhancement(img, orientation_map, freq_map, mask, block_size=16):
    enhanced_gabor = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            if np.mean(mask[i:i + block_size, j:j + block_size]) > 127:
                theta = orientation_map[i + block_size // 2, j + block_size // 2]
                freq = freq_map[i + block_size // 2, j + block_size // 2]
                if freq > 0:
                    block = img[i:i + block_size, j:j + block_size]
                    real, _ = gabor(block, frequency=freq, theta=theta)
                    enhanced_gabor[i:i + block_size, j:j + block_size] = real

    enhanced_gabor = cv2.normalize(enhanced_gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced_gabor

# Helper: Estimate Minutiae Orientation
def estimate_minutiae_orientation(orientation_map, minutiae):
    oriented_minutiae = []
    for x, y in minutiae:
        angle = orientation_map[y, x]
        oriented_minutiae.append({'x': x, 'y': y, 'angle': angle})
    return oriented_minutiae

# Helper: Poincare Index (for Core Detection)
def compute_poincare_index(orientation_map, mask):
    cores = []
    h, w = orientation_map.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x] == 0:
                continue
            block = orientation_map[y - 1:y + 2, x - 1:x + 2]
            angles = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    angles.append(block[1 + dy, 1 + dx])
            angle_diff = 0
            for i in range(len(angles)):
                diff = angles[(i + 1) % 8] - angles[i]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                angle_diff += diff
            poincare_index = angle_diff / (2 * np.pi)
            if 0.4 <= abs(poincare_index) <= 0.6:
                cores.append((x, y))
    return cores



# Usage
results = complete_fingerprint_feature_extraction(
   "/home/prayansh-chhablani/download.jpeg")