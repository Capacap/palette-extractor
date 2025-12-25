#!/usr/bin/env python3
"""
Gradient extraction via adjacency space embedding.

Approach:
1. Quantize colors to bins
2. Build adjacency graph (which colors appear next to which)
3. Represent each color as a vector of its adjacency relationships
4. Apply dimensionality reduction - gradients should form continuous paths
5. Extract gradient chains by tracing paths in the embedded space
"""

import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, uniform_filter, label

from extract_colors import rgb_to_lab, lab_to_rgb, JND


def load_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load image as LAB pixels."""
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img)
    h, w = pixels.shape[:2]
    lab = rgb_to_lab(pixels.reshape(-1, 3)).reshape(h, w, 3)
    return lab, (h, w)


def quantize(lab: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Quantize LAB to JND-sized bins."""
    bin_size = scale * JND
    return np.round(lab / bin_size).astype(np.int32)


def bin_to_lab(bin_tuple: tuple, scale: float = 3.0) -> np.ndarray:
    """Convert bin indices back to LAB values."""
    return np.array(bin_tuple) * scale * JND


def build_adjacency_graph(binned: np.ndarray) -> tuple[dict, dict]:
    """
    Build full adjacency graph without any distance cutoff.

    Returns:
        adjacency: dict mapping (bin1, bin2) -> count
        coverage: dict mapping bin -> pixel count
    """
    h, w = binned.shape[:2]
    adjacency = {}

    def add_edge(b1, b2):
        if b1 == b2:
            return
        key = (b1, b2) if b1 < b2 else (b2, b1)
        adjacency[key] = adjacency.get(key, 0) + 1

    # Horizontal neighbors
    for y in range(h):
        for x in range(w - 1):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y, x + 1])
            add_edge(b1, b2)

    # Vertical neighbors
    for y in range(h - 1):
        for x in range(w):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y + 1, x])
            add_edge(b1, b2)

    # Diagonal neighbors (down-right)
    for y in range(h - 1):
        for x in range(w - 1):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y + 1, x + 1])
            add_edge(b1, b2)

    # Diagonal neighbors (down-left)
    for y in range(h - 1):
        for x in range(1, w):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y + 1, x - 1])
            add_edge(b1, b2)

    # Coverage
    flat = binned.reshape(-1, 3)
    unique, counts = np.unique(flat, axis=0, return_counts=True)
    coverage = {tuple(b): c for b, c in zip(unique, counts)}

    return adjacency, coverage


def build_directional_adjacency(binned: np.ndarray) -> dict:
    """
    Build directional adjacency graph - captures which direction colors flow.

    For each ordered pair (color_a, color_b), counts how often color_b appears
    in each direction relative to color_a.

    Returns:
        directional: dict mapping (bin1, bin2) -> {'right': n, 'left': n, 'below': n, 'above': n}

    The key insight: in a gradient flowing left-to-right from blue to pink,
    (blue, pink)['right'] >> (blue, pink)['left']
    This asymmetry reveals the gradient direction.
    """
    h, w = binned.shape[:2]
    directional = {}

    def add_directional(b1, b2, direction):
        """Add count: b2 is in 'direction' relative to b1"""
        if b1 == b2:
            return
        key = (b1, b2)
        if key not in directional:
            directional[key] = {
                'right': 0, 'left': 0, 'below': 0, 'above': 0,
                'down_right': 0, 'down_left': 0, 'up_right': 0, 'up_left': 0
            }
        directional[key][direction] += 1

    # Horizontal neighbors: pixel at (y, x) and (y, x+1)
    for y in range(h):
        for x in range(w - 1):
            b_left = tuple(binned[y, x])
            b_right = tuple(binned[y, x + 1])
            # b_right is to the RIGHT of b_left
            add_directional(b_left, b_right, 'right')
            # b_left is to the LEFT of b_right
            add_directional(b_right, b_left, 'left')

    # Vertical neighbors: pixel at (y, x) and (y+1, x)
    for y in range(h - 1):
        for x in range(w):
            b_top = tuple(binned[y, x])
            b_bottom = tuple(binned[y + 1, x])
            # b_bottom is BELOW b_top
            add_directional(b_top, b_bottom, 'below')
            # b_top is ABOVE b_bottom
            add_directional(b_bottom, b_top, 'above')

    # Diagonal neighbors (down-right): pixel at (y, x) and (y+1, x+1)
    for y in range(h - 1):
        for x in range(w - 1):
            b_top_left = tuple(binned[y, x])
            b_bottom_right = tuple(binned[y + 1, x + 1])
            # b_bottom_right is DOWN_RIGHT of b_top_left
            add_directional(b_top_left, b_bottom_right, 'down_right')
            # b_top_left is UP_LEFT of b_bottom_right
            add_directional(b_bottom_right, b_top_left, 'up_left')

    # Diagonal neighbors (down-left): pixel at (y, x) and (y+1, x-1)
    for y in range(h - 1):
        for x in range(1, w):
            b_top_right = tuple(binned[y, x])
            b_bottom_left = tuple(binned[y + 1, x - 1])
            # b_bottom_left is DOWN_LEFT of b_top_right
            add_directional(b_top_right, b_bottom_left, 'down_left')
            # b_top_right is UP_RIGHT of b_bottom_left
            add_directional(b_bottom_left, b_top_right, 'up_right')

    return directional


def compute_local_contrast(colors: list, adjacency: dict, scale: float = 3.0) -> dict:
    """
    Compute local contrast for each color - average LAB distance to spatial neighbors.

    High local contrast = color stands out from its immediate surroundings in the image.
    This captures boundary/edge colors.

    Returns:
        dict mapping bin -> local_contrast score
    """
    # Build neighbor lookup with LAB distances
    neighbor_distances = {b: [] for b in colors}

    for (b1, b2), count in adjacency.items():
        if b1 not in neighbor_distances or b2 not in neighbor_distances:
            continue

        lab1 = bin_to_lab(b1, scale)
        lab2 = bin_to_lab(b2, scale)
        dist = np.linalg.norm(lab1 - lab2)

        # Weight by adjacency count (more contact = more relevant)
        neighbor_distances[b1].extend([dist] * min(count, 100))
        neighbor_distances[b2].extend([dist] * min(count, 100))

    # Compute mean distance for each color
    local_contrast = {}
    for b in colors:
        if neighbor_distances[b]:
            local_contrast[b] = np.mean(neighbor_distances[b])
        else:
            local_contrast[b] = 0.0

    return local_contrast


def compute_multihop_contrast(colors: list, adjacency: dict, coverage: dict,
                               scale: float = 3.0, max_hops: int = 3) -> dict:
    """
    Compute contrast considering multi-hop neighbors in the adjacency graph.

    For each color, finds neighbors up to max_hops away and computes
    the maximum LAB distance to any reachable color, weighted by coverage.

    This captures that dark eyelashes are "near" skin tones even if they
    transition through intermediate colors.

    Args:
        colors: list of bin tuples
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        max_hops: how many steps to look in adjacency graph

    Returns:
        dict mapping bin -> max contrast to reachable high-coverage colors
    """
    # Build neighbor lookup
    neighbors = {b: set() for b in colors}
    for (b1, b2), count in adjacency.items():
        if b1 in neighbors and b2 in neighbors:
            neighbors[b1].add(b2)
            neighbors[b2].add(b1)

    # Get LAB values
    lab_values = {b: bin_to_lab(b, scale) for b in colors}

    # For weighting by coverage
    total_pixels = sum(coverage.values())
    cov_weight = {b: coverage.get(b, 0) / total_pixels for b in colors}

    def get_reachable(start, hops):
        """Get all colors reachable within n hops, with their hop distance."""
        reachable = {start: 0}
        frontier = {start}

        for hop in range(1, hops + 1):
            new_frontier = set()
            for node in frontier:
                for neighbor in neighbors.get(node, []):
                    if neighbor not in reachable:
                        reachable[neighbor] = hop
                        new_frontier.add(neighbor)
            frontier = new_frontier

        return reachable

    multihop_contrast = {}

    for b in colors:
        reachable = get_reachable(b, max_hops)
        lab_b = lab_values[b]

        # Find max contrast to high-coverage reachable colors
        max_contrast = 0
        weighted_sum = 0
        weight_total = 0

        for other, hops in reachable.items():
            if other == b:
                continue

            lab_other = lab_values[other]
            dist = np.linalg.norm(lab_b - lab_other)

            # Weight: prefer closer hops and higher coverage
            hop_weight = 1.0 / (hops + 0.5)  # 1-hop=0.67, 2-hop=0.4, 3-hop=0.29
            weight = hop_weight * (cov_weight[other] + 0.001)  # small baseline

            weighted_sum += dist * weight
            weight_total += weight

            # Track max contrast (unweighted)
            if dist > max_contrast:
                max_contrast = dist

        # Return coverage-weighted mean contrast to reachable colors
        if weight_total > 0:
            multihop_contrast[b] = weighted_sum / weight_total
        else:
            multihop_contrast[b] = 0.0

    return multihop_contrast


def compute_pixel_local_contrast(lab_image: np.ndarray, binned: np.ndarray,
                                  colors: list, radius: int = 5) -> dict:
    """
    Compute local contrast at the pixel level using convolution.

    For each pixel, measures LAB distance to the local mean in a neighborhood.
    Then aggregates by color bin.

    Args:
        lab_image: (H, W, 3) LAB image
        binned: (H, W, 3) quantized bin indices
        colors: list of bin tuples to compute contrast for
        radius: neighborhood radius for local mean (box filter size = 2*radius+1)

    Returns:
        dict mapping bin -> mean local contrast for pixels of that color
    """
    h, w = lab_image.shape[:2]
    size = 2 * radius + 1

    # Compute local mean for each LAB channel using box filter
    local_mean = np.zeros_like(lab_image, dtype=np.float32)
    for c in range(3):
        local_mean[:, :, c] = uniform_filter(lab_image[:, :, c].astype(np.float32), size=size)

    # Compute per-pixel contrast (LAB distance to local mean)
    diff = lab_image.astype(np.float32) - local_mean
    pixel_contrast = np.sqrt(np.sum(diff ** 2, axis=2))

    # Aggregate by color bin
    color_to_contrasts = {b: [] for b in colors}

    # Create bin lookup for fast matching
    bin_tuples = np.zeros((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            bin_tuples[y, x] = tuple(binned[y, x])

    # This is slow - let's use a faster approach
    # Build a mapping from bin tuple to mask
    contrast_sums = {}
    contrast_counts = {}

    for b in colors:
        # Create mask for this bin
        mask = (binned[:, :, 0] == b[0]) & (binned[:, :, 1] == b[1]) & (binned[:, :, 2] == b[2])
        if mask.sum() > 0:
            contrast_sums[b] = pixel_contrast[mask].sum()
            contrast_counts[b] = mask.sum()
        else:
            contrast_sums[b] = 0
            contrast_counts[b] = 0

    # Compute mean contrast per bin
    local_contrast = {}
    for b in colors:
        if contrast_counts.get(b, 0) > 0:
            local_contrast[b] = contrast_sums[b] / contrast_counts[b]
        else:
            local_contrast[b] = 0.0

    return local_contrast


def compute_global_contrast(colors: list, coverage: dict, scale: float = 3.0,
                            radius: float = 20.0) -> dict:
    """
    Compute global contrast for each color - inverse of nearby coverage in LAB space.

    High global contrast = color is rare/isolated in the overall palette.
    A gray in a mostly-red image has high global contrast.

    Args:
        colors: list of bin tuples
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        radius: LAB distance within which to sum coverage

    Returns:
        dict mapping bin -> global_contrast score (0-1, higher = more isolated)
    """
    total_pixels = sum(coverage.values())

    # Get LAB values for all colors
    lab_values = {b: bin_to_lab(b, scale) for b in colors}

    # For each color, compute coverage within radius
    nearby_coverage = {}

    for b in colors:
        lab_b = lab_values[b]
        nearby = 0

        for other, lab_other in lab_values.items():
            dist = np.linalg.norm(lab_b - lab_other)
            if dist <= radius:
                # Weight by proximity (closer = counts more)
                weight = 1.0 - (dist / radius)
                nearby += coverage.get(other, 0) * weight

        nearby_coverage[b] = nearby / total_pixels

    # Convert to contrast: inverse of density, normalized
    max_density = max(nearby_coverage.values()) if nearby_coverage else 1.0

    global_contrast = {}
    for b in colors:
        # Invert: low density -> high contrast
        density = nearby_coverage[b] / (max_density + 1e-10)
        global_contrast[b] = 1.0 - density

    return global_contrast


def compute_color_metrics(colors: list, adjacency: dict, coverage: dict,
                          scale: float = 3.0) -> dict:
    """
    Compute per-color metrics for the color space.

    Returns dict with per-color metrics:
    - coverage: fraction of image
    - local_contrast: stands out from spatial neighbors
    - global_contrast: stands out from overall palette
    - chroma: saturation level (sqrt(a² + b²))
    - hue: hue angle in degrees (0-360, via atan2(b, a))
    - lab: LAB coordinates
    """
    total_pixels = sum(coverage.values())

    local = compute_local_contrast(colors, adjacency, scale)
    global_c = compute_global_contrast(colors, coverage, scale)

    results = {}
    for b in colors:
        lab = bin_to_lab(b, scale)
        chroma = np.sqrt(lab[1]**2 + lab[2]**2)
        hue = np.degrees(np.arctan2(lab[2], lab[1])) % 360  # 0-360 degrees

        results[b] = {
            'coverage': coverage.get(b, 0) / total_pixels,
            'local_contrast': local.get(b, 0),
            'global_contrast': global_c.get(b, 0),
            'chroma': chroma,
            'hue': hue,
            'lab': lab
        }

    return results


def compute_spatial_coherence(binned: np.ndarray, colors: list,
                               coverage: dict) -> dict:
    """
    Compute spatial coherence for each color.

    Coherence measures how clustered vs scattered a color's pixels are.
    High coherence = pixels form one or few large blobs (signal)
    Low coherence = pixels scattered across many small blobs (noise/dither)

    Returns dict with per-color metrics:
    - coherence: largest_blob_pixels / total_pixels (0-1)
    - blob_count: number of connected components
    - largest_blob: pixel count of largest connected component
    """
    h, w = binned.shape[:2]
    results = {}

    for b in colors:
        # Create binary mask for this color
        mask = np.all(binned == b, axis=2)
        total = coverage.get(b, 0)

        if total == 0:
            results[b] = {
                'coherence': 0.0,
                'blob_count': 0,
                'largest_blob': 0
            }
            continue

        # Find connected components (8-connectivity)
        structure = np.ones((3, 3), dtype=int)
        labeled, num_blobs = label(mask, structure=structure)

        if num_blobs == 0:
            results[b] = {
                'coherence': 0.0,
                'blob_count': 0,
                'largest_blob': 0
            }
            continue

        # Count pixels in each blob
        blob_sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
        largest_blob = int(np.max(blob_sizes))

        results[b] = {
            'coherence': largest_blob / total,
            'blob_count': num_blobs,
            'largest_blob': largest_blob
        }

    return results


def identify_noise_colors(coherence: dict, coverage: dict,
                          threshold: float = 0.3) -> set:
    """
    Identify colors likely to be noise based on relative coherence.

    Uses median coherence as baseline - a color is noise if:
    - coherence < threshold * median_coherence

    This adapts to image characteristics:
    - Smooth photo: high median coherence, strict filter
    - Pollock painting: low median coherence, permissive filter

    Args:
        coherence: dict from compute_spatial_coherence
        coverage: dict of bin -> pixel count (for weighting)
        threshold: fraction of median below which a color is noise

    Returns:
        set of bin tuples identified as noise
    """
    if not coherence:
        return set()

    # Compute coverage-weighted median coherence
    # Weight by coverage so dominant colors influence the baseline more
    total_coverage = sum(coverage.values())
    weighted_coherences = []
    weights = []

    for b, stats in coherence.items():
        coh = stats['coherence']
        cov = coverage.get(b, 0) / total_coverage
        weighted_coherences.append(coh)
        weights.append(cov)

    # Sort by coherence to find weighted median
    sorted_pairs = sorted(zip(weighted_coherences, weights))
    cumsum = 0
    median_coherence = sorted_pairs[-1][0]  # fallback to max
    for coh, w in sorted_pairs:
        cumsum += w
        if cumsum >= 0.5:
            median_coherence = coh
            break

    # Identify noise: coherence significantly below median
    noise_threshold = threshold * median_coherence
    noise = set()

    for b, stats in coherence.items():
        if stats['coherence'] < noise_threshold:
            noise.add(b)

    return noise, median_coherence


def classify_color_scheme(color_metrics: dict, min_coverage: float = 0.001) -> dict:
    """
    Classify the color scheme of an image based on hue/chroma distribution.

    Uses relative thresholds based on the image's own chroma distribution,
    so it works on everything from grayscale to highly saturated images.

    Scheme types:
    - achromatic: no significant chromatic content
    - neutral_accent: mostly neutral with small chromatic accent(s)
    - monochromatic: single hue family
    - analogous: adjacent hues (<90° spread)
    - analogous_accent: analogous base + complementary accent
    - complementary: two hue clusters ~180° apart
    - triadic: three hue clusters ~120° apart
    - complex: doesn't fit simple categories

    Returns dict with:
    - scheme_type: classification label
    - chroma_profile: analysis of saturation distribution
    - chromatic_colors: colors above chroma threshold
    - neutral_colors: colors below chroma threshold
    - hue_clusters: detected hue groupings (if chromatic)
    - dominant_hue: main hue angle (if applicable)
    - accent_hue: outlier hue (if applicable)
    - lightness_range: (min_L, max_L) of significant colors
    """
    # Filter to significant colors
    significant = {b: m for b, m in color_metrics.items()
                   if m['coverage'] >= min_coverage}

    if not significant:
        return {
            'scheme_type': 'unknown',
            'chroma_profile': {'type': 'unknown'},
            'chromatic_colors': [],
            'neutral_colors': [],
            'hue_clusters': [],
            'lightness_range': (0, 100),
        }

    # Compute coverage-weighted chroma statistics
    total_coverage = sum(m['coverage'] for m in significant.values())
    chromas = []
    weights = []
    for b, m in significant.items():
        chromas.append(m['chroma'])
        weights.append(m['coverage'] / total_coverage)

    chromas = np.array(chromas)
    weights = np.array(weights)

    # Weighted percentiles for chroma
    sorted_idx = np.argsort(chromas)
    sorted_chromas = chromas[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)

    median_chroma = sorted_chromas[np.searchsorted(cumsum, 0.5)]
    p25_chroma = sorted_chromas[np.searchsorted(cumsum, 0.25)]
    p75_chroma = sorted_chromas[np.searchsorted(cumsum, 0.75)]
    max_chroma = np.max(chromas)

    # Classify chroma profile
    if median_chroma < 10 and max_chroma < 20:
        chroma_type = 'achromatic'
    elif median_chroma < 15:
        chroma_type = 'low_chroma'
    elif (p75_chroma - p25_chroma) > 25:
        chroma_type = 'mixed'
    else:
        chroma_type = 'chromatic'

    chroma_profile = {
        'type': chroma_type,
        'median': float(median_chroma),
        'max': float(max_chroma),
        'iqr': float(p75_chroma - p25_chroma),
    }

    # Adaptive threshold for chromatic vs neutral
    # Use 50% of median, but at least 10 (absolute floor for "has color")
    chroma_threshold = max(10, median_chroma * 0.5)
    chroma_profile['threshold'] = float(chroma_threshold)

    # Separate chromatic and neutral colors
    chromatic_colors = []
    neutral_colors = []
    for b, m in significant.items():
        if m['chroma'] >= chroma_threshold:
            chromatic_colors.append((b, m))
        else:
            neutral_colors.append((b, m))

    # Compute lightness range
    all_L = [m['lab'][0] for m in significant.values()]
    lightness_range = (float(min(all_L)), float(max(all_L)))

    # If no chromatic colors, it's achromatic
    if not chromatic_colors:
        return {
            'scheme_type': 'achromatic',
            'chroma_profile': chroma_profile,
            'chromatic_colors': [],
            'neutral_colors': [b for b, m in neutral_colors],
            'hue_clusters': [],
            'lightness_range': lightness_range,
        }

    # Cluster chromatic colors by hue using weighted circular histogram
    hue_clusters = _cluster_hues_circular(chromatic_colors)

    # Determine scheme type based on chroma profile and hue clusters
    scheme_type, dominant_hue, accent_hue = _classify_from_clusters(
        chroma_type, hue_clusters, chromatic_colors, neutral_colors
    )

    result = {
        'scheme_type': scheme_type,
        'chroma_profile': chroma_profile,
        'chromatic_colors': [b for b, m in chromatic_colors],
        'neutral_colors': [b for b, m in neutral_colors],
        'hue_clusters': hue_clusters,
        'lightness_range': lightness_range,
    }

    if dominant_hue is not None:
        result['dominant_hue'] = dominant_hue
    if accent_hue is not None:
        result['accent_hue'] = accent_hue

    return result


def _cluster_hues_circular(chromatic_colors: list) -> list[dict]:
    """
    Cluster hues using circular peak detection on coverage-weighted histogram.

    Returns list of clusters, each with:
    - center: hue angle (0-360)
    - spread: angular spread of cluster
    - coverage: total coverage of colors in cluster
    - colors: list of color bins in this cluster
    """
    if not chromatic_colors:
        return []

    # Build coverage-weighted circular histogram (36 bins of 10° each)
    n_bins = 36
    bin_width = 360 / n_bins
    histogram = np.zeros(n_bins)
    color_bins = [[] for _ in range(n_bins)]  # track which colors in each bin

    for b, m in chromatic_colors:
        hue = m['hue']
        coverage = m['coverage']
        bin_idx = int(hue / bin_width) % n_bins
        histogram[bin_idx] += coverage
        color_bins[bin_idx].append(b)

    # Smooth histogram to reduce noise (circular convolution)
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    smoothed = np.convolve(np.tile(histogram, 3), kernel, mode='same')[n_bins:2*n_bins]

    # Find peaks (local maxima)
    peaks = []
    for i in range(n_bins):
        prev_i = (i - 1) % n_bins
        next_i = (i + 1) % n_bins
        if smoothed[i] > smoothed[prev_i] and smoothed[i] > smoothed[next_i]:
            if smoothed[i] > 0.01:  # ignore tiny peaks
                peaks.append(i)

    if not peaks:
        # No clear peaks - treat all chromatic as one cluster
        total_cov = sum(m['coverage'] for b, m in chromatic_colors)
        # Circular mean of hues
        sin_sum = sum(m['coverage'] * np.sin(np.radians(m['hue']))
                      for b, m in chromatic_colors)
        cos_sum = sum(m['coverage'] * np.cos(np.radians(m['hue']))
                      for b, m in chromatic_colors)
        mean_hue = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

        return [{
            'center': float(mean_hue),
            'spread': 360.0,  # unknown spread
            'coverage': float(total_cov),
            'colors': [b for b, m in chromatic_colors]
        }]

    # Assign colors to nearest peak (circular distance)
    clusters = {peak: {'colors': [], 'coverage': 0.0} for peak in peaks}

    for b, m in chromatic_colors:
        hue = m['hue']
        coverage = m['coverage']

        # Find nearest peak
        min_dist = float('inf')
        nearest_peak = peaks[0]
        for peak in peaks:
            peak_hue = (peak + 0.5) * bin_width
            dist = min(abs(hue - peak_hue), 360 - abs(hue - peak_hue))
            if dist < min_dist:
                min_dist = dist
                nearest_peak = peak

        clusters[nearest_peak]['colors'].append((b, m))
        clusters[nearest_peak]['coverage'] += coverage

    # Convert to output format with proper hue centers
    result = []
    for peak, data in clusters.items():
        if not data['colors']:
            continue

        # Compute circular mean hue for this cluster
        sin_sum = sum(m['coverage'] * np.sin(np.radians(m['hue']))
                      for b, m in data['colors'])
        cos_sum = sum(m['coverage'] * np.cos(np.radians(m['hue']))
                      for b, m in data['colors'])
        center = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

        # Compute spread (max angular distance from center)
        max_dist = 0
        for b, m in data['colors']:
            dist = min(abs(m['hue'] - center), 360 - abs(m['hue'] - center))
            max_dist = max(max_dist, dist)

        result.append({
            'center': float(center),
            'spread': float(max_dist * 2),  # diameter
            'coverage': float(data['coverage']),
            'colors': [b for b, m in data['colors']]
        })

    # Sort by coverage (dominant first)
    result.sort(key=lambda c: c['coverage'], reverse=True)
    return result


def _classify_from_clusters(chroma_type: str, hue_clusters: list,
                            chromatic_colors: list, neutral_colors: list) -> tuple:
    """
    Determine scheme type from chroma profile and hue clusters.

    Returns (scheme_type, dominant_hue, accent_hue)
    """
    n_clusters = len(hue_clusters)
    total_chromatic = sum(m['coverage'] for b, m in chromatic_colors)
    total_neutral = sum(m['coverage'] for b, m in neutral_colors)
    chromatic_ratio = total_chromatic / (total_chromatic + total_neutral) if (total_chromatic + total_neutral) > 0 else 0

    # No clusters = achromatic (shouldn't happen if we have chromatic colors)
    if n_clusters == 0:
        return 'achromatic', None, None

    dominant = hue_clusters[0]
    dominant_hue = dominant['center']

    # Single cluster
    if n_clusters == 1:
        spread = dominant['spread']
        if spread < 60:
            if chromatic_ratio < 0.3:
                return 'neutral_accent', dominant_hue, None
            return 'monochromatic', dominant_hue, None
        elif spread < 90:
            return 'analogous', dominant_hue, None
        else:
            return 'complex', dominant_hue, None

    # Two clusters
    if n_clusters == 2:
        second = hue_clusters[1]
        second_hue = second['center']

        # Angular distance between clusters
        angular_dist = min(abs(dominant_hue - second_hue),
                          360 - abs(dominant_hue - second_hue))

        # Coverage ratio of smaller to larger
        coverage_ratio = second['coverage'] / dominant['coverage'] if dominant['coverage'] > 0 else 0

        # Small secondary cluster = accent
        if coverage_ratio < 0.25:
            if angular_dist > 120:
                return 'analogous_accent', dominant_hue, second_hue
            else:
                return 'analogous', dominant_hue, None

        # Two roughly equal clusters
        if angular_dist > 150:
            return 'complementary', dominant_hue, second_hue
        elif angular_dist < 90:
            return 'analogous', dominant_hue, None
        else:
            # Split-complementary territory
            return 'split_complementary', dominant_hue, second_hue

    # Three clusters
    if n_clusters == 3:
        # Check if roughly triadic (120° apart)
        hues = sorted([c['center'] for c in hue_clusters])
        gaps = [(hues[1] - hues[0]) % 360,
                (hues[2] - hues[1]) % 360,
                (hues[0] - hues[2]) % 360]

        # Triadic if gaps are roughly equal (100-140° each)
        if all(80 < g < 160 for g in gaps):
            return 'triadic', dominant_hue, None

        # Otherwise check for dominant + accents
        dominant_coverage = dominant['coverage']
        other_coverage = sum(c['coverage'] for c in hue_clusters[1:])
        if dominant_coverage > other_coverage * 2:
            return 'analogous_accent', dominant_hue, hue_clusters[1]['center']

    # Four or more clusters, or complex patterns
    return 'complex', dominant_hue, None


def analyze_gradient_flow(directional: dict, colors: list, coverage: dict,
                          scale: float = 3.0, min_flow: int = 50) -> dict:
    """
    Analyze directional adjacency to find gradient flow patterns.

    For each color pair, computes:
    - horizontal_flow: right - left (positive = flows right)
    - vertical_flow: below - above (positive = flows down)
    - diag_dr_flow: down_right - up_left (positive = flows toward bottom-right)
    - diag_dl_flow: down_left - up_right (positive = flows toward bottom-left)

    Returns dict with flow analysis for significant pairs.
    """
    flow_analysis = {}

    for (b1, b2), dirs in directional.items():
        total = (dirs['right'] + dirs['left'] + dirs['below'] + dirs['above'] +
                 dirs['down_right'] + dirs['down_left'] + dirs['up_right'] + dirs['up_left'])
        if total < min_flow:
            continue

        h_flow = dirs['right'] - dirs['left']
        v_flow = dirs['below'] - dirs['above']
        dr_flow = dirs['down_right'] - dirs['up_left']  # main diagonal
        dl_flow = dirs['down_left'] - dirs['up_right']  # anti-diagonal

        # Compute flow strength (how asymmetric)
        h_total = dirs['right'] + dirs['left']
        v_total = dirs['below'] + dirs['above']
        dr_total = dirs['down_right'] + dirs['up_left']
        dl_total = dirs['down_left'] + dirs['up_right']

        h_asymmetry = abs(h_flow) / (h_total + 1) if h_total > 0 else 0
        v_asymmetry = abs(v_flow) / (v_total + 1) if v_total > 0 else 0
        dr_asymmetry = abs(dr_flow) / (dr_total + 1) if dr_total > 0 else 0
        dl_asymmetry = abs(dl_flow) / (dl_total + 1) if dl_total > 0 else 0

        flow_analysis[(b1, b2)] = {
            'h_flow': h_flow,
            'v_flow': v_flow,
            'dr_flow': dr_flow,
            'dl_flow': dl_flow,
            'h_asymmetry': h_asymmetry,
            'v_asymmetry': v_asymmetry,
            'dr_asymmetry': dr_asymmetry,
            'dl_asymmetry': dl_asymmetry,
            'total': total,
            'dirs': dirs
        }

    return flow_analysis


def find_flow_gradients(colors: list, directional: dict, coverage: dict,
                        scale: float = 3.0, min_chain_length: int = 3,
                        min_asymmetry: float = 0.3,
                        color_metrics: dict = None) -> list[dict]:
    """
    Find gradients by following directional flow through the adjacency graph.

    A gradient is a chain where colors consistently flow in one direction:
    - Horizontal gradient: each color has the next color predominantly to its right (or left)
    - Vertical gradient: each color has the next color predominantly below (or above)

    Starting points are selected by unified score: coverage + contrast metrics.
    Both high-coverage colors and high-contrast colors compete for gradient seeds.
    Flow edges are weighted by coherence (noisy transitions are deprioritized).

    Args:
        colors: list of bin tuples
        directional: directional adjacency dict
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        min_chain_length: minimum chain length for valid gradients
        min_asymmetry: minimum flow asymmetry to follow an edge
        color_metrics: optional dict with per-color metrics (coherence, contrast, etc.)
            (if None, falls back to coverage-only ranking)

    Returns list of gradient dicts with chain, direction, and metadata.
    """
    # Analyze flow patterns
    flow = analyze_gradient_flow(directional, colors, coverage, scale)

    # Build directed graph based on flow
    # Edge from A to B if B is predominantly in one direction from A
    # Weight edges by coherence: noisy transitions are less meaningful
    flow_graph = {b: {
        'right': [], 'left': [], 'below': [], 'above': [],
        'down_right': [], 'down_left': [], 'up_right': [], 'up_left': []
    } for b in colors}

    def get_coherence(b):
        """Get coherence for a color, defaulting to 1.0 if not available."""
        if color_metrics and b in color_metrics:
            return color_metrics[b].get('coherence', 1.0)
        return 1.0

    for (b1, b2), analysis in flow.items():
        if b1 not in flow_graph or b2 not in flow_graph:
            continue

        # Weight by geometric mean of coherences
        # Transitions between coherent colors are more meaningful
        coh1, coh2 = get_coherence(b1), get_coherence(b2)
        coherence_weight = np.sqrt(coh1 * coh2)

        # Check horizontal flow
        if analysis['h_asymmetry'] > min_asymmetry:
            if analysis['h_flow'] > 0:
                # b2 is predominantly RIGHT of b1
                weighted_flow = analysis['h_flow'] * coherence_weight
                flow_graph[b1]['right'].append((b2, weighted_flow, analysis['total']))
            else:
                # b2 is predominantly LEFT of b1
                weighted_flow = -analysis['h_flow'] * coherence_weight
                flow_graph[b1]['left'].append((b2, weighted_flow, analysis['total']))

        # Check vertical flow
        if analysis['v_asymmetry'] > min_asymmetry:
            if analysis['v_flow'] > 0:
                # b2 is predominantly BELOW b1
                weighted_flow = analysis['v_flow'] * coherence_weight
                flow_graph[b1]['below'].append((b2, weighted_flow, analysis['total']))
            else:
                # b2 is predominantly ABOVE b1
                weighted_flow = -analysis['v_flow'] * coherence_weight
                flow_graph[b1]['above'].append((b2, weighted_flow, analysis['total']))

        # Check main diagonal flow (down-right / up-left)
        if analysis['dr_asymmetry'] > min_asymmetry:
            if analysis['dr_flow'] > 0:
                # b2 is predominantly DOWN_RIGHT of b1
                weighted_flow = analysis['dr_flow'] * coherence_weight
                flow_graph[b1]['down_right'].append((b2, weighted_flow, analysis['total']))
            else:
                # b2 is predominantly UP_LEFT of b1
                weighted_flow = -analysis['dr_flow'] * coherence_weight
                flow_graph[b1]['up_left'].append((b2, weighted_flow, analysis['total']))

        # Check anti-diagonal flow (down-left / up-right)
        if analysis['dl_asymmetry'] > min_asymmetry:
            if analysis['dl_flow'] > 0:
                # b2 is predominantly DOWN_LEFT of b1
                weighted_flow = analysis['dl_flow'] * coherence_weight
                flow_graph[b1]['down_left'].append((b2, weighted_flow, analysis['total']))
            else:
                # b2 is predominantly UP_RIGHT of b1
                weighted_flow = -analysis['dl_flow'] * coherence_weight
                flow_graph[b1]['up_right'].append((b2, weighted_flow, analysis['total']))

    # Sort neighbors by coherence-weighted flow strength
    for b in flow_graph:
        for direction in flow_graph[b]:
            flow_graph[b][direction].sort(key=lambda x: x[1], reverse=True)

    def follow_flow(start, direction):
        """Follow flow in one direction to build a gradient chain."""
        chain = [start]
        current = start
        visited = {start}

        while True:
            candidates = flow_graph[current][direction]
            # Find best unvisited neighbor
            next_color = None
            for neighbor, strength, total in candidates:
                if neighbor not in visited:
                    next_color = neighbor
                    break

            if next_color is None:
                break

            chain.append(next_color)
            visited.add(next_color)
            current = next_color

        return chain

    # Compute unified starting score: coverage + contrast
    # Both high coverage and high contrast make a color important as gradient seed
    total_pixels = sum(coverage.values())

    def compute_starting_score(b):
        """Unified score combining coverage and contrast metrics."""
        # Normalize coverage to 0-1
        cov_normalized = coverage.get(b, 0) / total_pixels

        # Get contrast metrics (if available)
        if color_metrics and b in color_metrics:
            stats = color_metrics[b]
            # Normalize multihop contrast (typically 10-25 range) to ~0-1
            multihop = stats.get('multihop_contrast', 0) / 25.0
            # Global contrast already 0-1
            global_c = stats.get('global_contrast', 0)
            # Combined contrast: weight multihop more (it captures local importance)
            contrast_normalized = multihop * 0.7 + global_c * 0.3
        else:
            contrast_normalized = 0

        # Unified score: coverage is primary, contrast is secondary boost
        # Scale coverage to be competitive (max ~1.1 for 11% coverage)
        # Contrast adds a smaller boost (max ~0.15) so it can nudge rankings
        # but not completely override coverage
        cov_scaled = cov_normalized * 10  # 11% -> 1.1, 0.5% -> 0.05
        contrast_boost = contrast_normalized * 0.15  # max 0.15 boost

        return cov_scaled + contrast_boost

    # Sort all colors by unified starting score
    scored_colors = [(b, compute_starting_score(b)) for b in colors]
    scored_colors.sort(key=lambda x: x[1], reverse=True)

    all_gradients = []

    # Single unified loop - try top colors by combined score
    # Colors can appear in multiple gradients (no 'used' set)
    for direction in ['right', 'left', 'below', 'above',
                      'down_right', 'down_left', 'up_right', 'up_left']:
        for start, start_score in scored_colors[:40]:  # Try more candidates
            chain = follow_flow(start, direction)

            if len(chain) >= min_chain_length:
                chain_cov = sum(coverage.get(b, 0) for b in chain)

                # Compute LAB bounds for transition comparison
                labs = [bin_to_lab(b, scale) for b in chain]
                lab_array = np.array(labs)
                l_min, l_max = lab_array[:, 0].min(), lab_array[:, 0].max()
                a_min, a_max = lab_array[:, 1].min(), lab_array[:, 1].max()
                b_min, b_max = lab_array[:, 2].min(), lab_array[:, 2].max()

                all_gradients.append({
                    'chain': chain,
                    'direction': direction,
                    'coverage': chain_cov,
                    'l_range': l_max - l_min,
                    'a_range': a_max - a_min,
                    'b_range': b_max - b_min,
                    'l_bounds': (l_min, l_max),
                    'a_bounds': (a_min, a_max),
                    'b_bounds': (b_min, b_max),
                    'score': len(chain) * chain_cov,
                    'starting_score': start_score
                })

    # Sort by score (coverage-based gradients will rank higher)
    all_gradients.sort(key=lambda x: x['score'], reverse=True)

    def bounds_overlap(bounds1, bounds2):
        """Compute overlap ratio: how much of bounds1 is inside bounds2."""
        min1, max1 = bounds1
        min2, max2 = bounds2
        range1 = max1 - min1
        if range1 < 0.1:  # Avoid division by tiny ranges
            return 1.0 if min2 <= min1 <= max2 else 0.0
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        overlap = max(0, overlap_max - overlap_min)
        return overlap / range1

    # Deduplicate by LAB transition: drop gradients whose LAB range
    # is mostly contained within an existing gradient's range
    final_gradients = []
    for grad in all_gradients:
        is_redundant = False
        for existing in final_gradients:
            # Check if this gradient's LAB bounds are mostly inside existing's
            l_overlap = bounds_overlap(grad['l_bounds'], existing['l_bounds'])
            a_overlap = bounds_overlap(grad['a_bounds'], existing['a_bounds'])
            b_overlap = bounds_overlap(grad['b_bounds'], existing['b_bounds'])

            # If all three dimensions are ≥50% contained, it's redundant
            if l_overlap >= 0.5 and a_overlap >= 0.5 and b_overlap >= 0.5:
                is_redundant = True
                break
        if not is_redundant:
            final_gradients.append(grad)

    return final_gradients


def build_adjacency_matrix(adjacency: dict, coverage: dict,
                           min_coverage: float = 0.0001,
                           scale: float = 3.0,
                           lab_weight_sigma: float = 15.0) -> tuple[np.ndarray, list]:
    """
    Build adjacency matrix from graph, weighted by LAB similarity.

    Colors that are both spatially adjacent AND close in LAB space
    get stronger connections (smooth gradients). Adjacent but dissimilar
    colors (hard edges) get weaker connections.

    Args:
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        min_coverage: minimum coverage to include a color
        scale: JND scale factor for bin_to_lab
        lab_weight_sigma: sigma for Gaussian weighting by LAB distance
            - smaller = stronger preference for similar colors
            - larger = more uniform weighting

    Returns:
        matrix: NxN adjacency matrix (normalized by row)
        colors: list of bin tuples in matrix order
    """
    total_pixels = sum(coverage.values())

    # Filter to colors with minimum coverage
    colors = [b for b, c in coverage.items()
              if c / total_pixels >= min_coverage]
    color_to_idx = {b: i for i, b in enumerate(colors)}
    n = len(colors)

    print(f"Building {n}x{n} adjacency matrix (LAB-weighted, sigma={lab_weight_sigma})...")

    # Build matrix with LAB-weighted adjacency
    matrix = np.zeros((n, n), dtype=np.float32)
    for (b1, b2), count in adjacency.items():
        if b1 in color_to_idx and b2 in color_to_idx:
            i, j = color_to_idx[b1], color_to_idx[b2]

            # Calculate LAB distance
            lab1 = bin_to_lab(b1, scale)
            lab2 = bin_to_lab(b2, scale)
            lab_dist = np.linalg.norm(lab1 - lab2)

            # Weight by Gaussian: similar colors get weight ~1, dissimilar get weight ~0
            similarity = np.exp(-lab_dist / lab_weight_sigma)

            # Combined weight: count * similarity
            weight = count * similarity

            matrix[i, j] = weight
            matrix[j, i] = weight

    # Row-normalize (each color's adjacency vector sums to 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums

    return matrix, colors


def embed_adjacency_space(matrix: np.ndarray, method: str = 'pca',
                          n_components: int = 3) -> np.ndarray:
    """
    Embed adjacency vectors into lower-dimensional space.

    Args:
        matrix: NxN adjacency matrix
        method: 'pca' or 'tsne'
        n_components: dimensions to reduce to

    Returns:
        embedding: Nx(n_components) array
    """
    if method == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(matrix)
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    elif method == 'tsne':
        tsne = TSNE(n_components=min(n_components, 3), perplexity=30, random_state=42)
        embedding = tsne.fit_transform(matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding


def find_directional_gradients(embedding: np.ndarray, colors: list, adjacency: dict,
                                coverage: dict, scale: float = 3.0,
                                n_bins: int = 36, min_peak_prominence: float = 0.3,
                                min_colors_per_ray: int = 4) -> list[list[tuple]]:
    """
    Find gradients by detecting directional rays from the centroid.

    In adjacency space, gradients appear as rays emanating from a central cluster.
    Colors at the periphery are gradient endpoints; the center contains colors
    with mixed adjacency patterns.

    Args:
        embedding: Nx3 array of embedded coordinates
        colors: list of bin tuples
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        n_bins: number of angular bins for histogram (36 = 10 degree resolution)
        min_peak_prominence: minimum prominence for peak detection (relative to max)
        min_colors_per_ray: minimum colors to form a valid ray

    Returns:
        List of gradient chains, each ordered from center outward
    """
    n = len(colors)
    color_to_idx = {b: i for i, b in enumerate(colors)}

    # Build neighbor lookup
    neighbors = {b: set() for b in colors}
    for (b1, b2), count in adjacency.items():
        if b1 in neighbors and b2 in neighbors:
            neighbors[b1].add(b2)
            neighbors[b2].add(b1)

    # Step 1: Compute centroid (coverage-weighted)
    total_cov = sum(coverage.get(b, 1) for b in colors)
    weights = np.array([coverage.get(b, 1) / total_cov for b in colors])
    centroid = np.average(embedding, axis=0, weights=weights)

    print(f"Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

    # Step 2: Compute direction and distance from centroid for each color
    vectors = embedding - centroid
    distances = np.linalg.norm(vectors, axis=1)

    # Use PC1-PC2 plane for primary angular analysis
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # -pi to pi

    # Step 3: Build angular histogram to find ray directions
    # Weight by distance (peripheral colors define rays better)
    hist_weights = distances * weights  # distance * coverage

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(angles, bins=bin_edges, weights=hist_weights)

    # Smooth histogram to reduce noise
    hist_smooth = gaussian_filter1d(hist, sigma=1.5, mode='wrap')

    # Normalize for peak detection
    hist_norm = hist_smooth / (hist_smooth.max() + 1e-10)

    # Step 4: Find peaks (ray directions)
    # Handle wrap-around by extending the histogram
    hist_extended = np.concatenate([hist_norm, hist_norm, hist_norm])
    peaks, properties = find_peaks(hist_extended,
                                   prominence=min_peak_prominence,
                                   distance=2)

    # Filter to middle section and convert back to original indices
    valid_peaks = []
    for p in peaks:
        original_idx = p % n_bins
        if original_idx not in valid_peaks:
            valid_peaks.append(original_idx)

    # Convert peak indices to angles
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_angles = [bin_centers[p] for p in valid_peaks]

    print(f"Found {len(peak_angles)} directional rays:")
    for i, angle in enumerate(peak_angles):
        print(f"  Ray {i+1}: {np.degrees(angle):.0f}°")

    # Step 5: Assign colors to nearest ray
    ray_assignments = []
    for i, angle in enumerate(angles):
        # Find nearest peak angle (handling wrap-around)
        best_ray = 0
        best_dist = float('inf')
        for j, peak_angle in enumerate(peak_angles):
            # Angular distance with wrap-around
            diff = abs(angle - peak_angle)
            diff = min(diff, 2*np.pi - diff)
            if diff < best_dist:
                best_dist = diff
                best_ray = j
        ray_assignments.append(best_ray)

    # Step 6: For each ray, order colors by distance from centroid
    rays = []
    for ray_idx in range(len(peak_angles)):
        # Get colors in this ray
        ray_colors = [(colors[i], distances[i], coverage.get(colors[i], 0))
                      for i in range(n) if ray_assignments[i] == ray_idx]

        if len(ray_colors) < min_colors_per_ray:
            continue

        # Sort by distance from centroid (center to periphery)
        ray_colors.sort(key=lambda x: x[1])

        # Extract just the color tuples
        ordered_colors = [c for c, d, cov in ray_colors]

        rays.append({
            'angle': peak_angles[ray_idx],
            'colors': ordered_colors,
            'distances': [d for c, d, cov in ray_colors],
            'coverage': sum(cov for c, d, cov in ray_colors)
        })

    # Step 7: Validate connectivity - build connected chains within each ray
    validated_chains = []

    for ray in rays:
        ordered = ray['colors']

        # Find longest connected subsequence
        # Use greedy: start from center, extend outward following adjacency
        best_chain = []

        for start_idx in range(min(5, len(ordered))):  # Try starting from first few
            chain = [ordered[start_idx]]
            used = {ordered[start_idx]}

            # Extend outward (increasing distance)
            for color in ordered[start_idx + 1:]:
                # Check if connected to any color in chain
                if any(color in neighbors[c] for c in chain[-3:]):  # Check last 3
                    chain.append(color)
                    used.add(color)

            if len(chain) > len(best_chain):
                best_chain = chain

        if len(best_chain) >= min_colors_per_ray:
            validated_chains.append(best_chain)
            ray_cov = sum(coverage.get(b, 0) for b in best_chain)
            print(f"  Ray at {np.degrees(ray['angle']):.0f}°: {len(best_chain)} connected colors, {ray_cov} pixels")

    # Sort by coverage
    validated_chains.sort(key=lambda c: sum(coverage.get(b, 0) for b in c), reverse=True)

    return validated_chains


def find_graph_gradients(colors: list, adjacency: dict, coverage: dict,
                         scale: float = 3.0, min_chain_length: int = 5,
                         max_lab_step: float = 20.0) -> list[list[tuple]]:
    """
    Find gradients by tracing smooth paths through the adjacency graph.

    A gradient is a path where:
    1. Consecutive colors are spatially adjacent (connected in graph)
    2. LAB values change smoothly (small delta per step)
    3. Progression is monotonic in at least one LAB dimension

    Args:
        colors: list of bin tuples
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        min_chain_length: minimum colors to form a valid gradient
        max_lab_step: maximum LAB distance per step

    Returns:
        List of gradient chains, sorted by quality (length * coverage)
    """
    # Build neighbor lookup with edge weights
    neighbors = {b: {} for b in colors}  # {color: {neighbor: weight}}

    for (b1, b2), count in adjacency.items():
        if b1 not in neighbors or b2 not in neighbors:
            continue

        # Compute LAB distance
        lab1 = bin_to_lab(b1, scale)
        lab2 = bin_to_lab(b2, scale)
        lab_dist = np.linalg.norm(lab1 - lab2)

        # Skip edges with large color jumps (not gradients)
        if lab_dist > max_lab_step:
            continue

        # Weight: prefer high adjacency count and small LAB distance
        # smoothness = count / (lab_dist + 1)
        weight = count

        neighbors[b1][b2] = {'count': count, 'lab_dist': lab_dist, 'weight': weight}
        neighbors[b2][b1] = {'count': count, 'lab_dist': lab_dist, 'weight': weight}

    # Get LAB values for all colors
    lab_values = {b: bin_to_lab(b, scale) for b in colors}

    def find_monotonic_path(start, lab_dim, direction):
        """
        Extend a path in one direction, requiring monotonic change in lab_dim.
        direction: +1 for increasing, -1 for decreasing
        """
        path = [start]
        current = start

        while True:
            current_lab = lab_values[current][lab_dim]

            # Find best neighbor that continues monotonic progression
            best_next = None
            best_score = -1

            for neighbor, edge in neighbors[current].items():
                if neighbor in path:
                    continue

                neighbor_lab = lab_values[neighbor][lab_dim]
                delta = (neighbor_lab - current_lab) * direction

                # Must progress in the right direction (or stay same)
                if delta < -0.5:  # Small tolerance for noise
                    continue

                # Score: prefer larger steps (more gradient) but smooth
                score = edge['count'] * (1 + delta / 10)

                if score > best_score:
                    best_score = score
                    best_next = neighbor

            if best_next is None:
                break

            path.append(best_next)
            current = best_next

        return path

    def extend_bidirectional(start, lab_dim):
        """Extend path in both directions along lab_dim."""
        # Extend in increasing direction
        forward = find_monotonic_path(start, lab_dim, +1)
        # Extend in decreasing direction
        backward = find_monotonic_path(start, lab_dim, -1)

        # Combine: backward (reversed, excluding start) + forward
        if len(backward) > 1:
            full_path = backward[::-1][:-1] + forward
        else:
            full_path = forward

        return full_path

    # Find gradient chains starting from high-coverage colors
    sorted_colors = sorted(colors, key=lambda b: coverage.get(b, 0), reverse=True)

    all_chains = []
    used_colors = set()

    # Try each LAB dimension (L=lightness, a=green-red, b=blue-yellow)
    for lab_dim, dim_name in [(0, 'L'), (1, 'a'), (2, 'b')]:
        for start in sorted_colors[:50]:  # Try top 50 colors as starting points
            if start in used_colors:
                continue

            chain = extend_bidirectional(start, lab_dim)

            if len(chain) >= min_chain_length:
                # Check that chain actually spans a meaningful range
                chain_labs = [lab_values[b][lab_dim] for b in chain]
                lab_range = max(chain_labs) - min(chain_labs)

                if lab_range < 10:  # Skip if not much color change
                    continue

                chain_cov = sum(coverage.get(b, 0) for b in chain)

                all_chains.append({
                    'chain': chain,
                    'dim': dim_name,
                    'range': lab_range,
                    'coverage': chain_cov,
                    'score': len(chain) * chain_cov
                })

                # Mark colors as used (but allow overlap between dimensions)
                if lab_dim == 0:  # Only mark for L dimension to allow a/b to reuse
                    used_colors.update(chain)

    # Sort by score and deduplicate
    all_chains.sort(key=lambda x: x['score'], reverse=True)

    # Remove chains that are subsets of better chains
    final_chains = []
    for chain_info in all_chains:
        chain_set = set(chain_info['chain'])

        # Check if >70% overlap with existing chain
        is_duplicate = False
        for existing in final_chains:
            existing_set = set(existing)
            overlap = len(chain_set & existing_set)
            if overlap > 0.7 * len(chain_set):
                is_duplicate = True
                break

        if not is_duplicate:
            final_chains.append(chain_info['chain'])
            print(f"  {chain_info['dim']}-gradient: {len(chain_info['chain'])} colors, "
                  f"range={chain_info['range']:.0f}, cov={chain_info['coverage']}")

    return final_chains


def visualize_directional_analysis(embedding: np.ndarray, colors: list,
                                    coverage: dict, chains: list[list[tuple]],
                                    output_path: str, scale: float = 3.0):
    """
    Visualize the directional ray analysis with rays highlighted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    total_pixels = sum(coverage.values())

    # Compute centroid
    weights = np.array([coverage.get(b, 1) for b in colors])
    weights = weights / weights.sum()
    centroid = np.average(embedding, axis=0, weights=weights)

    # Get RGB colors and sizes
    rgbs = []
    sizes = []
    for b in colors:
        lab = bin_to_lab(b, scale)
        rgb = lab_to_rgb(lab.reshape(1, -1))[0] / 255.0
        rgbs.append(rgb)
        sizes.append(coverage.get(b, 0) / total_pixels * 5000 + 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Assign colors to chains for highlighting
    color_to_chain = {}
    chain_colors_plt = plt.cm.Set1(np.linspace(0, 1, max(len(chains), 1)))
    for chain_idx, chain in enumerate(chains):
        for b in chain:
            color_to_chain[b] = chain_idx

    for ax_idx, (pc_x, pc_y, title) in enumerate([(0, 1, 'PC1 vs PC2'), (0, 2, 'PC1 vs PC3')]):
        ax = axes[ax_idx]

        # Plot all colors
        ax.scatter(embedding[:, pc_x], embedding[:, pc_y], c=rgbs, s=sizes, alpha=0.5)

        # Mark centroid
        ax.scatter([centroid[pc_x]], [centroid[pc_y]], c='black', s=100, marker='x', linewidths=2)

        # Draw rays for each chain
        for chain_idx, chain in enumerate(chains[:8]):  # Max 8 rays
            if not chain:
                continue

            # Get chain endpoints
            chain_indices = [colors.index(b) for b in chain if b in colors]
            if len(chain_indices) < 2:
                continue

            # Draw arrow from centroid toward chain direction
            chain_center = np.mean(embedding[chain_indices], axis=0)
            direction = chain_center - centroid
            direction = direction / (np.linalg.norm(direction) + 1e-10) * 0.15

            ax.annotate('', xy=(centroid[pc_x] + direction[pc_x], centroid[pc_y] + direction[pc_y]),
                       xytext=(centroid[pc_x], centroid[pc_y]),
                       arrowprops=dict(arrowstyle='->', color=chain_colors_plt[chain_idx], lw=2))

            # Highlight chain colors with edge
            for b in chain:
                if b in colors:
                    idx = colors.index(b)
                    ax.scatter([embedding[idx, pc_x]], [embedding[idx, pc_y]],
                             facecolors='none', edgecolors=chain_colors_plt[chain_idx],
                             s=sizes[idx]*1.5, linewidths=2, alpha=0.8)

        ax.set_xlabel(f'PC{pc_x+1}')
        ax.set_ylabel(f'PC{pc_y+1}')
        ax.set_title(f'Directional Analysis ({title})')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved directional analysis to {output_path}")


def visualize_embedding(embedding: np.ndarray, colors: list, coverage: dict,
                        output_path: str, scale: float = 3.0):
    """
    Visualize colors in their embedded adjacency space.

    Creates a scatter plot where position = embedding coordinates,
    color = actual RGB color, size = coverage.
    """
    import matplotlib.pyplot as plt

    total_pixels = sum(coverage.values())

    # Get RGB colors and sizes
    rgbs = []
    sizes = []
    for b in colors:
        lab = bin_to_lab(b, scale)
        rgb = lab_to_rgb(lab.reshape(1, -1))[0] / 255.0
        rgbs.append(rgb)
        sizes.append(coverage.get(b, 0) / total_pixels * 5000 + 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot PC1 vs PC2
    ax = axes[0]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=rgbs, s=sizes, alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Adjacency Space (PC1 vs PC2)')
    ax.set_aspect('equal')

    # Plot PC1 vs PC3 if available
    if embedding.shape[1] >= 3:
        ax = axes[1]
        ax.scatter(embedding[:, 0], embedding[:, 2], c=rgbs, s=sizes, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC3')
        ax.set_title('Adjacency Space (PC1 vs PC3)')
        ax.set_aspect('equal')
    else:
        ax = axes[1]
        ax.scatter(embedding[:, 0], embedding[:, 1], c=rgbs, s=sizes, alpha=0.7)
        ax.set_title('(Same as left)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved embedding visualization to {output_path}")


def find_gradient_paths(embedding: np.ndarray, colors: list, adjacency: dict,
                        coverage: dict, scale: float = 3.0) -> list[list[tuple]]:
    """
    Find gradient paths by tracing connected colors through embedding space.

    Uses shortest path between PC1 extremes to find the main gradient,
    then greedily extracts remaining chains.
    """
    from collections import deque

    color_to_idx = {b: i for i, b in enumerate(colors)}
    n = len(colors)

    # Build neighbor lookup from adjacency
    neighbors = {b: set() for b in colors}
    for (b1, b2), count in adjacency.items():
        if b1 in neighbors and b2 in neighbors:
            neighbors[b1].add(b2)
            neighbors[b2].add(b1)

    chains = []
    used = set()

    # Find main gradient: shortest path from PC1 min to PC1 max
    pc1_values = embedding[:, 0]
    start_idx = np.argmin(pc1_values)
    end_idx = np.argmax(pc1_values)
    start_color = colors[start_idx]
    end_color = colors[end_idx]

    print(f"Finding path from PC1 min ({pc1_values[start_idx]:.2f}) to max ({pc1_values[end_idx]:.2f})")

    # BFS to find shortest path
    def bfs_path(start, end):
        if start == end:
            return [start]
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            for neighbor in neighbors[current]:
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    main_path = bfs_path(start_color, end_color)
    if main_path:
        print(f"Main gradient path: {len(main_path)} colors")
        chains.append(main_path)
        used.update(main_path)

    # Try to build longest connected path along PC1
    def longest_pc1_path():
        """Find the longest path that generally follows PC1 direction."""
        # Sort all colors by PC1
        sorted_indices = np.argsort(pc1_values)
        sorted_colors = [colors[i] for i in sorted_indices]

        # Try to connect as many as possible in PC1 order
        # Use dynamic programming: for each color, find longest path ending there
        best_path = []

        for i, color in enumerate(sorted_colors):
            # Find the longest path that can reach this color
            best_prev_path = []

            for prev_color in neighbors[color]:
                if prev_color in sorted_colors[:i]:  # Must be earlier in PC1 order
                    # Check what path ends at prev_color
                    for path in [best_path]:  # simplified - just check best so far
                        if path and path[-1] == prev_color:
                            if len(path) > len(best_prev_path):
                                best_prev_path = path

            # Also check if we can extend any existing path
            extended = False
            if best_path and best_path[-1] in neighbors[color]:
                best_path = best_path + [color]
                extended = True

            if not extended:
                # Start fresh or continue
                if len(best_prev_path) > 0:
                    best_path = best_prev_path + [color]
                elif not best_path:
                    best_path = [color]

        return best_path

    # Simpler approach: greedy walk that allows small backwards steps
    def flexible_pc1_path(start, direction=1, max_backtrack=0.02):
        """Follow adjacency graph, allowing small backward steps to stay connected."""
        path = [start]
        current = start
        stuck_count = 0

        while stuck_count < 3:
            current_idx = color_to_idx[current]
            current_pc1 = embedding[current_idx, 0]

            # Find best neighbor - prefer forward progress, allow small backward
            candidates = []
            for neighbor in neighbors[current]:
                if neighbor in path:
                    continue
                neighbor_idx = color_to_idx[neighbor]
                neighbor_pc1 = embedding[neighbor_idx, 0]
                progress = (neighbor_pc1 - current_pc1) * direction
                candidates.append((neighbor, progress, neighbor_pc1))

            if not candidates:
                break

            # Sort by progress, take best that's not too far backward
            candidates.sort(key=lambda x: -x[1])
            best = None
            for cand, progress, _ in candidates:
                if progress >= -max_backtrack:
                    best = cand
                    break

            if best is None:
                stuck_count += 1
                # Take any neighbor to continue
                best = candidates[0][0]
            else:
                stuck_count = 0

            path.append(best)
            current = best

        return path

    # Try flexible greedy path for PC1
    greedy_path = flexible_pc1_path(start_color, direction=1)
    if len(greedy_path) > len(main_path or []):
        print(f"Flexible PC1 path: {len(greedy_path)} colors (vs BFS {len(main_path or [])})")
        chains = [greedy_path]
    else:
        print(f"BFS path kept: {len(main_path or [])} colors")
        if main_path:
            chains = [main_path]

    # Find gradients along other principal components too (chains CAN share colors)
    def find_gradient_along_pc(pc_idx, label):
        """Find gradient along a specific principal component."""
        pc_values = embedding[:, pc_idx]
        start_idx = np.argmin(pc_values)
        end_idx = np.argmax(pc_values)
        start_color = colors[start_idx]

        # Greedy path along this PC
        path = [start_color]
        current = start_color

        while True:
            current_idx = color_to_idx[current]
            current_pc = embedding[current_idx, pc_idx]

            best_neighbor = None
            best_progress = 0

            for neighbor in neighbors[current]:
                if neighbor in path:  # Avoid cycles within this chain only
                    continue
                neighbor_idx = color_to_idx[neighbor]
                neighbor_pc = embedding[neighbor_idx, pc_idx]
                progress = neighbor_pc - current_pc

                if progress > best_progress:
                    best_progress = progress
                    best_neighbor = neighbor

            if best_neighbor is None or best_progress < 0.001:
                break

            path.append(best_neighbor)
            current = best_neighbor

        if len(path) >= 5:
            print(f"{label} path: {len(path)} colors")
            return path
        return None

    # Try PC2 and PC3 for secondary gradients
    if embedding.shape[1] >= 2:
        pc2_path = find_gradient_along_pc(1, "PC2")
        if pc2_path:
            chains.append(pc2_path)

    if embedding.shape[1] >= 3:
        pc3_path = find_gradient_along_pc(2, "PC3")
        if pc3_path:
            chains.append(pc3_path)

    # Find colors with high coverage NOT in main chain
    main_chain_set = set(chains[0]) if chains else set()
    missing_high_cov = [
        (colors[i], coverage.get(colors[i], 0))
        for i in range(len(colors))
        if colors[i] not in main_chain_set
    ]
    missing_high_cov.sort(key=lambda x: -x[1])

    if missing_high_cov:
        print(f"Top colors not in main chain: {len(missing_high_cov)}")
        for color, cov in missing_high_cov[:5]:
            lab_val = bin_to_lab(color, scale)
            print(f"  L={lab_val[0]:.0f} a={lab_val[1]:.0f} b={lab_val[2]:.0f} cov={cov}")

    # Try building chains from high-coverage colors not in main chain
    for start_color, _ in missing_high_cov[:15]:
        # Build chain in BOTH directions from this color
        def extend_chain(start, pc_direction):
            path = [start]
            current = start

            while True:
                current_idx = color_to_idx[current]
                current_pc1 = embedding[current_idx, 0]

                best_neighbor = None
                best_progress = -999

                for neighbor in neighbors[current]:
                    if neighbor in path:
                        continue
                    neighbor_idx = color_to_idx[neighbor]
                    neighbor_pc1 = embedding[neighbor_idx, 0]
                    progress = (neighbor_pc1 - current_pc1) * pc_direction

                    if progress > best_progress:
                        best_progress = progress
                        best_neighbor = neighbor

                if best_neighbor is None:
                    break
                # Allow some backward movement to stay connected
                if best_progress < -0.05:
                    break

                path.append(best_neighbor)
                current = best_neighbor

            return path

        # Extend forward and backward
        forward = extend_chain(start_color, 1)
        backward = extend_chain(start_color, -1)

        # Combine (backward reversed + forward without start)
        full_path = backward[::-1] + forward[1:] if len(forward) > 1 else backward[::-1]

        if len(full_path) >= 5:
            # Check uniqueness
            is_unique = True
            for existing in chains:
                overlap = len(set(full_path) & set(existing))
                if overlap > len(full_path) * 0.5:  # >50% overlap
                    is_unique = False
                    break

            if is_unique:
                chains.append(full_path)

    # Sort chains by coverage (keep first chain in place if it's substantial)
    if len(chains) > 1:
        main_chain = chains[0]
        main_cov = sum(coverage.get(b, 0) for b in main_chain)
        other_chains = sorted(chains[1:],
                              key=lambda c: sum(coverage.get(b, 0) for b in c),
                              reverse=True)

        # Only keep main first if it has good coverage
        if main_cov >= sum(coverage.get(b, 0) for b in other_chains[0]) * 0.5:
            chains = [main_chain] + other_chains
        else:
            chains = sorted(chains, key=lambda c: sum(coverage.get(b, 0) for b in c), reverse=True)

    return chains


def extract_accent_colors(gradients: list[dict], color_metrics: dict,
                           min_multihop: float = 20.0, min_global: float = 0.8,
                           max_accents: int = 8) -> list[dict]:
    """
    Extract high-contrast colors that aren't captured by the gradients.

    These are "accent colors" - visually important due to contrast but too
    isolated or small to form gradient chains.

    Args:
        gradients: list of gradient dicts with 'chain' key
        color_metrics: dict from compute_color_metrics
        min_multihop: minimum multi-hop contrast to be considered accent
        min_global: minimum global contrast to be considered accent
        max_accents: maximum accent colors to return

    Returns:
        list of accent color dicts with bin, lab, and metrics
    """
    # Collect all colors already in gradients
    gradient_colors = set()
    for grad in gradients:
        gradient_colors.update(grad['chain'])

    # Find high-contrast colors not in gradients
    accents = []
    for bin_tuple, stats in color_metrics.items():
        if bin_tuple in gradient_colors:
            continue

        # Check if high contrast by either metric
        is_accent = (
            stats.get('multihop_contrast', 0) >= min_multihop or
            stats.get('global_contrast', 0) >= min_global
        )

        if is_accent:
            accents.append({
                'bin': bin_tuple,
                'lab': stats['lab'],
                'coverage': stats['coverage'],
                'multihop_contrast': stats.get('multihop_contrast', 0),
                'global_contrast': stats['global_contrast'],
                'chroma': stats['chroma']
            })

    # Sort by combined contrast (multi-hop weighted more since it captures local importance)
    accents.sort(key=lambda x: x['multihop_contrast'] * 2 + x['global_contrast'] * 20, reverse=True)

    return accents[:max_accents]


def visualize_color_metrics(color_metrics: dict, output_path: str, scale: float = 3.0,
                            top_n: int = 10):
    """
    Visualize colors as swatches, organized by metric.

    Creates a grid showing top colors by:
    - Coverage (dominant colors)
    - Multi-hop contrast (colors that contrast with nearby high-coverage colors)
    - Global contrast (rare/isolated colors)
    """
    from PIL import ImageDraw, ImageFont

    swatch_size = 40
    padding = 8
    label_width = 180
    row_height = swatch_size + padding

    # Get top colors by each metric
    by_coverage = sorted(color_metrics.items(), key=lambda x: x[1]['coverage'], reverse=True)[:top_n]
    by_multihop = sorted(color_metrics.items(), key=lambda x: x[1]['multihop_contrast'], reverse=True)[:top_n]
    by_global = sorted(color_metrics.items(), key=lambda x: x[1]['global_contrast'], reverse=True)[:top_n]

    sections = [
        ("Top by Coverage", by_coverage),
        ("Top by Multi-Hop Contrast", by_multihop),
        ("Top by Global Contrast (Rare)", by_global),
    ]

    # Calculate image size
    img_width = label_width + top_n * swatch_size + padding * 2
    img_height = len(sections) * (row_height + 25) + padding * 2 + 20

    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding

    for section_name, colors in sections:
        # Section header
        draw.text((padding, y), section_name, fill=(0, 0, 0))
        y += 20

        # Draw swatches
        for i, (bin_tuple, stats) in enumerate(colors):
            x = label_width + i * swatch_size

            # Get RGB color
            lab = bin_to_lab(bin_tuple, scale)
            rgb = tuple(lab_to_rgb(lab.reshape(1, -1))[0])

            # Draw swatch
            draw.rectangle([x, y, x + swatch_size - 2, y + swatch_size - 2], fill=rgb)

            # Draw border for visibility
            draw.rectangle([x, y, x + swatch_size - 2, y + swatch_size - 2], outline=(128, 128, 128))

        # Add stats for first few
        stats_text = f"cov: {colors[0][1]['coverage']:.1%} → {colors[-1][1]['coverage']:.1%}"
        draw.text((padding, y + swatch_size // 2 - 5), stats_text, fill=(100, 100, 100))

        y += row_height + 5

    img.save(output_path)
    print(f"Saved color metrics swatches to {output_path}")


def visualize_gradients_and_accents(gradients: list[dict], accents: list[dict],
                                     coverage: dict, total_pixels: int,
                                     output_path: str, scale: float = 3.0,
                                     max_gradients: int = 5):
    """
    Visualize gradients and accent colors together as a complete palette.
    """
    from PIL import ImageDraw

    swatch_size = 30
    padding = 10
    text_width = 140

    # Limit gradients shown
    gradients = gradients[:max_gradients]

    # Calculate dimensions
    max_grad_len = max((len(g['chain']) for g in gradients), default=1)
    grad_width = text_width + max_grad_len * swatch_size + padding

    accent_width = text_width + len(accents) * swatch_size + padding if accents else 0

    img_width = max(grad_width, accent_width) + padding * 2
    img_height = (len(gradients) + 2) * (swatch_size + padding) + padding * 2

    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding

    # Title
    draw.text((padding, y), "Gradients:", fill=(0, 0, 0))
    y += 20

    # Draw gradients
    for grad in gradients:
        chain = grad['chain']
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100

        # Label
        label = f"{grad['direction']} {chain_cov:.1f}%"
        draw.text((padding, y + swatch_size // 2 - 5), label, fill=(80, 80, 80))

        # Swatches
        for i, bin_tuple in enumerate(chain):
            x = text_width + i * swatch_size
            lab = bin_to_lab(bin_tuple, scale)
            rgb = tuple(lab_to_rgb(lab.reshape(1, -1))[0])
            draw.rectangle([x, y, x + swatch_size - 2, y + swatch_size - 2], fill=rgb)

        y += swatch_size + padding // 2

    # Accent colors section
    if accents:
        y += padding
        draw.text((padding, y), "Accents:", fill=(0, 0, 0))
        y += 20

        for i, acc in enumerate(accents):
            x = text_width + i * swatch_size
            rgb = tuple(lab_to_rgb(acc['lab'].reshape(1, -1))[0])
            draw.rectangle([x, y, x + swatch_size - 2, y + swatch_size - 2], fill=rgb)
            draw.rectangle([x, y, x + swatch_size - 2, y + swatch_size - 2], outline=(100, 100, 100))

    img.save(output_path)
    print(f"Saved palette (gradients + accents) to {output_path}")


def visualize_chains(chains: list[list[tuple]], coverage: dict, total_pixels: int,
                     output_path: str, scale: float = 3.0, max_chains: int = 10,
                     gradients: list[dict] = None):
    """
    Visualize gradient chains as color strips.

    Args:
        chains: list of chain lists (for backward compatibility)
        coverage: dict of bin -> pixel count
        total_pixels: total image pixels
        output_path: where to save
        scale: JND scale
        max_chains: max to show
        gradients: optional list of gradient dicts (if provided, uses metadata like direction)
    """
    from PIL import ImageDraw

    swatch_size = 25
    padding = 10
    text_width = 120
    row_height = swatch_size + padding

    chains = chains[:max_chains]
    max_len = max(len(c) for c in chains) if chains else 1

    img_width = text_width + max_len * swatch_size + padding * 2
    img_height = len(chains) * row_height + padding * 2

    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    for row, chain in enumerate(chains):
        y = padding + row * row_height

        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100

        # Get direction from gradient metadata if available
        direction = ""
        if gradients and row < len(gradients):
            direction = gradients[row].get('direction', '')

        label = f"{direction} {chain_cov:.1f}% ({len(chain)})"
        draw.text((padding, y + swatch_size // 2 - 5), label, fill=(0, 0, 0))

        for i, bin_tuple in enumerate(chain):
            x = text_width + i * swatch_size
            lab = bin_to_lab(bin_tuple, scale)
            rgb = lab_to_rgb(lab.reshape(1, -1))[0]
            draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(rgb))

    img.save(output_path)
    print(f"Saved {len(chains)} chains to {output_path}")


if __name__ == '__main__':
    source_dir = Path('source_images')
    test_image = source_dir / 'soft_gradients.jpeg'

    print(f"Analyzing: {test_image.name}")
    print("=" * 60)

    # Load and quantize
    lab, (h, w) = load_image(str(test_image))
    binned = quantize(lab, scale=3.0)
    total_pixels = h * w

    print(f"Image: {w}x{h}")

    # Build adjacency graph
    adjacency, coverage = build_adjacency_graph(binned)
    print(f"Unique colors: {len(coverage)}")
    print(f"Adjacency pairs: {len(adjacency)}")

    # Build adjacency matrix
    matrix, colors = build_adjacency_matrix(adjacency, coverage, min_coverage=0.0005,
                                            scale=3.0, lab_weight_sigma=15.0)
    print(f"Colors after filtering: {len(colors)}")

    # Embed in lower dimensions
    embedding = embed_adjacency_space(matrix, method='pca', n_components=3)

    # Visualize embedding
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    visualize_embedding(embedding, colors, coverage,
                        str(output_dir / f"{test_image.stem}_adjacency_space.png"))

    # Find gradient paths using original PC-following method
    print("\n--- PC-Following Method ---")
    chains = find_gradient_paths(embedding, colors, adjacency, coverage)
    print(f"Found {len(chains)} gradient chains")

    for i, chain in enumerate(chains[:5]):
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        print(f"  Chain {i+1}: {len(chain)} colors, {chain_cov:.1f}%")

    visualize_chains(chains, coverage, total_pixels,
                     str(output_dir / f"{test_image.stem}_adjacency_chains.png"))

    # Find gradients using graph-based method (traces actual adjacency paths)
    print("\n--- Graph-Based Method (LAB monotonic) ---")
    graph_chains = find_graph_gradients(
        colors, adjacency, coverage,
        scale=3.0, min_chain_length=5, max_lab_step=25.0
    )
    print(f"Found {len(graph_chains)} graph gradients")

    for i, chain in enumerate(graph_chains[:8]):
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        print(f"  Gradient {i+1}: {len(chain)} colors, {chain_cov:.1f}%")

    visualize_chains(graph_chains, coverage, total_pixels,
                     str(output_dir / f"{test_image.stem}_graph_gradients.png"),
                     max_chains=10)

    # Build directional adjacency
    print("\n--- Building Directional Adjacency ---")
    directional = build_directional_adjacency(binned)
    print(f"Directional pairs: {len(directional)}")

    # Compute color metrics FIRST (needed for flow gradient seeding)
    print("\n--- Computing Color Metrics ---")
    color_metrics = compute_color_metrics(colors, adjacency, coverage, scale=3.0)

    # Compute multi-hop adjacency contrast (key for accent color detection)
    print("Computing multi-hop adjacency contrast...")
    multihop = compute_multihop_contrast(colors, adjacency, coverage, scale=3.0, max_hops=3)

    # Add multi-hop to color_metrics
    for b in colors:
        color_metrics[b]['multihop_contrast'] = multihop.get(b, 0)

    # Compute spatial coherence BEFORE gradient detection
    # (coherence is used to weight flow edges)
    print("Computing spatial coherence...")
    coherence = compute_spatial_coherence(binned, colors, coverage)

    # Add coherence to color_metrics
    for b in colors:
        coh = coherence.get(b, {'coherence': 0, 'blob_count': 0, 'largest_blob': 0})
        color_metrics[b]['coherence'] = coh['coherence']
        color_metrics[b]['blob_count'] = coh['blob_count']
        color_metrics[b]['largest_blob'] = coh['largest_blob']

    # Identify noise colors based on relative coherence
    noise_colors, median_coherence = identify_noise_colors(coherence, coverage, threshold=0.3)
    print(f"Median coherence: {median_coherence:.2f}, noise threshold: {0.3 * median_coherence:.2f}")
    print(f"Identified {len(noise_colors)} noise colors ({len(noise_colors)/len(colors)*100:.1f}%)")

    # Mark noise in color_metrics
    for b in colors:
        color_metrics[b]['is_noise'] = b in noise_colors

    # Classify color scheme
    print("\n--- Color Scheme Classification ---")
    scheme = classify_color_scheme(color_metrics, min_coverage=0.001)
    print(f"Scheme type: {scheme['scheme_type']}")
    print(f"Chroma profile: {scheme['chroma_profile']['type']} "
          f"(median={scheme['chroma_profile']['median']:.1f}, "
          f"max={scheme['chroma_profile']['max']:.1f})")
    print(f"Chromatic colors: {len(scheme['chromatic_colors'])}")
    print(f"Neutral colors: {len(scheme['neutral_colors'])}")
    print(f"Lightness range: L={scheme['lightness_range'][0]:.0f}-{scheme['lightness_range'][1]:.0f}")
    if scheme['hue_clusters']:
        print(f"Hue clusters: {len(scheme['hue_clusters'])}")
        for i, cluster in enumerate(scheme['hue_clusters']):
            print(f"  Cluster {i+1}: hue={cluster['center']:.0f}°, "
                  f"spread={cluster['spread']:.0f}°, "
                  f"coverage={cluster['coverage']*100:.1f}%")
    if 'dominant_hue' in scheme:
        print(f"Dominant hue: {scheme['dominant_hue']:.0f}°")
    if 'accent_hue' in scheme:
        print(f"Accent hue: {scheme['accent_hue']:.0f}°")

    # Find flow-based gradients with unified scoring
    print("\n--- Directional Flow Method (unified scoring) ---")
    flow_gradients = find_flow_gradients(
        colors, directional, coverage,
        scale=3.0, min_chain_length=3, min_asymmetry=0.25,
        color_metrics=color_metrics
    )
    print(f"Found {len(flow_gradients)} flow gradients")

    for i, grad in enumerate(flow_gradients[:12]):
        chain = grad['chain']
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        start_score = grad.get('starting_score', 0)
        print(f"  {grad['direction']:>5}: {len(chain):2d} colors, {chain_cov:5.1f}%, "
              f"L={grad['l_range']:.0f} a={grad['a_range']:.0f} b={grad['b_range']:.0f} "
              f"(seed={start_score:.4f})")

    # Visualize flow gradients
    flow_chains = [g['chain'] for g in flow_gradients]
    visualize_chains(flow_chains, coverage, total_pixels,
                     str(output_dir / f"{test_image.stem}_flow_gradients.png"),
                     max_chains=12, gradients=flow_gradients)

    # Compute pixel-level local contrast (for analysis)
    print("\n--- Metrics Analysis ---")
    print("Computing pixel-level local contrast...")
    pixel_contrast = compute_pixel_local_contrast(lab, binned, colors, radius=5)

    # Add pixel contrast to color_metrics
    for b in colors:
        color_metrics[b]['pixel_contrast'] = pixel_contrast.get(b, 0)

    # Show distribution of metrics
    coverages = [s['coverage'] for s in color_metrics.values()]
    local_contrasts = [s['local_contrast'] for s in color_metrics.values()]
    pixel_contrasts = [s['pixel_contrast'] for s in color_metrics.values()]
    multihop_contrasts = [s['multihop_contrast'] for s in color_metrics.values()]
    global_contrasts = [s['global_contrast'] for s in color_metrics.values()]
    chromas = [s['chroma'] for s in color_metrics.values()]
    coherences = [s['coherence'] for s in color_metrics.values()]
    blob_counts = [s['blob_count'] for s in color_metrics.values()]

    print(f"Coverage:        min={min(coverages):.4f}, max={max(coverages):.4f}, mean={np.mean(coverages):.4f}")
    print(f"Local contrast (1-hop adj):  min={min(local_contrasts):.1f}, max={max(local_contrasts):.1f}, mean={np.mean(local_contrasts):.1f}")
    print(f"Local contrast (pixel):      min={min(pixel_contrasts):.1f}, max={max(pixel_contrasts):.1f}, mean={np.mean(pixel_contrasts):.1f}")
    print(f"Multi-hop contrast (3-hop):  min={min(multihop_contrasts):.1f}, max={max(multihop_contrasts):.1f}, mean={np.mean(multihop_contrasts):.1f}")
    print(f"Global contrast: min={min(global_contrasts):.2f}, max={max(global_contrasts):.2f}, mean={np.mean(global_contrasts):.2f}")
    print(f"Chroma:          min={min(chromas):.1f}, max={max(chromas):.1f}, mean={np.mean(chromas):.1f}")
    print(f"Coherence:       min={min(coherences):.2f}, max={max(coherences):.2f}, mean={np.mean(coherences):.2f}")
    print(f"Blob count:      min={min(blob_counts)}, max={max(blob_counts)}, mean={np.mean(blob_counts):.1f}")

    # Find colors by different metrics
    print("\n--- Top Colors by Coverage ---")
    by_coverage = sorted(color_metrics.items(), key=lambda x: x[1]['coverage'], reverse=True)[:5]
    for b, s in by_coverage:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.3f} local={s['local_contrast']:.1f} global={s['global_contrast']:.2f}")

    print("\n--- Top Colors by Local Contrast ---")
    by_local = sorted(color_metrics.items(), key=lambda x: x[1]['local_contrast'], reverse=True)[:5]
    for b, s in by_local:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.3f} local={s['local_contrast']:.1f} global={s['global_contrast']:.2f}")

    print("\n--- Top Colors by Global Contrast (Rare in Palette) ---")
    by_global = sorted(color_metrics.items(), key=lambda x: x[1]['global_contrast'], reverse=True)[:5]
    for b, s in by_global:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.3f} local={s['local_contrast']:.1f} global={s['global_contrast']:.2f}")

    print("\n--- High Local Contrast but Low Coverage (edge/boundary colors) ---")
    # Normalize for comparison
    max_local = max(local_contrasts)
    max_cov = max(coverages)
    edge_colors = [
        (b, s) for b, s in color_metrics.items()
        if s['local_contrast'] / max_local > 0.5 and s['coverage'] / max_cov < 0.1
    ]
    edge_colors.sort(key=lambda x: x[1]['local_contrast'], reverse=True)
    for b, s in edge_colors[:5]:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.4f} local={s['local_contrast']:.1f} global={s['global_contrast']:.2f}")

    print("\n--- Darkest Colors (L < 25) ---")
    dark_colors = [(b, s) for b, s in color_metrics.items() if s['lab'][0] < 25]
    dark_colors.sort(key=lambda x: x[1]['lab'][0])
    if dark_colors:
        for b, s in dark_colors[:8]:
            print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
                  f"cov={s['coverage']:.4f} 1hop={s['local_contrast']:.1f} 3hop={s['multihop_contrast']:.1f} global={s['global_contrast']:.2f}")
    else:
        print("  (none found - may be filtered by min_coverage)")

    print("\n--- Top Colors by Multi-Hop Contrast ---")
    by_multihop = sorted(color_metrics.items(), key=lambda x: x[1]['multihop_contrast'], reverse=True)[:8]
    for b, s in by_multihop:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.4f} 1hop={s['local_contrast']:.1f} 3hop={s['multihop_contrast']:.1f} global={s['global_contrast']:.2f}")

    print("\n--- Top Colors by Coherence (most spatially coherent) ---")
    by_coherence = sorted(color_metrics.items(), key=lambda x: x[1]['coherence'], reverse=True)[:8]
    for b, s in by_coherence:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.4f} coherence={s['coherence']:.2f} blobs={s['blob_count']}")

    print("\n--- Noise Colors (relative coherence filter) ---")
    noise_list = [(b, s) for b, s in color_metrics.items() if s['is_noise']]
    noise_list.sort(key=lambda x: x[1]['coherence'])
    for b, s in noise_list[:8]:
        print(f"  L={s['lab'][0]:5.1f} a={s['lab'][1]:5.1f} b={s['lab'][2]:5.1f}  "
              f"cov={s['coverage']:.4f} coherence={s['coherence']:.2f} blobs={s['blob_count']}")
    if not noise_list:
        print("  (none identified as noise)")
    elif len(noise_list) > 8:
        print(f"  ... and {len(noise_list) - 8} more")

    # Visualize color metrics
    visualize_color_metrics(color_metrics, str(output_dir / f"{test_image.stem}_metrics.png"),
                            scale=3.0, top_n=12)

    # Show gradients seeded from dark/contrasting colors (L < 30)
    print("\n--- Gradients with Dark Starting Points (L < 30) ---")
    dark_seeded = []
    for grad in flow_gradients:
        first_lab = bin_to_lab(grad['chain'][0], 3.0)
        if first_lab[0] < 30:
            dark_seeded.append((grad, first_lab))

    if dark_seeded:
        for grad, first_lab in dark_seeded:
            chain = grad['chain']
            chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
            print(f"  {grad['direction']:>5}: {len(chain)} colors, {chain_cov:.2f}%, "
                  f"starts at L={first_lab[0]:.1f} a={first_lab[1]:.1f} b={first_lab[2]:.1f} "
                  f"(seed={grad.get('starting_score', 0):.4f})")
    else:
        print("  (none found - dark colors may not have strong directional flow)")

    # Check raw coverage for very dark bins
    print("\n--- Dark Bins in Raw Data (before filtering) ---")
    dark_bins = [(b, c) for b, c in coverage.items() if bin_to_lab(b, 3.0)[0] < 25]
    dark_bins.sort(key=lambda x: bin_to_lab(x[0], 3.0)[0])
    for b, c in dark_bins[:8]:
        lab = bin_to_lab(b, 3.0)
        print(f"  L={lab[0]:5.1f} a={lab[1]:5.1f} b={lab[2]:5.1f}  pixels={c:6d} ({c/total_pixels*100:.3f}%)")
