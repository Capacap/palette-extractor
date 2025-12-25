#!/usr/bin/env python3
"""
Unified color analysis pipeline.

Extracts color schemes from images and produces prose reports for LLM consumption.
Four stages: Data Preparation → Feature Extraction → Synthesis → Render
"""

import numpy as np
from PIL import Image
from scipy.ndimage import label
from dataclasses import dataclass, field
from typing import Optional
import math


# =============================================================================
# Constants
# =============================================================================

JND = 2.3  # Just Noticeable Difference in LAB units
FINE_SCALE = 1.0  # Fine bins: 1 JND = 2.3 LAB units
COARSE_SCALE = 5.0  # Coarse bins: ~12 LAB units

# Image size limits (security: prevent decompression bombs)
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels
MAX_IMAGE_DIMENSION = 10_000  # 10k pixels per side

# Algorithm parameters
HUE_CLUSTER_RANGE = 30  # Degrees within which hues are considered similar
MAX_GRADIENT_SEARCH_SEEDS = 50  # Limit gradient detection search
MAX_GRADIENT_CHAIN_LENGTH = 20  # Maximum colors in a gradient chain
MIN_SIGNIFICANCE_RATIO = 0.05  # Minimum significance relative to top color
MAX_NOTABLE_COLORS = 10  # Maximum colors to include in output


# =============================================================================
# Color Conversion
# =============================================================================

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (0-255) to LAB color space."""
    rgb_norm = rgb.astype(np.float64) / 255.0

    # Apply gamma correction
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # RGB to XYZ matrix
    r, g, b = rgb_linear[:, 0], rgb_linear[:, 1], rgb_linear[:, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to LAB (D65 reference white)
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn

    epsilon = 0.008856
    kappa = 903.3
    fx = np.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
    fy = np.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
    fz = np.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)

    return np.column_stack([L, a, b_val])


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB array to RGB (0-255)."""
    if lab.ndim == 1:
        lab = lab.reshape(1, -1)

    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    x = np.where(fx**3 > epsilon, fx**3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(fz**3 > epsilon, fz**3, (116 * fz - 16) / kappa)

    x *= 0.95047
    z *= 1.08883

    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b_out = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    rgb_linear = np.column_stack([r, g, b_out])
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.clip(rgb_linear, 0, None), 1/2.4) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def lab_to_hex(lab: np.ndarray) -> str:
    """Convert LAB to hex string."""
    rgb = lab_to_rgb(lab)
    if rgb.ndim > 1:
        rgb = rgb[0]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def lab_to_rgb_tuple(lab: np.ndarray) -> tuple:
    """Convert LAB to RGB tuple."""
    rgb = lab_to_rgb(lab)
    if rgb.ndim > 1:
        rgb = rgb[0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


# =============================================================================
# Color Utilities
# =============================================================================

def compute_chroma(lab: np.ndarray) -> float:
    """Compute chroma (saturation) from LAB coordinates."""
    return math.sqrt(lab[1]**2 + lab[2]**2)


def compute_hue(lab: np.ndarray) -> float:
    """Compute hue angle (0-360 degrees) from LAB coordinates."""
    return math.degrees(math.atan2(lab[2], lab[1])) % 360


def circular_hue_distance(hue1: float, hue2: float) -> float:
    """Compute minimum angular distance between two hues (0-180)."""
    diff = abs(hue1 - hue2)
    return min(diff, 360 - diff)


# =============================================================================
# Stage 1: Data Preparation
# =============================================================================

@dataclass
class BinData:
    """Data for a single color bin."""
    lab: np.ndarray  # Representative LAB value
    count: int  # Pixel count
    positions: list = field(default_factory=list)  # (y, x) positions


@dataclass
class PreparedData:
    """Output of Stage 1: Data Preparation."""
    fine_bins: dict  # bin_tuple -> BinData
    coarse_bins: dict  # bin_tuple -> BinData
    fine_to_coarse: dict  # fine_bin -> coarse_bin
    coarse_to_fine: dict  # coarse_bin -> [fine_bins]
    total_pixels: int
    image_shape: tuple  # (height, width)


def prepare_data(image_path: str) -> PreparedData:
    """
    Stage 1: Load image and quantize at two scales.

    Fine scale (JND): preserves gradient steps
    Coarse scale (~5x JND): captures major color blocks

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image or exceeds size limits
    """
    # Load image with validation
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")

    # Validate image dimensions (security: prevent decompression bombs)
    width, height = img.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions {width}x{height} exceed maximum "
            f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image has {width * height:,} pixels, exceeding maximum {MAX_IMAGE_PIXELS:,}"
        )

    img = img.convert('RGB')
    pixels = np.array(img)
    h, w = pixels.shape[:2]

    # Convert to LAB
    lab_flat = rgb_to_lab(pixels.reshape(-1, 3))
    lab_image = lab_flat.reshape(h, w, 3)

    # Quantize at both scales
    fine_size = FINE_SCALE * JND
    coarse_size = COARSE_SCALE * JND

    fine_binned = np.round(lab_image / fine_size).astype(np.int32)
    coarse_binned = np.round(lab_image / coarse_size).astype(np.int32)

    # Build bin data structures
    fine_bins = {}
    coarse_bins = {}
    fine_to_coarse = {}
    coarse_to_fine = {}

    for y in range(h):
        for x in range(w):
            fine_bin = tuple(fine_binned[y, x])
            coarse_bin = tuple(coarse_binned[y, x])

            # Fine bin
            if fine_bin not in fine_bins:
                fine_lab = np.array(fine_bin) * fine_size
                fine_bins[fine_bin] = BinData(lab=fine_lab, count=0, positions=[])
            fine_bins[fine_bin].count += 1
            fine_bins[fine_bin].positions.append((y, x))

            # Coarse bin
            if coarse_bin not in coarse_bins:
                coarse_lab = np.array(coarse_bin) * coarse_size
                coarse_bins[coarse_bin] = BinData(lab=coarse_lab, count=0, positions=[])
            coarse_bins[coarse_bin].count += 1
            coarse_bins[coarse_bin].positions.append((y, x))

            # Scale mapping
            fine_to_coarse[fine_bin] = coarse_bin
            if coarse_bin not in coarse_to_fine:
                coarse_to_fine[coarse_bin] = set()
            coarse_to_fine[coarse_bin].add(fine_bin)

    # Convert sets to lists
    coarse_to_fine = {k: list(v) for k, v in coarse_to_fine.items()}

    return PreparedData(
        fine_bins=fine_bins,
        coarse_bins=coarse_bins,
        fine_to_coarse=fine_to_coarse,
        coarse_to_fine=coarse_to_fine,
        total_pixels=h * w,
        image_shape=(h, w)
    )


# =============================================================================
# Stage 2: Feature Extraction
# =============================================================================

@dataclass
class StabilityInfo:
    """Scale stability classification for a coarse bin."""
    stability_type: str  # 'anchor', 'gradient', 'texture'
    fine_count: int  # Number of fine children
    fine_variance: float  # LAB variance of fine children
    dominant_axis: Optional[str] = None  # 'L', 'a', 'b' if gradient


@dataclass
class ColorFamily:
    """A group of colors with similar hue."""
    hue_center: float  # 0-360
    lightness_range: tuple  # (min_L, max_L)
    chroma_range: tuple  # (min, max)
    total_coverage: float  # 0-1
    members: list  # coarse bin tuples
    is_neutral: bool = False


@dataclass
class GradientChain:
    """A detected gradient in the image."""
    stops: list  # List of coarse bins (anchor colors only)
    fine_members: list  # All fine bins in the chain
    direction: str  # 'horizontal', 'vertical', 'diagonal', 'mixed'
    coverage: float  # Total pixel coverage
    lab_range: dict  # {'L': (min, max), 'a': (min, max), 'b': (min, max)}
    family_span: list  # Which families this gradient spans


@dataclass
class ColorMetrics:
    """Per-color metrics."""
    coverage: float
    chroma: float
    hue: float
    local_contrast: float
    coherence: float
    isolation: float
    lab: np.ndarray


@dataclass
class FeatureData:
    """Output of Stage 2: Feature Extraction."""
    stability: dict  # coarse_bin -> StabilityInfo
    families: list  # List of ColorFamily
    gradients: list  # List of GradientChain
    metrics: dict  # coarse_bin -> ColorMetrics
    adjacency: dict  # (bin1, bin2) -> count (fine scale)


def analyze_scale_stability(data: PreparedData) -> dict:
    """
    Stage 2a: Classify each coarse bin by its fine-scale structure.

    ANCHOR: 1-2 fine children, color is stable
    GRADIENT: 3+ fine children with monotonic LAB progression
    TEXTURE: 3+ fine children scattered in LAB space
    """
    stability = {}
    fine_size = FINE_SCALE * JND

    for coarse_bin, fine_list in data.coarse_to_fine.items():
        fine_count = len(fine_list)

        if fine_count <= 2:
            stability[coarse_bin] = StabilityInfo(
                stability_type='anchor',
                fine_count=fine_count,
                fine_variance=0.0
            )
            continue

        # Get LAB values of fine children
        fine_labs = np.array([np.array(fb) * fine_size for fb in fine_list])

        # Compute variance in each dimension
        var_L = np.var(fine_labs[:, 0])
        var_a = np.var(fine_labs[:, 1])
        var_b = np.var(fine_labs[:, 2])
        total_var = var_L + var_a + var_b

        # Check for monotonic progression (gradient vs texture)
        # A gradient has most variance in one axis with monotonic progression
        max_var_axis = np.argmax([var_L, var_a, var_b])
        axis_names = ['L', 'a', 'b']

        # Sort fine bins by the dominant axis
        sorted_labs = fine_labs[fine_labs[:, max_var_axis].argsort()]

        # Check monotonicity: differences should be mostly same sign
        diffs = np.diff(sorted_labs[:, max_var_axis])
        if len(diffs) > 0:
            pos_ratio = np.sum(diffs >= 0) / len(diffs)
            is_monotonic = pos_ratio > 0.7 or pos_ratio < 0.3
        else:
            is_monotonic = False

        # Gradient: one dominant axis with monotonic progression
        dominant_var_ratio = max(var_L, var_a, var_b) / (total_var + 1e-10)

        if is_monotonic and dominant_var_ratio > 0.5:
            stability[coarse_bin] = StabilityInfo(
                stability_type='gradient',
                fine_count=fine_count,
                fine_variance=total_var,
                dominant_axis=axis_names[max_var_axis]
            )
        else:
            stability[coarse_bin] = StabilityInfo(
                stability_type='texture',
                fine_count=fine_count,
                fine_variance=total_var
            )

    return stability


def detect_color_families(data: PreparedData, stability: dict) -> list:
    """
    Stage 2b: Group coarse bins by hue similarity.
    """
    coarse_size = COARSE_SCALE * JND
    families = []
    assigned = set()

    # Compute hue and chroma for each coarse bin
    bin_info = {}
    for coarse_bin, bin_data in data.coarse_bins.items():
        lab = bin_data.lab
        chroma = compute_chroma(lab)
        hue = compute_hue(lab)
        coverage = bin_data.count / data.total_pixels
        bin_info[coarse_bin] = {
            'lab': lab,
            'chroma': chroma,
            'hue': hue,
            'coverage': coverage,
            'L': lab[0]
        }

    # Adaptive chroma threshold based on distribution
    chromas = [info['chroma'] for info in bin_info.values()]
    median_chroma = np.median(chromas) if chromas else 0
    chroma_threshold = max(10, median_chroma * 0.5)

    # Separate chromatic and neutral
    chromatic_bins = {b: info for b, info in bin_info.items() if info['chroma'] >= chroma_threshold}
    neutral_bins = {b: info for b, info in bin_info.items() if info['chroma'] < chroma_threshold}

    # Sort by coverage to seed clusters with most significant colors
    sorted_chromatic = sorted(chromatic_bins.items(), key=lambda x: -x[1]['coverage'])

    for seed_bin, seed_info in sorted_chromatic:
        if seed_bin in assigned:
            continue

        # Find all bins within hue range
        cluster_members = [seed_bin]
        assigned.add(seed_bin)

        for other_bin, other_info in chromatic_bins.items():
            if other_bin in assigned:
                continue

            hue_diff = circular_hue_distance(seed_info['hue'], other_info['hue'])

            if hue_diff <= HUE_CLUSTER_RANGE:
                cluster_members.append(other_bin)
                assigned.add(other_bin)

        # Compute family properties
        member_infos = [bin_info[b] for b in cluster_members]
        hues = [info['hue'] for info in member_infos]
        lightnesses = [info['L'] for info in member_infos]
        chromas_cluster = [info['chroma'] for info in member_infos]
        coverages = [info['coverage'] for info in member_infos]

        # Circular mean for hue
        sin_sum = sum(math.sin(math.radians(h)) for h in hues)
        cos_sum = sum(math.cos(math.radians(h)) for h in hues)
        hue_center = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

        families.append(ColorFamily(
            hue_center=hue_center,
            lightness_range=(min(lightnesses), max(lightnesses)),
            chroma_range=(min(chromas_cluster), max(chromas_cluster)),
            total_coverage=sum(coverages),
            members=cluster_members,
            is_neutral=False
        ))

    # Add neutral family if significant
    if neutral_bins:
        neutral_members = list(neutral_bins.keys())
        neutral_infos = [bin_info[b] for b in neutral_members]
        lightnesses = [info['L'] for info in neutral_infos]
        coverages = [info['coverage'] for info in neutral_infos]
        chromas_neutral = [info['chroma'] for info in neutral_infos]

        families.append(ColorFamily(
            hue_center=0,  # N/A for neutral
            lightness_range=(min(lightnesses), max(lightnesses)),
            chroma_range=(min(chromas_neutral), max(chromas_neutral)),
            total_coverage=sum(coverages),
            members=neutral_members,
            is_neutral=True
        ))

    return families


def build_adjacency_graph(data: PreparedData) -> dict:
    """Build adjacency graph at fine scale with 8-connectivity."""
    h, w = data.image_shape
    adjacency = {}

    # Rebuild fine binned image for adjacency
    fine_size = FINE_SCALE * JND
    fine_binned = {}
    for fine_bin, bin_data in data.fine_bins.items():
        for pos in bin_data.positions:
            fine_binned[pos] = fine_bin

    def add_edge(b1, b2):
        if b1 == b2:
            return
        key = (b1, b2) if b1 < b2 else (b2, b1)
        adjacency[key] = adjacency.get(key, 0) + 1

    # Check all 8 neighbors for each pixel
    for (y, x), b1 in fine_binned.items():
        for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            ny, nx = y + dy, x + dx
            if (ny, nx) in fine_binned:
                add_edge(b1, fine_binned[(ny, nx)])

    return adjacency


def compute_color_metrics(data: PreparedData, adjacency: dict, stability: dict) -> dict:
    """
    Stage 2c: Compute per-color metrics for coarse bins.
    """
    coarse_size = COARSE_SCALE * JND
    fine_size = FINE_SCALE * JND
    metrics = {}

    # Build neighbor lookup for coarse bins (via fine adjacency)
    coarse_neighbors = {b: set() for b in data.coarse_bins}
    coarse_adjacency_count = {b: 0 for b in data.coarse_bins}

    for (f1, f2), count in adjacency.items():
        c1 = data.fine_to_coarse.get(f1)
        c2 = data.fine_to_coarse.get(f2)
        if c1 and c2 and c1 != c2:
            coarse_neighbors[c1].add(c2)
            coarse_neighbors[c2].add(c1)
            coarse_adjacency_count[c1] += count
            coarse_adjacency_count[c2] += count

    # Compute local contrast (average LAB distance to neighbors)
    local_contrast = {}
    for coarse_bin, neighbors in coarse_neighbors.items():
        if not neighbors:
            local_contrast[coarse_bin] = 0.0
            continue

        lab = data.coarse_bins[coarse_bin].lab
        distances = []
        for neighbor in neighbors:
            neighbor_lab = data.coarse_bins[neighbor].lab
            distances.append(np.linalg.norm(lab - neighbor_lab))
        local_contrast[coarse_bin] = np.mean(distances) if distances else 0.0

    # Compute coherence (largest blob / total pixels)
    coherence = compute_coherence(data)

    # Compute isolation (inverse of neighbor count, normalized)
    max_neighbors = max(len(n) for n in coarse_neighbors.values()) if coarse_neighbors else 1

    for coarse_bin, bin_data in data.coarse_bins.items():
        lab = bin_data.lab
        chroma = compute_chroma(lab)
        hue = compute_hue(lab)
        coverage = bin_data.count / data.total_pixels

        neighbor_count = len(coarse_neighbors[coarse_bin])
        isolation = 1.0 - (neighbor_count / (max_neighbors + 1))

        metrics[coarse_bin] = ColorMetrics(
            coverage=coverage,
            chroma=chroma,
            hue=hue,
            local_contrast=local_contrast.get(coarse_bin, 0.0),
            coherence=coherence.get(coarse_bin, 0.0),
            isolation=isolation,
            lab=lab
        )

    return metrics


def compute_coherence(data: PreparedData) -> dict:
    """Compute spatial coherence for each coarse bin."""
    h, w = data.image_shape
    coarse_size = COARSE_SCALE * JND

    # Rebuild coarse binned image
    coarse_binned = np.zeros((h, w, 3), dtype=np.int32)
    for coarse_bin, bin_data in data.coarse_bins.items():
        for y, x in bin_data.positions:
            coarse_binned[y, x] = coarse_bin

    coherence = {}
    structure = np.ones((3, 3), dtype=int)

    for coarse_bin, bin_data in data.coarse_bins.items():
        if bin_data.count == 0:
            coherence[coarse_bin] = 0.0
            continue

        # Create mask
        mask = np.all(coarse_binned == coarse_bin, axis=2)

        # Find connected components
        labeled, num_blobs = label(mask, structure=structure)

        if num_blobs == 0:
            coherence[coarse_bin] = 0.0
            continue

        blob_sizes = np.bincount(labeled.ravel())[1:]
        largest_blob = int(np.max(blob_sizes))
        coherence[coarse_bin] = largest_blob / bin_data.count

    return coherence


def detect_gradients(data: PreparedData, stability: dict, adjacency: dict, families: list) -> list:
    """
    Stage 2c: Detect gradient chains using fine-scale adjacency.

    Gradients are sequences of colors with:
    - Strong adjacency connections
    - Monotonic progression in LAB
    - Stops are ANCHOR colors only
    """
    fine_size = FINE_SCALE * JND
    gradients = []

    # Build fine neighbor graph
    fine_neighbors = {b: set() for b in data.fine_bins}
    for (f1, f2), count in adjacency.items():
        if count >= 10:  # Minimum adjacency threshold
            fine_neighbors[f1].add(f2)
            fine_neighbors[f2].add(f1)

    # Find gradient chains by tracing paths through color space
    visited_chains = set()

    # Start from high-coverage fine bins that map to anchor coarse bins
    anchor_coarse = {b for b, s in stability.items() if s.stability_type == 'anchor'}

    starting_fine = []
    for fine_bin, bin_data in data.fine_bins.items():
        coarse = data.fine_to_coarse[fine_bin]
        if coarse in anchor_coarse and bin_data.count > data.total_pixels * 0.001:
            starting_fine.append((fine_bin, bin_data.count))

    starting_fine.sort(key=lambda x: -x[1])

    for start_fine, _ in starting_fine[:MAX_GRADIENT_SEARCH_SEEDS]:
        # Try to extend gradient in each LAB dimension
        for axis in [0, 1, 2]:  # L, a, b
            chain = trace_gradient_chain(start_fine, axis, fine_neighbors, data, fine_size)

            if len(chain) >= 3:
                # Convert to coarse stops (anchors only)
                chain_key = tuple(sorted(chain))
                if chain_key not in visited_chains:
                    visited_chains.add(chain_key)

                    gradient = build_gradient_from_chain(
                        chain, data, stability, families, fine_size
                    )
                    if gradient and len(gradient.stops) >= 2:
                        gradients.append(gradient)

    # Deduplicate and sort by coverage
    gradients = deduplicate_gradients(gradients)
    gradients.sort(key=lambda g: -g.coverage)

    return gradients[:MAX_NOTABLE_COLORS]


def trace_gradient_chain(start: tuple, axis: int, neighbors: dict,
                         data: PreparedData, fine_size: float) -> list:
    """Trace a gradient chain along one LAB axis."""
    chain = [start]
    visited = {start}
    current = start

    start_lab = np.array(start) * fine_size

    # Extend in positive direction
    while True:
        current_lab = np.array(current) * fine_size
        best_next = None
        best_progress = 0

        for neighbor in neighbors.get(current, []):
            if neighbor in visited:
                continue

            neighbor_lab = np.array(neighbor) * fine_size

            # Check monotonic progress along axis
            progress = neighbor_lab[axis] - current_lab[axis]
            if progress > fine_size * 0.5:  # Moving in positive direction
                # Check other axes don't change too much
                other_change = sum(abs(neighbor_lab[i] - current_lab[i])
                                   for i in range(3) if i != axis)
                if other_change < abs(progress) * 2:  # Primarily along main axis
                    if progress > best_progress:
                        best_progress = progress
                        best_next = neighbor

        if best_next is None:
            break

        chain.append(best_next)
        visited.add(best_next)
        current = best_next

        if len(chain) > MAX_GRADIENT_CHAIN_LENGTH:
            break

    # Extend in negative direction from start
    current = start
    prefix = []

    while True:
        current_lab = np.array(current) * fine_size
        best_prev = None
        best_progress = 0

        for neighbor in neighbors.get(current, []):
            if neighbor in visited:
                continue

            neighbor_lab = np.array(neighbor) * fine_size

            progress = current_lab[axis] - neighbor_lab[axis]
            if progress > fine_size * 0.5:
                other_change = sum(abs(neighbor_lab[i] - current_lab[i])
                                   for i in range(3) if i != axis)
                if other_change < abs(progress) * 2:
                    if progress > best_progress:
                        best_progress = progress
                        best_prev = neighbor

        if best_prev is None:
            break

        prefix.insert(0, best_prev)
        visited.add(best_prev)
        current = best_prev

        if len(prefix) > MAX_GRADIENT_CHAIN_LENGTH:
            break

    return prefix + chain


def build_gradient_from_chain(chain: list, data: PreparedData, stability: dict,
                               families: list, fine_size: float) -> Optional[GradientChain]:
    """Build a GradientChain from a list of fine bins."""
    if len(chain) < 3:
        return None

    # Map to coarse bins and filter to anchors
    coarse_chain = []
    seen_coarse = set()

    for fine_bin in chain:
        coarse = data.fine_to_coarse[fine_bin]
        if coarse not in seen_coarse:
            stab = stability.get(coarse)
            if stab and stab.stability_type == 'anchor':
                coarse_chain.append(coarse)
                seen_coarse.add(coarse)

    if len(coarse_chain) < 2:
        return None

    # Compute coverage
    coverage = sum(data.fine_bins[fb].count for fb in chain) / data.total_pixels

    # Compute LAB range
    fine_labs = [np.array(fb) * fine_size for fb in chain]
    L_vals = [lab[0] for lab in fine_labs]
    a_vals = [lab[1] for lab in fine_labs]
    b_vals = [lab[2] for lab in fine_labs]

    lab_range = {
        'L': (min(L_vals), max(L_vals)),
        'a': (min(a_vals), max(a_vals)),
        'b': (min(b_vals), max(b_vals))
    }

    # Determine direction based on dominant change
    L_range = lab_range['L'][1] - lab_range['L'][0]
    a_range = lab_range['a'][1] - lab_range['a'][0]
    b_range = lab_range['b'][1] - lab_range['b'][0]

    if L_range > max(a_range, b_range):
        direction = 'lightness'
    elif a_range > b_range:
        direction = 'green-red'
    else:
        direction = 'blue-yellow'

    # Find which families this spans
    family_span = []
    for family in families:
        for stop in coarse_chain:
            if stop in family.members:
                if family not in family_span:
                    family_span.append(family)
                break

    return GradientChain(
        stops=coarse_chain,
        fine_members=chain,
        direction=direction,
        coverage=coverage,
        lab_range=lab_range,
        family_span=[f.hue_center if not f.is_neutral else 'neutral' for f in family_span]
    )


def deduplicate_gradients(gradients: list) -> list:
    """Remove duplicate/overlapping gradients."""
    if not gradients:
        return []

    # Sort by coverage descending
    gradients.sort(key=lambda g: -g.coverage)

    kept = []
    for grad in gradients:
        stops_set = set(grad.stops)

        # Check overlap with already kept gradients
        is_duplicate = False
        for kept_grad in kept:
            kept_stops = set(kept_grad.stops)
            overlap = len(stops_set & kept_stops)
            min_len = min(len(stops_set), len(kept_stops))

            if overlap >= min_len * 0.7:  # 70% overlap = duplicate
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(grad)

    return kept


def extract_features(data: PreparedData) -> FeatureData:
    """Stage 2: Extract all features from prepared data."""
    stability = analyze_scale_stability(data)
    families = detect_color_families(data, stability)
    adjacency = build_adjacency_graph(data)
    metrics = compute_color_metrics(data, adjacency, stability)
    gradients = detect_gradients(data, stability, adjacency, families)

    return FeatureData(
        stability=stability,
        families=families,
        gradients=gradients,
        metrics=metrics,
        adjacency=adjacency
    )


# =============================================================================
# Stage 3: Synthesis
# =============================================================================

@dataclass
class NotableColor:
    """A significant color in the palette."""
    coarse_bin: tuple
    lab: np.ndarray
    hex: str
    rgb: tuple
    name: str
    role: str  # 'dominant', 'secondary', 'accent', 'dark', 'light'
    coverage: float
    chroma: float
    significance: float
    characteristics: list  # Descriptive strings
    gradient_membership: list  # Indices into gradients list


@dataclass
class ContrastPair:
    """A pair of colors with high contrast."""
    color_a: str  # Color name
    color_b: str
    delta_l: float
    contrast_ratio: float
    wcag_level: str  # 'AAA', 'AA', 'AA-large', 'fail'


@dataclass
class HarmonicPair:
    """A pair of colors with similar hue."""
    color_a: str
    color_b: str
    hue_difference: float


@dataclass
class SynthesisResult:
    """Output of Stage 3: Synthesis."""
    scheme_type: str
    scheme_description: str
    notable_colors: list  # List of NotableColor
    gradients: list  # GradientChain (filtered to significant ones)
    contrast_pairs: list  # ContrastPair
    harmonic_pairs: list  # HarmonicPair
    distribution_analysis: str
    lightness_range: tuple
    chroma_range: tuple


def compute_significance(metrics: ColorMetrics, stability: StabilityInfo,
                         median_chroma: float, chroma_iqr: float) -> float:
    """Compute significance score for a color.

    Coverage is the primary factor. Chroma only provides a small bonus
    for colors that stand out from a muted background - it shouldn't
    override high coverage.
    """
    # Base: coverage (0-100 scale) - this is the primary factor
    score = metrics.coverage * 100

    # Chroma bonus: modest bonus for standing out in muted image
    # Capped to avoid overwhelming coverage
    if chroma_iqr > 0 and metrics.chroma > median_chroma:
        chroma_bonus = min(5, (metrics.chroma - median_chroma) / (chroma_iqr + 1) * 2)
        score += chroma_bonus

    # Isolation bonus: only significant if has some coverage
    if metrics.coverage > 0.001:
        score += metrics.isolation * 3

    # Coherence bonus: forms a blob, not noise
    score += metrics.coherence * 5

    # Stability bonus
    if stability.stability_type == 'anchor':
        score += 3
    elif stability.stability_type == 'gradient':
        score += 1

    return score


def generate_color_name(lab: np.ndarray) -> str:
    """Generate a descriptive name from LAB coordinates."""
    L = lab[0]
    chroma = compute_chroma(lab)
    hue = compute_hue(lab)

    # Neutral colors
    if chroma < 8:
        if L < 15:
            return "Near-Black"
        elif L < 35:
            return "Dark Gray"
        elif L < 65:
            return "Gray"
        elif L < 85:
            return "Light Gray"
        else:
            return "Near-White"

    # Hue name
    if hue < 30 or hue >= 330:
        hue_name = "Red"
    elif hue < 60:
        hue_name = "Orange"
    elif hue < 90:
        hue_name = "Yellow"
    elif hue < 150:
        hue_name = "Green"
    elif hue < 210:
        hue_name = "Cyan"
    elif hue < 270:
        hue_name = "Blue"
    else:
        hue_name = "Purple"

    # Lightness modifier
    if L < 20:
        lightness_mod = "Deep "
    elif L < 40:
        lightness_mod = "Dark "
    elif L < 60:
        lightness_mod = ""
    elif L < 80:
        lightness_mod = "Light "
    else:
        lightness_mod = "Pale "

    # Chroma modifier
    if chroma < 15:
        chroma_mod = "Grayish "
    elif chroma < 30:
        chroma_mod = "Muted "
    elif chroma > 50:
        chroma_mod = "Vivid "
    else:
        chroma_mod = ""

    return f"{lightness_mod}{chroma_mod}{hue_name}".strip()


def classify_role(metrics: ColorMetrics, all_metrics: dict, is_accent: bool) -> str:
    """Classify the descriptive role of a color."""
    if is_accent:
        return "accent"

    # Find coverage rank
    coverages = sorted([m.coverage for m in all_metrics.values()], reverse=True)
    rank = coverages.index(metrics.coverage) if metrics.coverage in coverages else len(coverages)

    if rank == 0:
        return "dominant"
    elif rank <= 2:
        return "secondary"
    elif metrics.lab[0] < 30:
        return "dark"
    elif metrics.lab[0] > 70:
        return "light"
    else:
        return "secondary"


def generate_characteristics(metrics: ColorMetrics, stability: StabilityInfo,
                             role: str, gradient_count: int) -> list:
    """Generate descriptive characteristics for a color."""
    chars = []

    if role == "dominant":
        chars.append("Largest coverage, anchors the palette")
    elif role == "accent":
        chars.append("High chroma focal point")
        if metrics.isolation > 0.7:
            chars.append("spatially isolated")

    if stability.stability_type == 'anchor':
        chars.append("Stable across scales")
    elif stability.stability_type == 'texture':
        chars.append("Textured/varied region")

    if metrics.coherence > 0.7:
        chars.append("Forms coherent region")
    elif metrics.coherence < 0.3:
        chars.append("Scattered distribution")

    if gradient_count > 0:
        chars.append(f"Part of {gradient_count} gradient(s)")

    return chars


def compute_wcag_contrast(lab1: np.ndarray, lab2: np.ndarray) -> tuple:
    """
    Compute WCAG contrast ratio between two colors.
    Returns (ratio, level).
    """
    # Convert to relative luminance via RGB
    rgb1 = lab_to_rgb(lab1.reshape(1, -1))[0] / 255.0
    rgb2 = lab_to_rgb(lab2.reshape(1, -1))[0] / 255.0

    def relative_luminance(rgb):
        r, g, b = rgb
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    ratio = (lighter + 0.05) / (darker + 0.05)

    if ratio >= 7:
        level = "AAA"
    elif ratio >= 4.5:
        level = "AA"
    elif ratio >= 3:
        level = "AA-large"
    else:
        level = "fail"

    return ratio, level


def analyze_distribution(notable_colors: list) -> str:
    """Analyze how the color distribution compares to 60-30-10."""
    if not notable_colors:
        return "No significant colors detected"

    # Sort by coverage
    sorted_colors = sorted(notable_colors, key=lambda c: -c.coverage)

    dominant_coverage = sorted_colors[0].coverage * 100

    # Sum coverage by role
    accent_coverage = sum(c.coverage * 100 for c in notable_colors if c.role == 'accent')
    secondary_coverage = sum(c.coverage * 100 for c in notable_colors
                            if c.role in ('secondary', 'dark', 'light', 'dominant'))
    secondary_coverage -= dominant_coverage  # Don't double count dominant

    # Calculate top 3 coverage
    top3_coverage = sum(c.coverage * 100 for c in sorted_colors[:3])

    if dominant_coverage >= 15:
        if accent_coverage < 1:
            return f"Dominant-led: {dominant_coverage:.0f}% primary, {secondary_coverage:.0f}% supporting"
        else:
            return f"Structured: {dominant_coverage:.0f}% dominant, {secondary_coverage:.0f}% secondary, {accent_coverage:.1f}% accent"
    elif top3_coverage >= 40:
        return f"Tiered: Top 3 colors cover {top3_coverage:.0f}% of image"
    else:
        return f"Distributed: {len(notable_colors)} colors share coverage"


def determine_scheme_type(families: list, notable_colors: list) -> tuple:
    """Determine the color scheme type."""
    chromatic_families = [f for f in families if not f.is_neutral]
    neutral_family = next((f for f in families if f.is_neutral), None)

    # Check for accents
    accents = [c for c in notable_colors if c.role == 'accent']

    if not chromatic_families:
        return "achromatic", "Grayscale palette with no chromatic content"

    if len(chromatic_families) == 1:
        if accents and neutral_family and neutral_family.total_coverage > 0.5:
            return "neutral_accent", f"Neutral base with {chromatic_families[0].hue_center:.0f}° accent"
        else:
            return "monochromatic", f"Single hue family around {chromatic_families[0].hue_center:.0f}°"

    # Multiple chromatic families - analyze hue relationships
    hues = [f.hue_center for f in chromatic_families]

    if len(hues) == 2:
        hue_diff = circular_hue_distance(hues[0], hues[1])

        if hue_diff < 60:
            return "analogous", f"Adjacent hues ({hues[0]:.0f}° and {hues[1]:.0f}°)"
        elif 150 < hue_diff < 210:
            return "complementary", f"Opposing hues ({hues[0]:.0f}° and {hues[1]:.0f}°)"

    if len(hues) == 3:
        # Check for triadic (roughly 120° apart)
        hues_sorted = sorted(hues)
        diffs = [hues_sorted[1] - hues_sorted[0],
                 hues_sorted[2] - hues_sorted[1],
                 (360 + hues_sorted[0]) - hues_sorted[2]]

        if all(80 < d < 160 for d in diffs):
            return "triadic", "Three hues roughly 120° apart"

    # Check for analogous with accent
    if len(chromatic_families) >= 2:
        main_hues = sorted(hues)[:2]
        main_diff = circular_hue_distance(main_hues[0], main_hues[1])

        if main_diff < 60 and accents:
            return "analogous_accent", "Analogous base with contrasting accent"

    return "complex", f"Multi-hue palette with {len(chromatic_families)} color families"


def synthesize(data: PreparedData, features: FeatureData) -> SynthesisResult:
    """Stage 3: Synthesize features into actionable analysis."""

    # Compute chroma statistics for significance scoring
    chromas = [m.chroma for m in features.metrics.values()]
    median_chroma = np.median(chromas) if chromas else 0
    chroma_iqr = np.percentile(chromas, 75) - np.percentile(chromas, 25) if chromas else 1

    # Compute significance for all coarse bins
    significance_scores = {}
    for coarse_bin, metrics in features.metrics.items():
        stab = features.stability.get(coarse_bin, StabilityInfo('texture', 0, 0))
        significance_scores[coarse_bin] = compute_significance(
            metrics, stab, median_chroma, chroma_iqr
        )

    # Select notable colors (top by significance, minimum threshold)
    sorted_bins = sorted(significance_scores.items(), key=lambda x: -x[1])

    # Threshold: minimum significance relative to top color
    notable_threshold = max(1, sorted_bins[0][1] * MIN_SIGNIFICANCE_RATIO) if sorted_bins else 1

    notable_bins = [(b, s) for b, s in sorted_bins if s >= notable_threshold][:MAX_NOTABLE_COLORS]

    # Identify accents: high chroma + low coverage + high isolation
    # Must have at least some coverage to be notable (>0.1%)
    accent_bins = set()
    for coarse_bin, metrics in features.metrics.items():
        if (metrics.chroma > median_chroma + chroma_iqr and
            metrics.coverage < 0.05 and
            metrics.coverage > 0.001 and  # At least 0.1% coverage
            metrics.isolation > 0.3):
            accent_bins.add(coarse_bin)

    # Build NotableColor objects
    notable_colors = []
    for coarse_bin, sig_score in notable_bins:
        metrics = features.metrics[coarse_bin]
        stab = features.stability.get(coarse_bin, StabilityInfo('texture', 0, 0))

        is_accent = coarse_bin in accent_bins
        role = classify_role(metrics, features.metrics, is_accent)

        # Count gradient membership
        gradient_count = sum(1 for g in features.gradients if coarse_bin in g.stops)

        notable_colors.append(NotableColor(
            coarse_bin=coarse_bin,
            lab=metrics.lab,
            hex=lab_to_hex(metrics.lab),
            rgb=lab_to_rgb_tuple(metrics.lab),
            name=generate_color_name(metrics.lab),
            role=role,
            coverage=metrics.coverage,
            chroma=metrics.chroma,
            significance=sig_score,
            characteristics=generate_characteristics(metrics, stab, role, gradient_count),
            gradient_membership=[i for i, g in enumerate(features.gradients) if coarse_bin in g.stops]
        ))

    # Sort notable colors: dominant first, then by coverage
    role_order = {'dominant': 0, 'secondary': 1, 'dark': 2, 'light': 3, 'accent': 4}
    notable_colors.sort(key=lambda c: (role_order.get(c.role, 5), -c.coverage))

    # Find contrast pairs
    contrast_pairs = []
    for i, c1 in enumerate(notable_colors):
        for c2 in notable_colors[i+1:]:
            delta_l = abs(c1.lab[0] - c2.lab[0])
            if delta_l > 30:  # Meaningful contrast
                ratio, level = compute_wcag_contrast(c1.lab, c2.lab)
                contrast_pairs.append(ContrastPair(
                    color_a=c1.name,
                    color_b=c2.name,
                    delta_l=delta_l,
                    contrast_ratio=ratio,
                    wcag_level=level
                ))

    contrast_pairs.sort(key=lambda p: -p.contrast_ratio)
    contrast_pairs = contrast_pairs[:5]  # Top 5

    # Find harmonic pairs (similar hue) - limit to pairs with different names
    harmonic_pairs = []
    chromatic_notable = [c for c in notable_colors if c.chroma >= 15]
    seen_pairs = set()

    for i, c1 in enumerate(chromatic_notable):
        for c2 in chromatic_notable[i+1:]:
            # Skip if same name (different shades of same color)
            if c1.name == c2.name:
                continue

            hue1 = compute_hue(c1.lab)
            hue2 = compute_hue(c2.lab)
            hue_diff = circular_hue_distance(hue1, hue2)

            if hue_diff < HUE_CLUSTER_RANGE:
                # Deduplicate by name pair
                pair_key = tuple(sorted([c1.name, c2.name]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    harmonic_pairs.append(HarmonicPair(
                        color_a=c1.name,
                        color_b=c2.name,
                        hue_difference=hue_diff
                    ))

    harmonic_pairs = harmonic_pairs[:5]  # Limit to top 5

    # Determine scheme type
    scheme_type, scheme_desc = determine_scheme_type(features.families, notable_colors)

    # Lightness and chroma ranges
    all_L = [m.lab[0] for m in features.metrics.values()]
    all_chroma = [m.chroma for m in features.metrics.values()]

    return SynthesisResult(
        scheme_type=scheme_type,
        scheme_description=scheme_desc,
        notable_colors=notable_colors,
        gradients=features.gradients,
        contrast_pairs=contrast_pairs,
        harmonic_pairs=harmonic_pairs,
        distribution_analysis=analyze_distribution(notable_colors),
        lightness_range=(min(all_L), max(all_L)),
        chroma_range=(min(all_chroma), max(all_chroma))
    )


# =============================================================================
# Stage 4: Render
# =============================================================================

def render(synthesis: SynthesisResult, features: FeatureData) -> str:
    """Stage 4: Render synthesis result as prose."""
    lines = []

    # Header
    lines.append(f"SCHEME: {synthesis.scheme_type}")
    lines.append(synthesis.scheme_description)
    lines.append(f"Lightness range: {synthesis.lightness_range[0]:.0f}-{synthesis.lightness_range[1]:.0f} | "
                 f"Chroma range: {synthesis.chroma_range[0]:.0f}-{synthesis.chroma_range[1]:.0f}")
    lines.append(f"Notable colors: {len(synthesis.notable_colors)}")
    lines.append("")

    # Colors section
    lines.append("COLORS:")
    lines.append("")

    for color in synthesis.notable_colors:
        lines.append(f"[{color.role.capitalize()}] {color.name}")
        lines.append(f"  Hex: {color.hex} | RGB: {color.rgb} | LAB: ({color.lab[0]:.0f}, {color.lab[1]:.0f}, {color.lab[2]:.0f})")
        coverage_pct = color.coverage * 100
        if coverage_pct >= 0.1:
            lines.append(f"  Coverage: {coverage_pct:.1f}% | Chroma: {color.chroma:.0f}")
        else:
            lines.append(f"  Coverage: <0.1% | Chroma: {color.chroma:.0f}")
        if color.characteristics:
            lines.append(f"  {'. '.join(color.characteristics)}.")
        lines.append("")

    # Gradients section
    if synthesis.gradients:
        lines.append("GRADIENTS:")
        lines.append("")

        for i, grad in enumerate(synthesis.gradients):
            # Build stop names
            stop_names = []
            for stop in grad.stops:
                metrics = features.metrics.get(stop)
                if metrics:
                    stop_names.append(generate_color_name(metrics.lab))
                else:
                    stop_names.append("Unknown")

            lines.append(f"{' → '.join(stop_names)}")
            lines.append(f"  Stops:")

            for j, stop in enumerate(grad.stops):
                metrics = features.metrics.get(stop)
                if metrics:
                    hex_val = lab_to_hex(metrics.lab)
                    rgb_val = lab_to_rgb_tuple(metrics.lab)
                    lab_val = metrics.lab
                    name = generate_color_name(lab_val)
                    lines.append(f"    {j+1}. {name} {hex_val} / RGB{rgb_val} / LAB({lab_val[0]:.0f}, {lab_val[1]:.0f}, {lab_val[2]:.0f})")

            lines.append(f"  Direction: {grad.direction} | Coverage: {grad.coverage*100:.1f}%")

            L_range = grad.lab_range['L'][1] - grad.lab_range['L'][0]
            lines.append(f"  Lightness span: {L_range:.0f} (L {grad.lab_range['L'][0]:.0f} → {grad.lab_range['L'][1]:.0f})")
            lines.append("")

    # Relationships section
    lines.append("RELATIONSHIPS:")
    lines.append("")

    if synthesis.contrast_pairs:
        lines.append("Contrast pairs (figural - good for emphasis):")
        for pair in synthesis.contrast_pairs:
            lines.append(f"  - {pair.color_a} / {pair.color_b}: "
                        f"Ratio {pair.contrast_ratio:.1f}:1 (WCAG {pair.wcag_level}) | ΔL={pair.delta_l:.0f}")
        lines.append("")

    if synthesis.harmonic_pairs:
        lines.append("Harmonic pairs (cohesive - good for backgrounds):")
        for pair in synthesis.harmonic_pairs:
            lines.append(f"  - {pair.color_a} and {pair.color_b}: "
                        f"Similar hue ({pair.hue_difference:.0f}° apart)")
        lines.append("")

    lines.append(f"Distribution: {synthesis.distribution_analysis}")

    return "\n".join(lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def analyze_image(image_path: str) -> str:
    """Run the full analysis pipeline on an image."""
    # Stage 1: Data Preparation
    data = prepare_data(image_path)

    # Stage 2: Feature Extraction
    features = extract_features(data)

    # Stage 3: Synthesis
    synthesis = synthesize(data, features)

    # Stage 4: Render
    output = render(synthesis, features)

    return output


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze.py <image_path>")
        print("       python analyze.py --batch <directory>")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        from pathlib import Path

        directory = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('source_images')

        for image_path in sorted(directory.glob('*.jpeg')) + sorted(directory.glob('*.png')) + sorted(directory.glob('*.jpg')):
            print(f"\n{'='*70}")
            print(f"IMAGE: {image_path.name}")
            print('='*70)
            print()

            result = analyze_image(str(image_path))
            print(result)
    else:
        result = analyze_image(sys.argv[1])
        print(result)
