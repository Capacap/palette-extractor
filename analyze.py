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


def build_adjacency_graph(data: PreparedData) -> tuple[dict, dict]:
    """
    Build adjacency graph at fine scale with 8-connectivity.

    Returns:
        adjacency: {(b1, b2): count} - symmetric edge counts
        directional: {(b1, b2): {'right': n, 'left': n, ...}} - directional counts
    """
    # Direction mappings: (dy, dx) -> direction name
    # Direction is "where b2 is relative to b1"
    DIRECTIONS = {
        (-1, -1): 'up_left',    (-1, 0): 'above',    (-1, 1): 'up_right',
        (0, -1): 'left',                              (0, 1): 'right',
        (1, -1): 'down_left',   (1, 0): 'below',     (1, 1): 'down_right'
    }
    ALL_DIRS = list(DIRECTIONS.values())

    adjacency = {}
    directional = {}

    # Rebuild fine binned image for adjacency
    fine_binned = {}
    for fine_bin, bin_data in data.fine_bins.items():
        for pos in bin_data.positions:
            fine_binned[pos] = fine_bin

    # Check all 8 neighbors for each pixel
    for (y, x), b1 in fine_binned.items():
        for (dy, dx), dir_name in DIRECTIONS.items():
            ny, nx = y + dy, x + dx
            if (ny, nx) not in fine_binned:
                continue

            b2 = fine_binned[(ny, nx)]
            if b1 == b2:
                continue

            # Symmetric count
            sym_key = (b1, b2) if b1 < b2 else (b2, b1)
            adjacency[sym_key] = adjacency.get(sym_key, 0) + 1

            # Directional count: b2 is in dir_name direction from b1
            dir_key = (b1, b2)
            if dir_key not in directional:
                directional[dir_key] = {d: 0 for d in ALL_DIRS}
            directional[dir_key][dir_name] += 1

    return adjacency, directional


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


def compute_fine_coherence(data: PreparedData, coarse_coherence: dict) -> dict:
    """
    Map fine bins to their parent coarse bin's coherence.

    Computing coherence per fine bin is expensive (O(n) label operations).
    Instead, use the coarse bin's coherence as a proxy — fine bins inherit
    their parent's coherence score.
    """
    coherence = {}
    for fine_bin in data.fine_bins:
        coarse = data.fine_to_coarse.get(fine_bin)
        if coarse:
            coherence[fine_bin] = coarse_coherence.get(coarse, 0.0)
        else:
            coherence[fine_bin] = 0.0
    return coherence


def analyze_gradient_flow(directional: dict, min_total: int = 50) -> dict:
    """
    Analyze directional adjacency to find gradient flow patterns.

    For each color pair, computes:
    - h_flow: right - left (positive = flows right)
    - v_flow: below - above (positive = flows down)
    - Asymmetry: how one-directional the flow is (0-1)

    Args:
        directional: {(b1, b2): {'right': n, 'left': n, ...}}
        min_total: minimum total edge count to include pair

    Returns:
        {(b1, b2): {'h_flow': n, 'v_flow': n, 'h_asymmetry': f, ...}}
    """
    flow_analysis = {}

    for (b1, b2), dirs in directional.items():
        total = sum(dirs.values())
        if total < min_total:
            continue

        # Horizontal flow (positive = b2 is right of b1)
        h_flow = dirs['right'] - dirs['left']
        # Vertical flow (positive = b2 is below b1)
        v_flow = dirs['below'] - dirs['above']
        # Diagonal flows
        dr_flow = dirs['down_right'] - dirs['up_left']
        dl_flow = dirs['down_left'] - dirs['up_right']

        # Compute asymmetry (how one-directional)
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
            'total': total
        }

    return flow_analysis


def detect_flow_gradients(data: PreparedData, directional: dict,
                          fine_coherence: dict, families: list,
                          min_chain_length: int = 3,
                          min_asymmetry: float = 0.3) -> list:
    """
    Detect gradients by following directional flow through the adjacency graph.

    A gradient is a chain where colors consistently flow in one spatial direction:
    - Horizontal: each color has the next color predominantly to its right (or left)
    - Vertical: each color has the next color predominantly below (or above)
    - Diagonal: flows along diagonals

    This approach can capture multi-hue gradients (e.g., red→orange→yellow)
    as long as they're spatially smooth.

    Args:
        data: Prepared image data
        directional: Directional adjacency from build_adjacency_graph
        fine_coherence: Per-fine-bin coherence scores
        families: Color families for span detection
        min_chain_length: Minimum colors to form a gradient
        min_asymmetry: Minimum flow asymmetry to follow an edge

    Returns:
        List of GradientChain objects
    """
    fine_size = FINE_SCALE * JND
    colors = list(data.fine_bins.keys())

    # Analyze flow patterns
    # Lower threshold because fine bins have fewer pixels per edge than coarse bins
    flow = analyze_gradient_flow(directional, min_total=10)

    # Build directed flow graph
    # Edge from A to B if B is predominantly in one direction from A
    ALL_DIRECTIONS = ['right', 'left', 'below', 'above',
                      'down_right', 'down_left', 'up_right', 'up_left']
    flow_graph = {b: {d: [] for d in ALL_DIRECTIONS} for b in colors}

    def get_coherence(b):
        return fine_coherence.get(b, 1.0)

    for (b1, b2), analysis in flow.items():
        if b1 not in flow_graph or b2 not in flow_graph:
            continue

        # Weight by geometric mean of coherences
        coh1, coh2 = get_coherence(b1), get_coherence(b2)
        coherence_weight = np.sqrt(coh1 * coh2)

        # Check horizontal flow
        if analysis['h_asymmetry'] > min_asymmetry:
            if analysis['h_flow'] > 0:
                weighted = analysis['h_flow'] * coherence_weight
                flow_graph[b1]['right'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['h_flow'] * coherence_weight
                flow_graph[b1]['left'].append((b2, weighted, analysis['total']))

        # Check vertical flow
        if analysis['v_asymmetry'] > min_asymmetry:
            if analysis['v_flow'] > 0:
                weighted = analysis['v_flow'] * coherence_weight
                flow_graph[b1]['below'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['v_flow'] * coherence_weight
                flow_graph[b1]['above'].append((b2, weighted, analysis['total']))

        # Check diagonal flows
        if analysis['dr_asymmetry'] > min_asymmetry:
            if analysis['dr_flow'] > 0:
                weighted = analysis['dr_flow'] * coherence_weight
                flow_graph[b1]['down_right'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['dr_flow'] * coherence_weight
                flow_graph[b1]['up_left'].append((b2, weighted, analysis['total']))

        if analysis['dl_asymmetry'] > min_asymmetry:
            if analysis['dl_flow'] > 0:
                weighted = analysis['dl_flow'] * coherence_weight
                flow_graph[b1]['down_left'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['dl_flow'] * coherence_weight
                flow_graph[b1]['up_right'].append((b2, weighted, analysis['total']))

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

    # Compute seed scores combining multiple factors
    total_pixels = data.total_pixels

    # Find LAB extremes for bonus scoring
    all_labs = {b: np.array(b) * fine_size for b in colors}
    l_values = {b: lab[0] for b, lab in all_labs.items()}
    a_values = {b: lab[1] for b, lab in all_labs.items()}
    b_values = {b: lab[2] for b, lab in all_labs.items()}

    l_min_bin = min(l_values, key=l_values.get)
    l_max_bin = max(l_values, key=l_values.get)
    a_min_bin = min(a_values, key=a_values.get)
    a_max_bin = max(a_values, key=a_values.get)
    b_min_bin = min(b_values, key=b_values.get)
    b_max_bin = max(b_values, key=b_values.get)
    extreme_bins = {l_min_bin, l_max_bin, a_min_bin, a_max_bin, b_min_bin, b_max_bin}

    # Compute local contrast for each color (avg LAB distance to flow neighbors)
    local_contrast = {}
    for b in colors:
        neighbors = set()
        for direction in ALL_DIRECTIONS:
            for neighbor, _, _ in flow_graph[b][direction]:
                neighbors.add(neighbor)
        if neighbors:
            b_lab = all_labs[b]
            distances = [np.linalg.norm(b_lab - all_labs[n]) for n in neighbors if n in all_labs]
            local_contrast[b] = np.mean(distances) if distances else 0
        else:
            local_contrast[b] = 0

    # Normalize contrast (typically 0-30 LAB units range)
    max_contrast = max(local_contrast.values()) if local_contrast else 1
    contrast_normalized = {b: c / max_contrast for b, c in local_contrast.items()}

    def compute_seed_score(b):
        # Coverage: scale so 10% coverage ≈ 1.0
        cov = data.fine_bins[b].count / total_pixels
        cov_score = cov * 10

        # Contrast boost: up to 0.3 for high contrast colors
        contrast_score = contrast_normalized.get(b, 0) * 0.3

        # LAB extreme boost: bonus for colors at extremes
        extreme_score = 0.2 if b in extreme_bins else 0

        # Coherence weight: reduce score for scattered/noisy colors
        coherence = fine_coherence.get(b, 0.5)

        return (cov_score + contrast_score + extreme_score) * coherence

    scored_colors = [(b, compute_seed_score(b)) for b in colors]
    scored_colors.sort(key=lambda x: x[1], reverse=True)

    all_gradients = []

    # Try all directions from top-scored colors
    for direction in ALL_DIRECTIONS:
        for start, _ in scored_colors[:MAX_GRADIENT_SEARCH_SEEDS]:
            chain = follow_flow(start, direction)

            if len(chain) >= min_chain_length:
                chain_coverage = sum(data.fine_bins[b].count for b in chain) / total_pixels

                # Compute LAB bounds
                labs = [np.array(b) * fine_size for b in chain]
                lab_array = np.array(labs)
                l_min, l_max = lab_array[:, 0].min(), lab_array[:, 0].max()
                a_min, a_max = lab_array[:, 1].min(), lab_array[:, 1].max()
                b_min, b_max = lab_array[:, 2].min(), lab_array[:, 2].max()

                all_gradients.append({
                    'chain': chain,
                    'direction': direction,
                    'coverage': chain_coverage,
                    'l_range': l_max - l_min,
                    'a_range': a_max - a_min,
                    'b_range': b_max - b_min,
                    'l_bounds': (l_min, l_max),
                    'a_bounds': (a_min, a_max),
                    'b_bounds': (b_min, b_max),
                    'score': len(chain) * chain_coverage
                })

    # Sort by score
    all_gradients.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate by LAB bounds overlap
    def bounds_overlap(bounds1, bounds2):
        """Compute overlap ratio: how much of bounds1 is inside bounds2."""
        min1, max1 = bounds1
        min2, max2 = bounds2
        range1 = max1 - min1
        if range1 < 0.1:
            return 1.0 if min2 <= min1 <= max2 else 0.0
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        overlap = max(0, overlap_max - overlap_min)
        return overlap / range1

    final_gradients = []
    for grad in all_gradients:
        is_redundant = False
        for existing in final_gradients:
            l_overlap = bounds_overlap(grad['l_bounds'], existing['l_bounds'])
            a_overlap = bounds_overlap(grad['a_bounds'], existing['a_bounds'])
            b_overlap = bounds_overlap(grad['b_bounds'], existing['b_bounds'])
            if l_overlap >= 0.5 and a_overlap >= 0.5 and b_overlap >= 0.5:
                is_redundant = True
                break
        if not is_redundant:
            final_gradients.append(grad)

    # Convert to GradientChain objects
    result = []
    for grad in final_gradients[:MAX_NOTABLE_COLORS]:
        # Map spatial direction to display direction
        dir_name = grad['direction']
        if dir_name in ('right', 'left'):
            display_dir = 'horizontal'
        elif dir_name in ('below', 'above'):
            display_dir = 'vertical'
        elif dir_name in ('down_right', 'up_left'):
            display_dir = 'diagonal'
        else:
            display_dir = 'anti-diagonal'

        # Map fine bins to coarse bins for stops
        coarse_stops = []
        seen_coarse = set()
        for fine_bin in grad['chain']:
            coarse = data.fine_to_coarse.get(fine_bin)
            if coarse and coarse not in seen_coarse:
                coarse_stops.append(coarse)
                seen_coarse.add(coarse)

        if len(coarse_stops) < 2:
            continue

        # Filter out low-variation gradients (look like solid blocks)
        l_span = grad['l_bounds'][1] - grad['l_bounds'][0]
        a_span = grad['a_bounds'][1] - grad['a_bounds'][0]
        b_span = grad['b_bounds'][1] - grad['b_bounds'][0]
        total_span = l_span + a_span + b_span
        if total_span < 40:  # ~17 JND minimum variation
            continue

        # Determine which color families this gradient spans
        family_indices = set()
        for stop in coarse_stops:
            lab = data.coarse_bins[stop].lab
            hue = compute_hue(lab)
            chroma = compute_chroma(lab)
            # Skip neutral colors (low chroma)
            if chroma < 8:
                continue
            for i, family in enumerate(families):
                if family.is_neutral:
                    continue
                if circular_hue_distance(hue, family.hue_center) <= HUE_CLUSTER_RANGE:
                    family_indices.add(i)
                    break

        result.append(GradientChain(
            stops=coarse_stops,
            fine_members=grad['chain'],
            direction=display_dir,
            coverage=grad['coverage'],
            lab_range={
                'L': grad['l_bounds'],
                'a': grad['a_bounds'],
                'b': grad['b_bounds']
            },
            family_span=list(family_indices)
        ))

    return result


def extract_features(data: PreparedData) -> FeatureData:
    """Stage 2: Extract all features from prepared data."""
    stability = analyze_scale_stability(data)
    families = detect_color_families(data, stability)
    adjacency, directional = build_adjacency_graph(data)
    metrics = compute_color_metrics(data, adjacency, stability)

    # Extract coarse coherence from metrics, then map to fine bins
    coarse_coherence = {b: m.coherence for b, m in metrics.items()}
    fine_coherence = compute_fine_coherence(data, coarse_coherence)

    gradients = detect_flow_gradients(data, directional, fine_coherence, families)

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


def text_color_for_background(L: float) -> str:
    """Return black or white text color based on background lightness."""
    return "#000" if L > 50 else "#fff"


def render_html(synthesis: SynthesisResult, features: FeatureData, image_path: str) -> str:
    """Stage 4b: Render synthesis result as HTML."""
    from html import escape

    safe_path = escape(image_path)

    # Build name-to-hex lookup once for contrast/harmonic pairs
    name_to_hex = {c.name: c.hex for c in synthesis.notable_colors}

    # CSS styles
    css = """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.5;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.2rem; margin: 2rem 0 1rem; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; }
        .meta { color: #666; font-size: 0.9rem; margin-bottom: 1rem; }
        .palette-strip {
            display: flex;
            height: 80px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1.5rem 0;
        }
        .palette-strip .swatch {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding: 0.5rem;
            font-size: 0.7rem;
            font-weight: 500;
        }
        .color-card {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            display: grid;
            grid-template-columns: 60px 1fr;
            gap: 1rem;
        }
        .color-card .swatch {
            width: 60px;
            height: 60px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            font-weight: 600;
        }
        .color-card .info { font-size: 0.85rem; }
        .color-card .role { font-weight: 600; text-transform: capitalize; }
        .color-card .name { color: #666; }
        .color-card .values { font-family: monospace; color: #555; font-size: 0.8rem; }
        .color-card .chars { font-style: italic; color: #777; margin-top: 0.25rem; }
        .gradient-block {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .gradient-bar {
            height: 40px;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .gradient-stops {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .gradient-stop {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
        }
        .gradient-stop .swatch {
            width: 24px;
            height: 24px;
            border-radius: 4px;
        }
        .gradient-meta { font-size: 0.85rem; color: #666; margin-top: 0.75rem; }
        .contrast-pair {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .contrast-demo {
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .contrast-demo .sample { font-weight: 600; }
        .contrast-info { font-size: 0.85rem; color: #666; }
        .contrast-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        .badge-aaa { background: #22c55e; color: #fff; }
        .badge-aa { background: #3b82f6; color: #fff; }
        .badge-aa-large { background: #f59e0b; color: #fff; }
        .badge-fail { background: #ef4444; color: #fff; }
        .harmonic-pair {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        .harmonic-pair .swatch {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
        .distribution { font-size: 0.9rem; color: #555; margin-top: 1rem; }
    """

    # Build HTML
    lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'  <title>Palette: {safe_path}</title>',
        f'  <style>{css}</style>',
        '</head>',
        '<body>',
    ]

    # Header
    lines.append(f'<h1>{synthesis.scheme_type}</h1>')
    lines.append(f'<p class="meta">{synthesis.scheme_description}</p>')
    lines.append(f'<p class="meta">Source: {safe_path}</p>')
    lines.append(f'<p class="meta">Lightness: {synthesis.lightness_range[0]:.0f}–{synthesis.lightness_range[1]:.0f} | '
                 f'Chroma: {synthesis.chroma_range[0]:.0f}–{synthesis.chroma_range[1]:.0f} | '
                 f'{len(synthesis.notable_colors)} colors</p>')

    # Palette strip
    lines.append('<div class="palette-strip">')
    total_coverage = sum(c.coverage for c in synthesis.notable_colors) or 1
    for color in synthesis.notable_colors:
        width_pct = max(5, (color.coverage / total_coverage) * 100)  # min 5% for visibility
        text_color = text_color_for_background(color.lab[0])
        lines.append(f'  <div class="swatch" style="background:{color.hex}; color:{text_color}; flex:{width_pct:.1f}">{color.hex}</div>')
    lines.append('</div>')

    # Color details
    lines.append('<h2>Colors</h2>')
    for color in synthesis.notable_colors:
        text_color = text_color_for_background(color.lab[0])
        coverage_str = f"{color.coverage*100:.1f}%" if color.coverage >= 0.001 else "<0.1%"
        lines.append('<div class="color-card">')
        lines.append(f'  <div class="swatch" style="background:{color.hex}; color:{text_color}">{color.hex}</div>')
        lines.append('  <div class="info">')
        lines.append(f'    <span class="role">{color.role}</span> <span class="name">{color.name}</span>')
        lines.append(f'    <div class="values">RGB{color.rgb} · LAB({color.lab[0]:.0f}, {color.lab[1]:.0f}, {color.lab[2]:.0f})</div>')
        lines.append(f'    <div class="values">Coverage: {coverage_str} · Chroma: {color.chroma:.0f}</div>')
        if color.characteristics:
            lines.append(f'    <div class="chars">{". ".join(color.characteristics)}.</div>')
        lines.append('  </div>')
        lines.append('</div>')

    # Gradients
    if synthesis.gradients:
        lines.append('<h2>Gradients</h2>')
        fine_size = FINE_SCALE * JND

        for grad in synthesis.gradients:
            # Subsample fine_members for cleaner CSS gradient (max ~20 stops)
            max_css_stops = 20
            fine_members = grad.fine_members
            if len(fine_members) > max_css_stops:
                # Evenly sample across the chain
                indices = [int(i * (len(fine_members) - 1) / (max_css_stops - 1))
                          for i in range(max_css_stops)]
                fine_members = [fine_members[i] for i in indices]

            # Build CSS gradient from subsampled fine_members
            fine_stops_css = []
            for i, fine_bin in enumerate(fine_members):
                lab = np.array(fine_bin) * fine_size
                hex_val = lab_to_hex(lab)
                pct = (i / max(1, len(fine_members) - 1)) * 100
                fine_stops_css.append(f"{hex_val} {pct:.0f}%")

            # Build labeled stops from coarse bins (representative colors)
            stop_info = []
            for stop in grad.stops:
                metrics = features.metrics.get(stop)
                if metrics:
                    hex_val = lab_to_hex(metrics.lab)
                    name = generate_color_name(metrics.lab)
                    stop_info.append((hex_val, name, metrics.lab))

            # Skip if no valid stops found
            if not fine_stops_css:
                continue

            # Always render horizontally for easier visual comparison
            # (actual spatial direction shown in metadata below)
            gradient_css = f"linear-gradient(to right, {', '.join(fine_stops_css)})"

            lines.append('<div class="gradient-block">')
            lines.append(f'  <div class="gradient-bar" style="background:{gradient_css}"></div>')
            lines.append('  <div class="gradient-stops">')
            for hex_val, name, _ in stop_info:
                lines.append(f'    <div class="gradient-stop"><div class="swatch" style="background:{hex_val}"></div>{name}</div>')
            lines.append('  </div>')
            L_range = grad.lab_range['L'][1] - grad.lab_range['L'][0]
            lines.append(f'  <div class="gradient-meta">Direction: {grad.direction} · Coverage: {grad.coverage*100:.1f}% · '
                        f'Lightness span: {L_range:.0f}</div>')
            lines.append('</div>')

    # Relationships
    lines.append('<h2>Relationships</h2>')

    # Contrast pairs
    if synthesis.contrast_pairs:
        lines.append('<h3 style="font-size:1rem; margin:1rem 0 0.5rem;">Contrast Pairs</h3>')
        for pair in synthesis.contrast_pairs:
            bg_hex = name_to_hex.get(pair.color_a, '#888')
            fg_hex = name_to_hex.get(pair.color_b, '#fff')

            badge_class = {
                'AAA': 'badge-aaa',
                'AA': 'badge-aa',
                'AA-large': 'badge-aa-large',
            }.get(pair.wcag_level, 'badge-fail')

            lines.append('<div class="contrast-pair">')
            lines.append(f'  <div class="contrast-demo" style="background:{bg_hex}; color:{fg_hex}">')
            lines.append(f'    <span class="sample">Aa</span> Sample text for readability')
            lines.append('  </div>')
            lines.append(f'  <div class="contrast-info">{pair.color_a} / {pair.color_b}: '
                        f'{pair.contrast_ratio:.1f}:1 <span class="contrast-badge {badge_class}">{pair.wcag_level}</span></div>')
            lines.append('</div>')

    # Harmonic pairs
    if synthesis.harmonic_pairs:
        lines.append('<h3 style="font-size:1rem; margin:1rem 0 0.5rem;">Harmonic Pairs</h3>')
        for pair in synthesis.harmonic_pairs:
            hex_a = name_to_hex.get(pair.color_a, '#888')
            hex_b = name_to_hex.get(pair.color_b, '#888')
            lines.append('<div class="harmonic-pair">')
            lines.append(f'  <div class="swatch" style="background:{hex_a}"></div>')
            lines.append(f'  <div class="swatch" style="background:{hex_b}"></div>')
            lines.append(f'  <span>{pair.color_a} and {pair.color_b}: {pair.hue_difference:.0f}° apart</span>')
            lines.append('</div>')

    # Distribution
    lines.append(f'<p class="distribution"><strong>Distribution:</strong> {synthesis.distribution_analysis}</p>')

    lines.append('</body>')
    lines.append('</html>')

    return '\n'.join(lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(image_path: str) -> tuple[SynthesisResult, FeatureData]:
    """Run analysis pipeline stages 1-3.

    Returns:
        Tuple of (synthesis_result, feature_data) for rendering.
    """
    # Stage 1: Data Preparation
    data = prepare_data(image_path)

    # Stage 2: Feature Extraction
    features = extract_features(data)

    # Stage 3: Synthesis
    synthesis = synthesize(data, features)

    return synthesis, features


def analyze_image(image_path: str) -> tuple[str, str]:
    """Run the full analysis pipeline on an image.

    Returns:
        Tuple of (prose_output, html_output)
    """
    synthesis, features = run_pipeline(image_path)
    prose = render(synthesis, features)
    html = render_html(synthesis, features, image_path)
    return prose, html


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Analyze an image and extract its color palette.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the image file'
    )
    parser.add_argument(
        '--output', '-o',
        nargs='?',
        const=True,
        default=None,
        help='Write HTML report. Optionally specify path, otherwise auto-names from input.'
    )

    args = parser.parse_args()
    image_path = Path(args.input)

    # Run analysis
    try:
        prose, html = analyze_image(str(image_path))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing image: {e}", file=sys.stderr)
        sys.exit(1)

    # Always print prose to terminal
    print(prose)

    # Write HTML if requested
    if args.output:
        if args.output is True:
            output_path = image_path.with_name(f"{image_path.stem}-palette.html")
        else:
            output_path = Path(args.output)

        try:
            output_path.write_text(html)
            print(f"\nWrote: {output_path}")
        except OSError as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            sys.exit(1)
