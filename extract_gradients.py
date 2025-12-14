#!/usr/bin/env python3
"""
Extract gradients from images.

Approach:
1. Segment image into brightness regions (L channel bands)
2. Quantize colors within each region to 3x JND
3. Build adjacency graph with LAB distance cutoff per region
4. Find gradient chains within each region
"""

import numpy as np
from PIL import Image
from pathlib import Path

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


def build_adjacency(binned: np.ndarray, max_lab_distance: float = 30.0, scale: float = 3.0) -> dict:
    """
    Build adjacency graph with distance cutoff.

    Only counts adjacencies where the two colors are within max_lab_distance
    in LAB space. This filters out hard edges between unrelated colors.
    """
    h, w = binned.shape[:2]
    adjacency = {}

    def add_edge(b1, b2):
        # Convert to LAB and check distance
        lab1 = bin_to_lab(b1, scale)
        lab2 = bin_to_lab(b2, scale)
        dist = np.linalg.norm(lab1 - lab2)

        if dist > max_lab_distance:
            return  # Too far apart, not a gradient transition

        key = (b1, b2) if b1 < b2 else (b2, b1)
        adjacency[key] = adjacency.get(key, 0) + 1

    # Check horizontal neighbors
    for y in range(h):
        for x in range(w - 1):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y, x + 1])
            if b1 != b2:
                add_edge(b1, b2)

    # Check vertical neighbors
    for y in range(h - 1):
        for x in range(w):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y + 1, x])
            if b1 != b2:
                add_edge(b1, b2)

    return adjacency


def get_bin_coverage(binned: np.ndarray) -> dict:
    """Get pixel count for each bin."""
    flat = binned.reshape(-1, 3)
    unique, counts = np.unique(flat, axis=0, return_counts=True)
    return {tuple(b): c for b, c in zip(unique, counts)}


def segment_by_brightness(lab: np.ndarray, num_bands: int = 5) -> list[dict]:
    """
    Segment image into brightness regions based on L channel.

    Returns list of regions, each containing:
    - mask: boolean array of pixels in this region
    - L_range: (min, max) lightness values
    - name: human-readable name
    """
    L = lab[:, :, 0]
    L_min, L_max = L.min(), L.max()

    # Create adaptive bands based on actual L distribution
    # Use percentiles for more even distribution
    percentiles = np.linspace(0, 100, num_bands + 1)
    thresholds = np.percentile(L, percentiles)

    regions = []
    band_names = ['deep shadow', 'shadow', 'midtone', 'highlight', 'bright']

    for i in range(num_bands):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == num_bands - 1:
            mask = (L >= lo) & (L <= hi)
        else:
            mask = (L >= lo) & (L < hi)

        if mask.sum() > 0:
            regions.append({
                'mask': mask,
                'L_range': (lo, hi),
                'name': band_names[i] if i < len(band_names) else f'band_{i}'
            })

    return regions


def cluster_by_chromaticity(lab: np.ndarray, mask: np.ndarray, num_clusters: int = 2) -> list[dict]:
    """
    Sub-cluster a brightness region by chromaticity (a, b channels).

    Uses k-means on the (a, b) values to separate warm from cool colors.
    Returns list of sub-regions with masks and descriptive names.
    """
    from scipy.cluster.vq import kmeans2

    # Get (a, b) values for masked pixels
    ab = lab[:, :, 1:3]  # shape (h, w, 2)
    masked_ab = ab[mask].astype(np.float64)

    if len(masked_ab) < num_clusters:
        return [{'mask': mask, 'name': 'mixed', 'centroid': (0, 0)}]

    # K-means clustering on chromaticity
    centroids, labels = kmeans2(masked_ab, num_clusters, minit='++')

    # Create sub-masks
    h, w = mask.shape
    flat_mask = mask.flatten()
    sub_regions = []

    for i in range(num_clusters):
        # Build mask for this cluster
        sub_mask = np.zeros(h * w, dtype=bool)
        sub_mask[flat_mask] = (labels == i)
        sub_mask = sub_mask.reshape(h, w)

        if sub_mask.sum() == 0:
            continue

        # Name based on centroid position in a,b space
        a_cent, b_cent = centroids[i]
        # a: negative=green, positive=red
        # b: negative=blue, positive=yellow
        warmth = a_cent + b_cent * 0.5  # weighted toward red/yellow

        if warmth > 5:
            temp = 'warm'
        elif warmth < -5:
            temp = 'cool'
        else:
            temp = 'neutral'

        sub_regions.append({
            'mask': sub_mask,
            'name': temp,
            'centroid': (a_cent, b_cent)
        })

    # Sort by warmth (cool first, warm last)
    sub_regions.sort(key=lambda r: r['centroid'][0] + r['centroid'][1] * 0.5)
    return sub_regions


def find_cross_region_adjacencies(binned: np.ndarray, mask1: np.ndarray, mask2: np.ndarray,
                                   max_lab_distance: float = 12.0, scale: float = 3.0) -> dict:
    """
    Find adjacencies between two regions (e.g., warm and cool).
    These represent transition zones where gradients cross chromaticity boundaries.
    """
    h, w = binned.shape[:2]
    adjacency = {}

    def add_edge(b1, b2):
        lab1 = bin_to_lab(b1, scale)
        lab2 = bin_to_lab(b2, scale)
        dist = np.linalg.norm(lab1 - lab2)

        if dist > max_lab_distance:
            return

        key = (b1, b2) if b1 < b2 else (b2, b1)
        adjacency[key] = adjacency.get(key, 0) + 1

    # Check horizontal neighbors (one in mask1, one in mask2)
    for y in range(h):
        for x in range(w - 1):
            if (mask1[y, x] and mask2[y, x + 1]) or (mask2[y, x] and mask1[y, x + 1]):
                b1 = tuple(binned[y, x])
                b2 = tuple(binned[y, x + 1])
                if b1 != b2:
                    add_edge(b1, b2)

    # Check vertical neighbors
    for y in range(h - 1):
        for x in range(w):
            if (mask1[y, x] and mask2[y + 1, x]) or (mask2[y, x] and mask1[y + 1, x]):
                b1 = tuple(binned[y, x])
                b2 = tuple(binned[y + 1, x])
                if b1 != b2:
                    add_edge(b1, b2)

    return adjacency


def build_transition_chains(binned: np.ndarray, lab: np.ndarray,
                            sub_regions: list[dict], scale: float = 3.0) -> list[list[tuple]]:
    """
    Build chains that span across chromaticity boundaries (warm <-> cool transitions).

    Combines colors from both sub-regions that are adjacent to each other,
    then builds chains through the transition zone.
    """
    if len(sub_regions) < 2:
        return []

    # Get the two main sub-regions (typically cool and warm)
    mask1 = sub_regions[0]['mask']
    mask2 = sub_regions[1]['mask']

    # Find cross-region adjacencies
    cross_adj = find_cross_region_adjacencies(binned, mask1, mask2, max_lab_distance=12.0, scale=scale)

    if not cross_adj:
        return []

    # Get colors that participate in cross-region transitions
    transition_colors = set()
    for (b1, b2) in cross_adj.keys():
        transition_colors.add(b1)
        transition_colors.add(b2)

    # Create combined mask for transition zone (colors near the boundary)
    combined_mask = mask1 | mask2

    # Get coverage for colors in the transition
    coverage = get_bin_coverage_masked(binned, combined_mask)

    # Filter to only colors that participate in transitions
    transition_coverage = {b: c for b, c in coverage.items() if b in transition_colors}

    # Also get within-region adjacencies for these colors
    adj1 = build_adjacency_masked(binned, mask1, max_lab_distance=10.0, scale=scale)
    adj2 = build_adjacency_masked(binned, mask2, max_lab_distance=10.0, scale=scale)

    # Combine all adjacencies, keeping only those involving transition colors
    combined_adj = {}
    for adj in [adj1, adj2, cross_adj]:
        for (b1, b2), count in adj.items():
            if b1 in transition_colors or b2 in transition_colors:
                key = (b1, b2) if b1 < b2 else (b2, b1)
                combined_adj[key] = combined_adj.get(key, 0) + count

    # Build chains through the transition
    chains = build_chains(combined_adj, transition_coverage, scale=scale, max_drift=30.0)

    return chains


def build_adjacency_masked(binned: np.ndarray, mask: np.ndarray,
                           max_lab_distance: float = 8.0, scale: float = 3.0) -> dict:
    """
    Build adjacency graph only for pixels within mask.
    """
    h, w = binned.shape[:2]
    adjacency = {}

    def add_edge(b1, b2):
        lab1 = bin_to_lab(b1, scale)
        lab2 = bin_to_lab(b2, scale)
        dist = np.linalg.norm(lab1 - lab2)

        if dist > max_lab_distance:
            return

        key = (b1, b2) if b1 < b2 else (b2, b1)
        adjacency[key] = adjacency.get(key, 0) + 1

    # Check horizontal neighbors (both pixels must be in mask)
    for y in range(h):
        for x in range(w - 1):
            if mask[y, x] and mask[y, x + 1]:
                b1 = tuple(binned[y, x])
                b2 = tuple(binned[y, x + 1])
                if b1 != b2:
                    add_edge(b1, b2)

    # Check vertical neighbors
    for y in range(h - 1):
        for x in range(w):
            if mask[y, x] and mask[y + 1, x]:
                b1 = tuple(binned[y, x])
                b2 = tuple(binned[y + 1, x])
                if b1 != b2:
                    add_edge(b1, b2)

    return adjacency


def get_bin_coverage_masked(binned: np.ndarray, mask: np.ndarray) -> dict:
    """Get pixel count for each bin, only within mask."""
    masked_bins = binned[mask]
    if len(masked_bins) == 0:
        return {}
    unique, counts = np.unique(masked_bins, axis=0, return_counts=True)
    return {tuple(b): c for b, c in zip(unique, counts)}


def build_chains(adjacency: dict, coverage: dict, scale: float = 3.0,
                  max_drift: float = 25.0, lookback: int = 3) -> list[list[tuple]]:
    """
    Build gradient chains by following connected colors.

    Checks that new colors stay within max_drift of the last `lookback` colors.
    This breaks chains when they drift too far in color space.

    Returns list of chains, where each chain is a list of bin tuples
    ordered by lightness.
    """
    # Build neighbor lookup
    neighbors = {}
    for (b1, b2), count in adjacency.items():
        if b1 not in neighbors:
            neighbors[b1] = []
        if b2 not in neighbors:
            neighbors[b2] = []
        neighbors[b1].append((b2, count))
        neighbors[b2].append((b1, count))

    def lab_distance(b1, b2):
        return np.linalg.norm(bin_to_lab(b1, scale) - bin_to_lab(b2, scale))

    def can_extend_chain(chain, new_bin):
        """Check if new_bin is within max_drift of recent chain members."""
        for old_bin in chain[-lookback:]:
            if lab_distance(old_bin, new_bin) > max_drift:
                return False
        return True

    # Build chains greedily, starting from highest-coverage colors
    all_bins = sorted(coverage.keys(), key=lambda b: coverage[b], reverse=True)
    used = set()
    chains = []

    for start_bin in all_bins:
        if start_bin in used or start_bin not in neighbors:
            continue

        # Start a new chain
        chain = [start_bin]
        used.add(start_bin)

        # Extend in both directions by following neighbors
        changed = True
        while changed:
            changed = False

            # Try to extend from the end
            last = chain[-1]
            for neighbor, _ in sorted(neighbors.get(last, []), key=lambda x: -x[1]):
                if neighbor not in used and can_extend_chain(chain, neighbor):
                    chain.append(neighbor)
                    used.add(neighbor)
                    changed = True
                    break

            # Try to extend from the start
            first = chain[0]
            for neighbor, _ in sorted(neighbors.get(first, []), key=lambda x: -x[1]):
                if neighbor not in used and can_extend_chain(chain[::-1], neighbor):
                    chain.insert(0, neighbor)
                    used.add(neighbor)
                    changed = True
                    break

        if len(chain) >= 2:
            # Keep the chain order as built (follows adjacency)
            # Just ensure it goes dark-to-light for consistency
            L_first = bin_to_lab(chain[0], scale)[0]
            L_last = bin_to_lab(chain[-1], scale)[0]
            if L_first > L_last:
                chain = chain[::-1]
            chains.append(chain)

    # Sort chains by total coverage
    chains.sort(key=lambda c: sum(coverage.get(b, 0) for b in c), reverse=True)
    return chains


def visualize_chains(chains: list[list[tuple]], coverage: dict, total_pixels: int,
                     output_path: str, scale: float = 3.0, max_chains: int = 15):
    """
    Visualize gradient chains as color strips.
    """
    from PIL import ImageDraw

    swatch_size = 30
    padding = 10
    text_width = 100
    row_height = swatch_size + padding

    # Limit chains
    chains = chains[:max_chains]
    max_len = max(len(c) for c in chains) if chains else 1

    img_width = text_width + max_len * swatch_size + padding * 2
    img_height = len(chains) * row_height + padding

    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    for row, chain in enumerate(chains):
        y = padding + row * row_height

        # Calculate chain stats
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        L_min = bin_to_lab(chain[0], scale)[0]
        L_max = bin_to_lab(chain[-1], scale)[0]

        # Draw label
        label = f"{chain_cov:.1f}% L:{L_min:.0f}-{L_max:.0f}"
        draw.text((padding, y + swatch_size // 2 - 5), label, fill=(0, 0, 0))

        # Draw color swatches
        for i, bin_tuple in enumerate(chain):
            x = text_width + i * swatch_size
            lab = bin_to_lab(bin_tuple, scale)
            rgb = lab_to_rgb(lab.reshape(1, -1))[0]
            draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(rgb))

    img.save(output_path)
    print(f"Saved {len(chains)} chains to {output_path}")


def visualize_regions(regions_data: list[dict], output_path: str, scale: float = 3.0):
    """
    Visualize gradient chains grouped by brightness region.

    regions_data: list of dicts with 'name', 'L_range', 'chains', 'coverage', 'total_pixels'
    """
    from PIL import ImageDraw

    swatch_size = 25
    padding = 8
    text_width = 140
    row_height = swatch_size + padding
    section_gap = 15

    # Calculate dimensions
    total_rows = 0
    max_chain_len = 1
    for rd in regions_data:
        chains = rd['chains'][:5]  # Max 5 chains per region
        total_rows += len(chains) + 1  # +1 for header
        for c in chains:
            max_chain_len = max(max_chain_len, len(c))

    img_width = text_width + max_chain_len * swatch_size + padding * 2
    img_height = total_rows * row_height + len(regions_data) * section_gap + padding * 2

    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    y = padding
    for rd in regions_data:
        name = rd['name']
        L_lo, L_hi = rd['L_range']
        chains = rd['chains'][:5]
        coverage = rd['coverage']
        total_pixels = rd['total_pixels']
        region_pixels = sum(coverage.values())
        region_pct = region_pixels / total_pixels * 100

        # Draw region header
        header = f"── {name.upper()} (L:{L_lo:.0f}-{L_hi:.0f}, {region_pct:.1f}%) ──"
        draw.text((padding, y), header, fill=(80, 80, 80))
        y += row_height

        # Draw chains
        for chain in chains:
            chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
            L_min = bin_to_lab(chain[0], scale)[0]
            L_max = bin_to_lab(chain[-1], scale)[0]

            label = f"{chain_cov:.1f}% L:{L_min:.0f}-{L_max:.0f}"
            draw.text((padding + 10, y + swatch_size // 2 - 5), label, fill=(0, 0, 0))

            for i, bin_tuple in enumerate(chain):
                x = text_width + i * swatch_size
                lab = bin_to_lab(bin_tuple, scale)
                rgb = lab_to_rgb(lab.reshape(1, -1))[0]
                draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(rgb))

            y += row_height

        y += section_gap

    img.save(output_path)
    print(f"Saved region visualization to {output_path}")


if __name__ == '__main__':
    source_dir = Path('source_images')
    test_image = source_dir / 'soft_gradients.jpeg'

    print(f"Analyzing: {test_image.name}")
    print("=" * 60)

    # Load and quantize at 3x JND
    lab, (h, w) = load_image(str(test_image))
    binned = quantize(lab, scale=3.0)
    total_pixels = h * w

    print(f"Image: {w}x{h}")

    # Segment by brightness
    regions = segment_by_brightness(lab, num_bands=5)
    print(f"\nBrightness regions: {len(regions)}")

    regions_data = []
    for region in regions:
        mask = region['mask']
        name = region['name']
        L_lo, L_hi = region['L_range']

        print(f"\n── {name.upper()} (L:{L_lo:.0f}-{L_hi:.0f}) ──")

        # Sub-cluster by chromaticity (warm vs cool)
        sub_regions = cluster_by_chromaticity(lab, mask, num_clusters=2)

        for sub in sub_regions:
            sub_mask = sub['mask']
            sub_name = f"{name} {sub['name']}"
            pixel_count = sub_mask.sum()
            pct = pixel_count / total_pixels * 100

            print(f"  {sub['name'].upper()} ({pct:.1f}%)")

            # Get coverage and adjacency for this sub-region
            coverage = get_bin_coverage_masked(binned, sub_mask)
            adj = build_adjacency_masked(binned, sub_mask, max_lab_distance=8.0, scale=3.0)

            print(f"    Colors: {len(coverage)}, Adjacencies: {len(adj)}")

            # Build chains for this sub-region
            chains = build_chains(adj, coverage, scale=3.0)

            # Show top chains
            for i, chain in enumerate(chains[:2]):
                chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
                L_min = bin_to_lab(chain[0], 3.0)[0]
                L_max = bin_to_lab(chain[-1], 3.0)[0]
                print(f"    Chain {i+1}: {len(chain)} colors, {chain_cov:.1f}%, L:{L_min:.0f}-{L_max:.0f}")

            regions_data.append({
                'name': sub_name,
                'L_range': (L_lo, L_hi),
                'chains': chains,
                'coverage': coverage,
                'total_pixels': total_pixels
            })

        # Build transition chains (warm <-> cool)
        if len(sub_regions) >= 2:
            transition_chains = build_transition_chains(binned, lab, sub_regions, scale=3.0)
            if transition_chains:
                print(f"  TRANSITION (warm <-> cool)")
                combined_mask = sub_regions[0]['mask'] | sub_regions[1]['mask']
                transition_coverage = get_bin_coverage_masked(binned, combined_mask)

                for i, chain in enumerate(transition_chains[:2]):
                    chain_cov = sum(transition_coverage.get(b, 0) for b in chain) / total_pixels * 100
                    L_min = bin_to_lab(chain[0], 3.0)[0]
                    L_max = bin_to_lab(chain[-1], 3.0)[0]
                    print(f"    Chain {i+1}: {len(chain)} colors, {chain_cov:.1f}%, L:{L_min:.0f}-{L_max:.0f}")

                regions_data.append({
                    'name': f"{name} transition",
                    'L_range': (L_lo, L_hi),
                    'chains': transition_chains,
                    'coverage': transition_coverage,
                    'total_pixels': total_pixels
                })

    # Output visualization
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{test_image.stem}_regions.png"
    visualize_regions(regions_data, str(output_path), scale=3.0)
