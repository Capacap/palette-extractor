#!/usr/bin/env python3
"""
Extract gradients from images.

Approach:
1. Quantize to 3x JND (coarse bins for better connectivity)
2. Build adjacency graph with LAB distance cutoff
3. Find gradient chains through connected colors
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


if __name__ == '__main__':
    source_dir = Path('source_images')
    test_image = source_dir / 'soft_gradients.jpeg'

    print(f"Analyzing: {test_image.name}")
    print("=" * 60)

    # Load and quantize at 3x JND
    lab, (h, w) = load_image(str(test_image))
    binned = quantize(lab, scale=3.0)

    total_pixels = h * w
    coverage = get_bin_coverage(binned)

    print(f"Image: {w}x{h}")
    print(f"Unique colors (3x JND): {len(coverage)}")

    # Build adjacency with distance cutoff
    adj = build_adjacency(binned, max_lab_distance=8.0, scale=3.0)
    print(f"Adjacency pairs (within 8 LAB): {len(adj)}")

    # Show top adjacencies
    print(f"\nTop 20 adjacencies by count:")
    sorted_adj = sorted(adj.items(), key=lambda x: x[1], reverse=True)
    for (b1, b2), count in sorted_adj[:20]:
        lab1 = bin_to_lab(b1, 3.0)
        lab2 = bin_to_lab(b2, 3.0)
        dist = np.linalg.norm(lab1 - lab2)
        cov1 = coverage.get(b1, 0) / total_pixels * 100
        cov2 = coverage.get(b2, 0) / total_pixels * 100
        print(f"  L={lab1[0]:4.0f} <-> L={lab2[0]:4.0f}  dist={dist:5.1f}  count={count:5}  cov={cov1:.1f}%+{cov2:.1f}%")

    # Show colors by lightness
    print(f"\nColors by lightness:")
    by_L = {}
    for bin_tuple, count in coverage.items():
        lab_val = bin_to_lab(bin_tuple, 3.0)
        L_bucket = int(lab_val[0] // 20) * 20
        if L_bucket not in by_L:
            by_L[L_bucket] = []
        by_L[L_bucket].append((bin_tuple, lab_val, count))

    for L_bucket in sorted(by_L.keys()):
        colors = by_L[L_bucket]
        total_cov = sum(c[2] for c in colors) / total_pixels * 100
        print(f"  L {L_bucket:2}-{L_bucket+20}: {len(colors):3} colors, {total_cov:5.1f}% coverage")

    # Build chains
    chains = build_chains(adj, coverage, scale=3.0)
    print(f"\nFound {len(chains)} chains")
    for i, chain in enumerate(chains[:10]):
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        L_min = bin_to_lab(chain[0], 3.0)[0]
        L_max = bin_to_lab(chain[-1], 3.0)[0]
        print(f"  Chain {i+1}: {len(chain)} colors, {chain_cov:.1f}%, L:{L_min:.0f}-{L_max:.0f}")

    # Output visualization
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{test_image.stem}_chains.png"
    visualize_chains(chains, coverage, total_pixels, str(output_path), scale=3.0)
