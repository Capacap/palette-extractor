#!/usr/bin/env python3
"""
Extract quantized colors from an image as LAB vectors with pixel counts.
"""

import numpy as np
from PIL import Image


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB array to RGB (0-255)."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # LAB to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    x = np.where(fx**3 > epsilon, fx**3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(fz**3 > epsilon, fz**3, (116 * fz - 16) / kappa)

    # D65 reference white
    x *= 0.95047
    z *= 1.08883

    # XYZ to linear RGB
    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b_out = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    # Apply gamma correction
    rgb_linear = np.column_stack([r, g, b_out])
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.clip(rgb_linear, 0, None), 1/2.4) - 0.055, 12.92 * rgb_linear)

    # Clip and convert to 0-255
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (0-255) to LAB color space."""
    # Normalize RGB to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0

    # RGB to XYZ (sRGB with D65 illuminant)
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

    # Apply LAB nonlinearity
    epsilon = 0.008856
    kappa = 903.3
    fx = np.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
    fy = np.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
    fz = np.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)

    return np.column_stack([L, a, b_val])


JND = 2.3  # Just Noticeable Difference in LAB units


def extract_colors(image_path: str, jnd_threshold: float = 1.0) -> np.ndarray:
    """
    Quantize image pixels into perceptually distinct colors.

    Args:
        image_path: Path to the input image
        jnd_threshold: Number of JNDs for bin size (1.0 = 2.3 LAB units)

    Returns:
        numpy array of shape (n_colors, 4) where columns are [L, a, b, pixels]
        Sorted by pixel count descending.
    """
    # Load image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img).reshape(-1, 3)

    # Convert RGB to LAB
    pixels_lab = rgb_to_lab(pixels)

    # Bin LAB values into JND-sized buckets
    bin_size = jnd_threshold * JND
    binned = np.round(pixels_lab / bin_size).astype(np.int32)

    # Find unique bins and count pixels
    unique_bins, counts = np.unique(binned, axis=0, return_counts=True)

    # Convert bin indices back to LAB values (bin centers)
    centers = unique_bins.astype(np.float64) * bin_size

    # Build result array: [L, a, b, pixels]
    results = np.column_stack([centers, counts.astype(np.float64)])

    # Sort by pixel count descending
    results = results[results[:, 3].argsort()[::-1]]

    return results


def group_colors(colors: np.ndarray, distance_threshold: float = 15.0, min_coverage: float = 0.01) -> np.ndarray:
    """
    Group similar colors into clusters by chromaticity (a, b), weighted by pixel count.

    Args:
        colors: Array of shape (n, 4) with columns [L, a, b, pixels]
        distance_threshold: Max distance in a,b space to merge colors
        min_coverage: Minimum percentage of total pixels for a cluster (0.01 = 0.01%)

    Returns:
        Array of shape (n_clusters, 4) with columns [L, a, b, pixels]
        Sorted by pixel count descending.
    """
    # Work with a copy, sorted by pixels descending
    colors = colors[colors[:, 3].argsort()[::-1]].copy()
    total_pixels = colors[:, 3].sum()

    clusters = []

    for color in colors:
        lab = color[:3]
        pixels = color[3]

        # Find nearest existing cluster by chromaticity only (a, b)
        merged = False
        for cluster in clusters:
            cluster_lab = cluster[:3] / cluster[3]  # Weighted center
            # Distance in a,b space only (ignore lightness)
            dist = np.linalg.norm(lab[1:3] - cluster_lab[1:3])

            if dist < distance_threshold:
                # Merge: accumulate weighted LAB and pixel count
                cluster[:3] += lab * pixels
                cluster[3] += pixels
                merged = True
                break

        if not merged:
            # New cluster: store weighted LAB sum and pixel count
            clusters.append(np.array([*(lab * pixels), pixels]))

    # Convert to final array with actual LAB values
    results = np.array(clusters)
    results[:, :3] /= results[:, 3:4]  # Divide by pixel count to get center

    # Filter out noise (clusters below minimum coverage threshold)
    min_pixels = total_pixels * (min_coverage / 100)
    results = results[results[:, 3] >= min_pixels]

    # Sort by pixel count descending
    results = results[results[:, 3].argsort()[::-1]]

    return results


def extract_shades(colors: np.ndarray, clusters: np.ndarray, distance_threshold: float = 25.0) -> list[dict]:
    """
    Extract shades (shadow/midtone/highlight) for each color family.

    Args:
        colors: Original quantized colors from extract_colors()
        clusters: Grouped clusters from group_colors()
        distance_threshold: Same threshold used in group_colors()

    Returns:
        List of family dicts with 'lab', 'pixels', and 'shades' (shadow/midtone/highlight)
    """
    families = []

    for cluster in clusters:
        cluster_lab = cluster[:3]
        cluster_pixels = cluster[3]

        # Find all original colors that belong to this cluster
        member_colors = []
        for color in colors:
            dist = np.linalg.norm(color[1:3] - cluster_lab[1:3])
            if dist < distance_threshold:
                member_colors.append(color)

        member_colors = np.array(member_colors)

        # Analyze lightness distribution
        L_values = member_colors[:, 0]
        L_min, L_max = L_values.min(), L_values.max()
        L_range = L_max - L_min

        # Determine number of shades based on coverage and L range
        coverage_pct = cluster_pixels / colors[:, 3].sum() * 100

        # Only split if: >1% coverage AND >30 L range (perceptually distinct shades)
        if coverage_pct > 1.0 and L_range > 30:
            # Split into 3 shades by lightness terciles
            L_terciles = np.percentile(L_values, [33, 67])

            shades = {}
            for shade_name, L_low, L_high in [
                ('shadow', L_min, L_terciles[0]),
                ('midtone', L_terciles[0], L_terciles[1]),
                ('highlight', L_terciles[1], L_max + 1)
            ]:
                mask = (member_colors[:, 0] >= L_low) & (member_colors[:, 0] < L_high)
                if mask.any():
                    shade_colors = member_colors[mask]
                    # Weighted average
                    weights = shade_colors[:, 3]
                    shade_lab = np.average(shade_colors[:, :3], axis=0, weights=weights)
                    shade_pixels = weights.sum()
                    shades[shade_name] = {'lab': shade_lab, 'pixels': shade_pixels}
        else:
            # Single shade - use cluster center
            shades = {'midtone': {'lab': cluster_lab, 'pixels': cluster_pixels}}

        families.append({
            'lab': cluster_lab,
            'pixels': cluster_pixels,
            'shades': shades
        })

    return families


def visualize_clusters(clusters: np.ndarray, output_path: str) -> None:
    """
    Create a swatch image visualizing the color clusters with percentages.

    Args:
        clusters: Array of shape (n, 4) with columns [L, a, b, pixels]
        output_path: Path to save the output image
    """
    from PIL import ImageDraw, ImageFont

    total_pixels = clusters[:, 3].sum()
    swatch_size = 80
    padding = 10
    text_height = 25
    cols = min(len(clusters), 6)
    rows = (len(clusters) + cols - 1) // cols

    img_width = cols * (swatch_size + padding) + padding
    img_height = rows * (swatch_size + text_height + padding) + padding

    # Convert LAB to RGB
    rgb_colors = lab_to_rgb(clusters[:, :3])

    # Create image
    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    for i, (rgb, cluster) in enumerate(zip(rgb_colors, clusters)):
        row = i // cols
        col = i % cols

        x = padding + col * (swatch_size + padding)
        y = padding + row * (swatch_size + text_height + padding)

        # Draw swatch
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(rgb))

        # Draw percentage
        percentage = cluster[3] / total_pixels * 100
        text = f"{percentage:.1f}%"

        # Center text under swatch
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_x = x + (swatch_size - text_width) // 2
        draw.text((text_x, y + swatch_size + 4), text, fill=(0, 0, 0))

    img.save(output_path)
    print(f"Saved visualization to {output_path}")


def visualize_families(families: list[dict], output_path: str) -> None:
    """
    Visualize color families with their shades.
    """
    from PIL import ImageDraw

    swatch_size = 60
    padding = 10
    text_height = 20

    # Calculate dimensions
    max_shades = max(len(f['shades']) for f in families)
    img_width = max_shades * (swatch_size + padding) + padding + 100  # Extra for label
    img_height = len(families) * (swatch_size + text_height + padding) + padding

    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    total_pixels = sum(f['pixels'] for f in families)

    for row, family in enumerate(families):
        y = padding + row * (swatch_size + text_height + padding)
        coverage = family['pixels'] / total_pixels * 100

        # Draw family label
        draw.text((padding, y + swatch_size // 2 - 5), f"{coverage:.1f}%", fill=(0, 0, 0))

        # Draw shades
        shade_order = ['shadow', 'midtone', 'highlight']
        col = 0
        for shade_name in shade_order:
            if shade_name in family['shades']:
                shade = family['shades'][shade_name]
                x = 80 + col * (swatch_size + padding)

                rgb = lab_to_rgb(shade['lab'].reshape(1, -1))[0]
                draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(rgb))

                # Shade label
                label = shade_name[0].upper()  # S, M, H
                draw.text((x + swatch_size // 2 - 4, y + swatch_size + 2), label, fill=(100, 100, 100))
                col += 1

    img.save(output_path)
    print(f"Saved visualization to {output_path}")


def process_image(image_path: str, output_path: str) -> list[dict]:
    """Process a single image and save visualization."""
    # Stage 1: Perceptual quantization
    colors = extract_colors(image_path)

    # Stage 2: Group into clusters
    clusters = group_colors(colors, distance_threshold=25.0)

    # Stage 3: Extract shades per family
    families = extract_shades(colors, clusters, distance_threshold=25.0)

    # Visualize
    visualize_families(families, output_path)

    return families


if __name__ == '__main__':
    import os
    from pathlib import Path

    source_dir = Path('source_images')
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    for image_path in sorted(source_dir.glob('*.jpeg')):
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print('='*60)

        output_path = output_dir / f"{image_path.stem}_palette.png"
        families = process_image(str(image_path), str(output_path))

        for i, f in enumerate(families):
            shade_names = list(f['shades'].keys())
            coverage = f['pixels'] / sum(fam['pixels'] for fam in families) * 100
            print(f"  Family {i+1}: {coverage:.1f}% - {len(shade_names)} shades ({', '.join(shade_names)})")

    print(f"\nDone! Results saved to {output_dir}/")
