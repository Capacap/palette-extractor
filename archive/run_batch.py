#!/usr/bin/env python3
"""
Batch runner for adjacency_space color analysis.

Runs the full analysis pipeline on all images in source_images/,
saving results to output/<timestamp>/ for comparison across runs.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from adjacency_space import (
    load_image, quantize, bin_to_lab,
    build_adjacency_graph, build_directional_adjacency,
    build_adjacency_matrix, embed_adjacency_space,
    compute_color_metrics, compute_multihop_contrast,
    compute_spatial_coherence, identify_noise_colors,
    compute_pixel_local_contrast,
    find_gradient_paths, find_graph_gradients, find_flow_gradients,
    visualize_embedding, visualize_chains, visualize_color_metrics,
)
from extract_colors import lab_to_rgb


def analyze_image(image_path: Path, output_dir: Path, scale: float = 3.0) -> dict:
    """
    Run full adjacency space analysis on a single image.

    Args:
        image_path: Path to image file
        output_dir: Directory to save outputs
        scale: JND scale factor (default 3.0)

    Returns:
        dict with analysis summary
    """
    stem = image_path.stem
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {image_path.name}")
    print('=' * 60)

    # Load and quantize
    lab, (h, w) = load_image(str(image_path))
    binned = quantize(lab, scale=scale)
    total_pixels = h * w

    print(f"Image: {w}x{h} ({total_pixels:,} pixels)")

    # Build adjacency graph
    adjacency, coverage = build_adjacency_graph(binned)
    print(f"Unique colors: {len(coverage)}")
    print(f"Adjacency pairs: {len(adjacency)}")

    # Build adjacency matrix
    matrix, colors = build_adjacency_matrix(
        adjacency, coverage, min_coverage=0.0005,
        scale=scale, lab_weight_sigma=15.0
    )
    print(f"Colors after filtering: {len(colors)}")

    # Embed in lower dimensions
    embedding = embed_adjacency_space(matrix, method='pca', n_components=3)

    # Visualize embedding
    visualize_embedding(
        embedding, colors, coverage,
        str(output_dir / f"{stem}_adjacency_space.png"),
        scale=scale
    )

    # Find gradient paths using PC-following method
    print("\n--- PC-Following Method ---")
    chains = find_gradient_paths(embedding, colors, adjacency, coverage, scale=scale)
    print(f"Found {len(chains)} gradient chains")

    visualize_chains(
        chains, coverage, total_pixels,
        str(output_dir / f"{stem}_adjacency_chains.png"),
        scale=scale
    )

    # Find gradients using graph-based method
    print("\n--- Graph-Based Method (LAB monotonic) ---")
    graph_chains = find_graph_gradients(
        colors, adjacency, coverage,
        scale=scale, min_chain_length=5, max_lab_step=25.0
    )
    print(f"Found {len(graph_chains)} graph gradients")

    visualize_chains(
        graph_chains, coverage, total_pixels,
        str(output_dir / f"{stem}_graph_gradients.png"),
        scale=scale, max_chains=10
    )

    # Build directional adjacency
    print("\n--- Building Directional Adjacency ---")
    directional = build_directional_adjacency(binned)
    print(f"Directional pairs: {len(directional)}")

    # Compute color metrics
    print("\n--- Computing Color Metrics ---")
    color_metrics = compute_color_metrics(colors, adjacency, coverage, scale=scale)

    # Compute multi-hop contrast
    print("Computing multi-hop adjacency contrast...")
    multihop = compute_multihop_contrast(colors, adjacency, coverage, scale=scale, max_hops=3)
    for b in colors:
        color_metrics[b]['multihop_contrast'] = multihop.get(b, 0)

    # Compute spatial coherence
    print("Computing spatial coherence...")
    coherence = compute_spatial_coherence(binned, colors, coverage)
    for b in colors:
        coh = coherence.get(b, {'coherence': 0, 'blob_count': 0, 'largest_blob': 0})
        color_metrics[b]['coherence'] = coh['coherence']
        color_metrics[b]['blob_count'] = coh['blob_count']
        color_metrics[b]['largest_blob'] = coh['largest_blob']

    # Identify noise colors
    noise_colors, median_coherence = identify_noise_colors(coherence, coverage, threshold=0.3)
    print(f"Median coherence: {median_coherence:.2f}, noise colors: {len(noise_colors)}")
    for b in colors:
        color_metrics[b]['is_noise'] = b in noise_colors

    # Find flow-based gradients
    print("\n--- Directional Flow Method ---")
    flow_gradients = find_flow_gradients(
        colors, directional, coverage,
        scale=scale, min_chain_length=3, min_asymmetry=0.25,
        color_metrics=color_metrics
    )
    print(f"Found {len(flow_gradients)} flow gradients")

    # Print top gradients
    for i, grad in enumerate(flow_gradients[:8]):
        chain = grad['chain']
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        print(f"  {grad['direction']:>10}: {len(chain):2d} colors, {chain_cov:5.1f}%, "
              f"L={grad['l_range']:.0f} a={grad['a_range']:.0f} b={grad['b_range']:.0f}")

    # Visualize flow gradients
    flow_chains = [g['chain'] for g in flow_gradients]
    visualize_chains(
        flow_chains, coverage, total_pixels,
        str(output_dir / f"{stem}_flow_gradients.png"),
        scale=scale, max_chains=12, gradients=flow_gradients
    )

    # Compute pixel-level contrast
    print("\nComputing pixel-level local contrast...")
    pixel_contrast = compute_pixel_local_contrast(lab, binned, colors, radius=5)
    for b in colors:
        color_metrics[b]['pixel_contrast'] = pixel_contrast.get(b, 0)

    # Visualize color metrics
    visualize_color_metrics(
        color_metrics,
        str(output_dir / f"{stem}_metrics.png"),
        scale=scale, top_n=12
    )

    # Compute summary statistics
    coverages = [s['coverage'] for s in color_metrics.values()]
    coherences = [s['coherence'] for s in color_metrics.values()]

    summary = {
        'image': image_path.name,
        'width': w,
        'height': h,
        'total_pixels': total_pixels,
        'unique_colors': len(coverage),
        'filtered_colors': len(colors),
        'adjacency_pairs': len(adjacency),
        'pc_chains': len(chains),
        'graph_gradients': len(graph_chains),
        'flow_gradients': len(flow_gradients),
        'noise_colors': len(noise_colors),
        'median_coherence': median_coherence,
        'top_gradient_coverage': (
            sum(coverage.get(b, 0) for b in flow_gradients[0]['chain']) / total_pixels
            if flow_gradients else 0
        ),
    }

    return summary


def run_batch(source_dir: Path, output_base: Path, scale: float = 3.0) -> list[dict]:
    """
    Run analysis on all images in source directory.

    Args:
        source_dir: Directory containing images
        output_base: Base output directory (timestamp subdir will be created)
        scale: JND scale factor

    Returns:
        list of summary dicts for each image
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [
        p for p in sorted(source_dir.iterdir())
        if p.suffix.lower() in image_extensions
    ]

    if not images:
        print(f"No images found in {source_dir}")
        return []

    print(f"Found {len(images)} images to process")

    # Process each image
    summaries = []
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing {image_path.name}")
        try:
            summary = analyze_image(image_path, output_dir, scale=scale)
            summaries.append(summary)
        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")
            summaries.append({
                'image': image_path.name,
                'error': str(e)
            })

    # Write summary report
    report_path = output_dir / "summary.txt"
    with open(report_path, 'w') as f:
        f.write(f"Batch Analysis Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Scale: {scale}\n")
        f.write(f"Images: {len(images)}\n")
        f.write("=" * 60 + "\n\n")

        for s in summaries:
            if 'error' in s:
                f.write(f"{s['image']}: ERROR - {s['error']}\n\n")
                continue

            f.write(f"{s['image']}\n")
            f.write(f"  Size: {s['width']}x{s['height']} ({s['total_pixels']:,} px)\n")
            f.write(f"  Colors: {s['unique_colors']} unique, {s['filtered_colors']} after filter\n")
            f.write(f"  Gradients: {s['flow_gradients']} flow, {s['graph_gradients']} graph, {s['pc_chains']} PC\n")
            f.write(f"  Coherence: {s['median_coherence']:.2f} median, {s['noise_colors']} noise colors\n")
            f.write(f"  Top gradient coverage: {s['top_gradient_coverage']:.1%}\n")
            f.write("\n")

    print(f"\n{'=' * 60}")
    print(f"Batch complete! Results saved to: {output_dir}")
    print(f"Summary report: {report_path}")

    return summaries


if __name__ == '__main__':
    source_dir = Path('source_images')
    output_base = Path('output')

    # Allow scale override via command line
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

    summaries = run_batch(source_dir, output_base, scale=scale)

    # Print final summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Image':<30} {'Size':>12} {'Colors':>8} {'Gradients':>10}")
    print("-" * 60)

    for s in summaries:
        if 'error' in s:
            print(f"{s['image']:<30} ERROR: {s['error'][:30]}")
        else:
            size = f"{s['width']}x{s['height']}"
            print(f"{s['image']:<30} {size:>12} {s['filtered_colors']:>8} {s['flow_gradients']:>10}")
