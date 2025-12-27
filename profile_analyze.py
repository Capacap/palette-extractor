#!/usr/bin/env python3
"""Profile analyze.py to identify performance bottlenecks."""

import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

# Import the pipeline
from analyze import prepare_data, extract_features, synthesize, render

def profile_image(image_path: str, verbose: bool = True):
    """Profile a single image through the full pipeline."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Profiling: {Path(image_path).name}")
        print(f"{'='*60}")

    # Stage timings
    timings = {}

    # Stage 1: Data Preparation
    start = time.perf_counter()
    data = prepare_data(image_path)
    timings['prepare_data'] = time.perf_counter() - start

    if verbose:
        print(f"  Fine bins: {len(data.fine_bins):,}")
        print(f"  Coarse bins: {len(data.coarse_bins):,}")
        print(f"  Total pixels: {data.total_pixels:,}")

    # Stage 2: Feature Extraction
    start = time.perf_counter()
    features = extract_features(data)
    timings['extract_features'] = time.perf_counter() - start

    if verbose:
        print(f"  Gradients detected: {len(features.gradients)}")
        print(f"  Color families: {len(features.families)}")

    # Stage 3: Synthesis
    start = time.perf_counter()
    result = synthesize(data, features)
    timings['synthesize'] = time.perf_counter() - start

    # Stage 4: Render
    start = time.perf_counter()
    output = render(result, features)
    timings['render'] = time.perf_counter() - start

    total = sum(timings.values())
    timings['total'] = total

    if verbose:
        print(f"\nStage timings:")
        for stage, t in timings.items():
            pct = (t / total * 100) if stage != 'total' else 100
            print(f"  {stage:20s}: {t:6.3f}s ({pct:5.1f}%)")

    return timings, data

def detailed_profile(image_path: str):
    """Run detailed cProfile on extract_features (the main compute stage)."""

    print(f"\n{'='*60}")
    print(f"Detailed profile of extract_features()")
    print(f"{'='*60}")

    # Prepare data first (outside profiling)
    data = prepare_data(image_path)

    # Profile extract_features
    profiler = cProfile.Profile()
    profiler.enable()
    features = extract_features(data)
    profiler.disable()

    # Format output
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions

    print(stream.getvalue())

    return features

def main():
    images_dir = Path(__file__).parent / "source_images"
    images = list(images_dir.glob("*.jpeg"))

    if not images:
        print("No images found in source_images/")
        sys.exit(1)

    print(f"Found {len(images)} test images")

    # Quick timing for all images
    all_timings = []
    for img in images:
        timings, data = profile_image(str(img))
        all_timings.append((img.name, timings, len(data.fine_bins), len(data.coarse_bins)))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Image':<35} {'Fine':>8} {'Coarse':>8} {'Total':>8}")
    print("-" * 60)
    for name, timings, fine, coarse in all_timings:
        print(f"{name:<35} {fine:>8,} {coarse:>8,} {timings['total']:>7.3f}s")

    # Detailed profile on first image
    if images:
        detailed_profile(str(images[0]))

if __name__ == "__main__":
    main()
