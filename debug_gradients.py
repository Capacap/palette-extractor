#!/usr/bin/env python3
"""Debug script to analyze gradient directional consistency."""

import sys
import numpy as np
from pathlib import Path

# Import from analyze.py
from analyze import (
    prepare_data, extract_features,
    JND, FINE_SCALE
)


def compute_local_angles(fine_members: list) -> list:
    """
    Compute local angle at each interior point in the gradient chain.

    At each point (except endpoints), compute the angle between:
    - Incoming direction (from previous point)
    - Outgoing direction (to next point)

    Returns list of angles in degrees.
    """
    if len(fine_members) < 3:
        return []

    fine_size = FINE_SCALE * JND

    # Convert fine bins to LAB coordinates
    labs = [np.array(fb) * fine_size for fb in fine_members]

    angles = []
    for i in range(1, len(labs) - 1):
        # Incoming vector: from previous to current
        v_in = labs[i] - labs[i-1]
        # Outgoing vector: from current to next
        v_out = labs[i+1] - labs[i]

        # Compute angle between vectors
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)

        if norm_in < 1e-6 or norm_out < 1e-6:
            # Zero-length vector (duplicate point)
            angles.append(0.0)
            continue

        # Cosine of angle
        cos_angle = np.dot(v_in, v_out) / (norm_in * norm_out)
        # Clamp to [-1, 1] to handle numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle_deg = np.degrees(np.arccos(cos_angle))
        angles.append(angle_deg)

    return angles


def compute_step_magnitudes(fine_members: list) -> list:
    """Compute LAB distance between consecutive points."""
    if len(fine_members) < 2:
        return []

    fine_size = FINE_SCALE * JND
    labs = [np.array(fb) * fine_size for fb in fine_members]

    magnitudes = []
    for i in range(len(labs) - 1):
        dist = np.linalg.norm(labs[i+1] - labs[i])
        magnitudes.append(dist)

    return magnitudes


def analyze_gradient(grad, idx: int):
    """Analyze a single gradient chain."""
    fine_members = grad.fine_members

    print(f"\n{'='*60}")
    print(f"Gradient {idx}: {grad.direction}, coverage={grad.coverage:.1%}")
    print(f"  Chain length: {len(fine_members)} fine bins, {len(grad.stops)} coarse stops")
    print(f"  L range: {grad.lab_range['L']}")

    # Compute local angles
    angles = compute_local_angles(fine_members)
    if angles:
        print(f"\n  Local angles (degrees):")
        print(f"    Min: {min(angles):.1f}°")
        print(f"    Max: {max(angles):.1f}°")
        print(f"    Mean: {np.mean(angles):.1f}°")
        print(f"    Median: {np.median(angles):.1f}°")
        print(f"    Std: {np.std(angles):.1f}°")

        # Count angles above thresholds
        above_45 = sum(1 for a in angles if a > 45)
        above_90 = sum(1 for a in angles if a > 90)
        above_120 = sum(1 for a in angles if a > 120)
        print(f"\n    Angles > 45°: {above_45}/{len(angles)} ({100*above_45/len(angles):.0f}%)")
        print(f"    Angles > 90°: {above_90}/{len(angles)} ({100*above_90/len(angles):.0f}%)")
        print(f"    Angles > 120°: {above_120}/{len(angles)} ({100*above_120/len(angles):.0f}%)")

        # Show first few angles for intuition
        print(f"\n    First 10 angles: {[f'{a:.0f}°' for a in angles[:10]]}")
        if len(angles) > 10:
            print(f"    Last 10 angles: {[f'{a:.0f}°' for a in angles[-10:]]}")

    # Compute step magnitudes
    mags = compute_step_magnitudes(fine_members)
    if mags:
        print(f"\n  Step magnitudes (LAB distance):")
        print(f"    Min: {min(mags):.1f}")
        print(f"    Max: {max(mags):.1f}")
        print(f"    Mean: {np.mean(mags):.1f}")
        print(f"    Std: {np.std(mags):.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_gradients.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing: {image_path}")
    print("="*60)

    # Load and analyze
    data = prepare_data(str(image_path))
    features = extract_features(data)

    print(f"\nFound {len(features.gradients)} gradients")

    for i, grad in enumerate(features.gradients):
        analyze_gradient(grad, i)

    # Summary comparison
    if features.gradients:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for i, grad in enumerate(features.gradients):
            angles = compute_local_angles(grad.fine_members)
            if angles:
                above_90 = sum(1 for a in angles if a > 90)
                pct_above_90 = 100 * above_90 / len(angles)
                print(f"  G{i}: mean={np.mean(angles):.0f}°, median={np.median(angles):.0f}°, "
                      f">90°={pct_above_90:.0f}%, L-span={grad.lab_range['L'][1]-grad.lab_range['L'][0]:.0f}")


if __name__ == "__main__":
    main()
