#!/usr/bin/env python3
"""Batch analyze images and generate HTML reports."""

import argparse
import sys
import time
from pathlib import Path

from analyze import run_pipeline, render_html


def find_images(directory: Path) -> list[Path]:
    """Find all image files in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []
    for ext in extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description='Batch analyze images and generate HTML reports.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Directory containing images to analyze'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Directory for HTML output files'
    )
    parser.add_argument(
        '--no-downscale',
        action='store_true',
        help='Process at full resolution instead of downscaling to 256px'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(2)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(2)

    total = len(images)
    succeeded = 0
    failed = []
    downscale = not args.no_downscale

    batch_start = time.perf_counter()

    for i, image_path in enumerate(images, 1):
        try:
            img_start = time.perf_counter()
            synthesis, features = run_pipeline(str(image_path), downscale=downscale)
            html = render_html(synthesis, features, str(image_path))
            img_elapsed = time.perf_counter() - img_start

            output_file = output_dir / f"{image_path.stem}-palette.html"
            if output_file.exists():
                print(f"  Warning: Overwriting {output_file.name}", file=sys.stderr)
            output_file.write_text(html)

            print(f"[{i}/{total}] {image_path.name} → {synthesis.scheme_type} ({img_elapsed:.2f}s)")
            succeeded += 1

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[{i}/{total}] {image_path.name} → ERROR: {error_msg}", file=sys.stderr)
            failed.append((image_path.name, error_msg))

    batch_elapsed = time.perf_counter() - batch_start

    # Summary
    print()
    print(f"Completed: {succeeded}/{total} succeeded in {batch_elapsed:.2f}s")
    if succeeded > 0:
        print(f"Average: {batch_elapsed / succeeded:.2f}s per image")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)


if __name__ == '__main__':
    main()
