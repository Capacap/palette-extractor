#!/usr/bin/env python3
"""Batch analyze images and generate HTML reports."""

import argparse
import sys
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

    for i, image_path in enumerate(images, 1):
        try:
            synthesis, features = run_pipeline(str(image_path))
            html = render_html(synthesis, features, str(image_path))

            output_file = output_dir / f"{image_path.stem}-palette.html"
            if output_file.exists():
                print(f"  Warning: Overwriting {output_file.name}", file=sys.stderr)
            output_file.write_text(html)

            print(f"[{i}/{total}] {image_path.name} → {synthesis.scheme_type}")
            succeeded += 1

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[{i}/{total}] {image_path.name} → ERROR: {error_msg}", file=sys.stderr)
            failed.append((image_path.name, error_msg))

    # Summary
    print()
    print(f"Completed: {succeeded}/{total} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)


if __name__ == '__main__':
    main()
