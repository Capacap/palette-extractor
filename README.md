# image-palette-extractor

Extract color palettes from images. Produces prose output suitable for LLM consumption, with optional HTML reports for human review.

## How it works

Unlike simple color extraction that clusters pixels by RGB similarity, image-palette-extractor analyzes how colors relate to each other spatially:

- **Adjacency graph** — Colors are understood through their neighbors, not in isolation. A yellow next to black has different significance than yellow next to orange.
- **LAB color space** — All perceptual calculations use LAB, where distances match human perception. Colors 2.3 units apart are one "just noticeable difference."
- **Role assignment** — Colors get roles (dominant, accent, light, dark) based on coverage, contrast, and spatial coherence — not just frequency.
- **Salience over coverage** — A rare high-contrast accent can outrank a common background color in visual importance.

## Installation

Requires Python 3.10+.

```bash
# With uv (recommended)
uvx image-palette-extractor -i image.png

# Or install globally
uv tool install image-palette-extractor
image-palette-extractor -i image.png

# Or with pip
pip install image-palette-extractor
```

## Usage

```bash
image-palette-extractor -i <image_path> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-i, --input` | Path to image file (required) |
| `-o, --output` | Write HTML report. Optionally specify path, otherwise auto-names from input. |
| `--no-downscale` | Process at full resolution instead of downscaling to 256px |

## Output

Text output includes:

- **Overview** — Scheme classification (monochromatic, analogous, complementary, etc.), lightness/chroma ranges, distribution
- **Colors** — Notable colors ranked by prominence with roles: Dominant, Secondary, Accent, Light, Dark
- **Relationships** — WCAG-rated contrast pairs and harmonic color pairings

Each color shows hex, RGB, LAB values, coverage percentage, and perceptual notes.

## Examples

```bash
# Analyze and print to terminal
image-palette-extractor -i photo.jpg

# Generate HTML report (auto-named photo-palette.html)
image-palette-extractor -i photo.jpg -o

# Generate HTML report with custom path
image-palette-extractor -i photo.jpg -o report.html

# Full resolution analysis (slower, more precise)
image-palette-extractor -i photo.jpg --no-downscale
```
