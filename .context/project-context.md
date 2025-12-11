# Color Palette Extractor

Single-file Python script to extract color palettes from images.

## Approach

### Stage 1: Perceptual Quantization
Bin pixels into JND-sized buckets (2.3 LAB units) to get all perceptually distinct colors.

### Stage 2: Chromaticity Grouping
Group colors by chromaticity (a, b) ignoring lightness. This merges different brightness levels of the same hue into families. Uses greedy clustering from largest to smallest.

### Stage 3: Shade Extraction
For families with enough coverage (>1%) and lightness range (>30 L units), extract shadow/midtone/highlight shades. Smaller families get a single midtone.

## Key Design Decisions

- **LAB color space**: Perceptually uniform, separates lightness from chromaticity
- **JND binning**: 2.3 LAB units = Just Noticeable Difference, grounded in perception science
- **Chromaticity-only grouping**: Different brightnesses of same hue belong together
- **Coverage-based filtering**: 0.01% minimum filters noise, scales with image size
- **Adaptive shading**: Only split families that have enough data to warrant it
