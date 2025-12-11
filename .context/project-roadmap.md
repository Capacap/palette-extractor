# Current Focus

## Status: Working v0.2

Single-file implementation with 3-stage pipeline:
1. JND perceptual quantization (2.3 LAB units)
2. Chromaticity-based family grouping (ignores lightness)
3. Adaptive shade extraction (shadow/midtone/highlight for large families)

Batch processes all images in source_images/ to output/.

## Next: Spatial Reasoning

Explore encoding pixel coordinates to understand spatial relationships:
- Find colors that are frequently adjacent (likely form gradients or edges)
- Detect color regions vs scattered pixels
- Identify accent colors by isolation (small but spatially concentrated)
- Could help distinguish intentional palette choices from noise/artifacts

Potential approaches:
- Run-length encoding to find contiguous color regions
- Neighbor analysis: for each color, what colors surround it?
- Spatial clustering: are pixels of this color concentrated or dispersed?

## Other Improvements

- Color naming (map LAB to human-readable names)
- JSON/CSS output for design tools
- CLI arguments for thresholds
- Handle more image formats (PNG, WebP)
