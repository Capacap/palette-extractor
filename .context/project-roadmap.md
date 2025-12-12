# Current Focus

## Status: Experimental - Gradient Detection

Single-file implementation with 3-stage palette pipeline (working):
1. JND perceptual quantization (2.3 LAB units)
2. Chromaticity-based family grouping (ignores lightness)
3. Adaptive shade extraction (shadow/midtone/highlight for large families)

Separate `extract_gradients.py` for gradient experiments (cleaner separation).

## Gradient Detection: Current Approach

Using 3x JND + adjacency distance cutoff + drift checking:

1. **Quantize at 3x JND** - 417 colors (manageable, good connectivity)
2. **Build adjacency with distance cutoff** - only count neighbors within 8 LAB as adjacent (filters hard edges)
3. **Chain building with drift check** - new colors must be within 25 LAB of last 3 chain members (breaks chains when they wander too far in color space)
4. **Greedy extension** - start from highest-coverage colors, extend in both directions

### Results
- 38 chains found (vs 1 giant chain without drift check)
- Main chain: 118 colors, 86.4% coverage
- Separate dark chain: 31 colors, 8.1% coverage
- Gradients follow adjacency order (smoother than L-sorted)

### What works
- Distance cutoff filters hard edges between unrelated colors
- Drift check breaks chains when color changes direction
- Following adjacency order produces smoother gradients than sorting by L

### What needs work
- Gradients still not perfectly smooth
- Large chains absorb too many colors
- Need better way to identify distinct gradient regions

## Next Steps

**Region-based approach**: Identify spatial regions in the image first, then find gradients within each region. This would:
- Separate the blue background from the warm foreground
- Allow each region to have its own gradient
- Prevent chains from jumping between unrelated areas

Possible approaches:
- Use connected components of similar colors
- Spatial clustering (not just color clustering)
- Watershed or other segmentation algorithms

## Files

- `extract_colors.py` - Palette extraction (stable, working)
- `extract_gradients.py` - Gradient experiments (in progress)

## Other Improvements

- Color naming (map LAB to human-readable names)
- JSON/CSS output for design tools
- CLI arguments for thresholds
- Handle more image formats (PNG, WebP)
