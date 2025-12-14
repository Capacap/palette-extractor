# Current Focus

## Status: Color Relationship Mapping

Building a system to map color spaces and understand how colors relate spatially and perceptually. Gradients are one view into this structure.

## What We Have

### Salience Metrics
- **Multi-hop contrast**: LAB distance to colors reachable within 3 hops
- **Global contrast**: Inverse density in LAB space (rare colors score high)
- **Local contrast**: Average LAB distance to spatial neighbors

### Directional Flow
Track spatial relationships between adjacent colors:
- `(blue, pink)['right']` = count of times pink is RIGHT of blue
- Asymmetry reveals gradient flow direction

### Unified Scoring
```python
score = coverage * 10 + salience * 0.15
```
Coverage primary, salience as secondary boost.

## Next: Spatial Coherence

Add a coherence metric to distinguish signal from noise:
- **Coherence**: `largest_blob_pixels / total_pixels` (0-1)
- **Blob count**: Number of connected components

High coherence = color forms coherent region(s)
Low coherence = scattered/noise/dither

Use cases:
- Filter low-coherence colors as noise
- Weight relationships by coherence (transitions between coherent colors are more meaningful)

## Files

- `extract_colors.py` - Palette extraction (stable)
- `adjacency_space.py` - **Current focus** - color relationship analysis

## Future Exploration

- **Gradient merging**: Join chain fragments that belong together
- **Diagonal directions**: 8-way instead of 4-way adjacency
- **Multi-scale flow**: Track flow at different neighborhood sizes

## Notes

**Key insight**: Both high coverage and high contrast make a color important. The unified score lets them compete fairly.

**Goal reframe**: Not just gradient extraction, but mapping what colors exist and how they relate. Gradients are a visualization of color transitions, not the end goal.
