# Current Focus

## Status: Unified Gradient Detection with Significance

Integrated significance metrics into gradient detection. Starting points are now selected by unified score combining coverage and significance, allowing dark/contrasting colors to emerge naturally.

## What We Built

### Significance Metrics
- **Multi-hop contrast**: LAB distance to colors reachable within 3 hops in adjacency graph
- **Global contrast**: Inverse density in LAB space (rare colors score high)
- **Local contrast**: Average LAB distance to spatial neighbors

### Unified Starting Score
Instead of separate phases for coverage-based and significance-based gradients:
```python
score = coverage * 10 + significance * 0.15
```
- Coverage remains primary (high-coverage colors rank first)
- Significance provides secondary boost for dark/contrasting colors
- Dark accent colors emerge naturally without separate phase

### Directional Flow Detection
For each color pair, track spatial direction:
- `(blue, pink)['right']` = count of times pink is RIGHT of blue
- Asymmetry reveals gradient flow direction

### Results on soft_gradients.jpeg
- **Main gradient** (31.2%, 29 colors): warm→cool face lighting
- **Dark gradients captured naturally**: L=13.8, L=20.7 starting points
- No separate "accent" phase needed

## Methods Comparison

| Method | L gradients | Hue gradients | Dark accents | Spatial |
|--------|-------------|---------------|--------------|---------|
| PC-following | Yes | Weak | No | None |
| Graph-based (LAB monotonic) | Yes | No | Yes (merged) | Adjacency |
| **Directional flow + significance** | Yes | **Yes** | **Yes (emergent)** | **Direction-aware** |

## Files

- `extract_colors.py` - Palette extraction (stable)
- `extract_gradients.py` - Region-based approach (archived)
- `adjacency_space.py` - **Current focus** - unified gradient detection

## Key Functions

- `compute_multihop_contrast()` - 3-hop contrast metric
- `compute_global_contrast()` - LAB space density inverse
- `find_flow_gradients()` - Unified scoring, directional flow following

## Next Steps to Explore

### Gradient Merging
- Some gradients may be fragments of the same visual gradient
- Merge chains that share endpoints and have compatible directions/colors

### Further Enrichment
- **Diagonal directions**: Add 4 diagonal adjacency directions
- **Multi-scale flow**: Track flow at different neighborhood sizes

## Notes

**Key insight**: Both high coverage and high contrast make a color important. The unified score lets them compete fairly for gradient seeds.

**Breakthrough**: Dark accent colors (eyelashes, horns) now emerge naturally through the same mechanism that finds main gradients - no bolt-on required.
