# Current Focus

## Status: Directional Flow Gradients

Enriched adjacency with directional information. The `find_flow_gradients()` method now captures **hue gradients** (not just lightness) by tracking spatial flow direction.

## What We Built

### Directional Adjacency (New)
For each color pair, track not just "they're adjacent" but "in which direction":
- `(blue, pink)['right']` = count of times pink is RIGHT of blue
- `(blue, pink)['left']` = count of times pink is LEFT of blue
- Same for `above`/`below`

**Asymmetry reveals gradient flow**: If `right >> left`, the gradient flows left-to-right.

### Flow Gradient Detection
1. Build directional adjacency from image
2. Compute flow asymmetry for each color pair
3. Build directed graph: edge A→B if B is predominantly in one direction from A
4. Follow flow chains to extract gradients
5. Report gradient direction (right/left/above/below) and LAB ranges

### Results on soft_gradients.jpeg
- **Main gradient** (31.2%, 29 colors): warm→cool face lighting, `a=34, b=76` range
- **Dark gradients** captured: blues at L=14, L=21 (horns, shadows)
- Found gradients in multiple directions (above, below, right, left)
- Hue transitions now detected, not just lightness

### Key Parameters
- `min_chain_length=3` (captures short dark region gradients)
- `min_asymmetry=0.25` (detects weaker directional flow)

## Methods Comparison

| Method | Finds L gradients | Finds hue gradients | Spatial structure |
|--------|-------------------|---------------------|-------------------|
| PC-following | Yes | Weak | None |
| Graph-based (LAB monotonic) | Yes | No | Adjacency only |
| **Directional flow** | Yes | **Yes** | **Direction-aware** |

## Files

- `extract_colors.py` - Palette extraction (stable)
- `extract_gradients.py` - Region-based approach (archived)
- `adjacency_space.py` - **Current focus** - multiple gradient detection methods

## Next Steps to Explore

### Contrast-Based Significance Weighting
**Key insight**: Colors that contrast from their surroundings are visually significant even with small coverage. The dark horns/eyelashes stand out precisely because they contrast strongly.

Ideas:
- **Local contrast score**: For each color, measure average LAB distance to its neighbors
- **Significance = coverage × contrast**: Small but high-contrast features get boosted
- **Edge detection proxy**: High-contrast colors often define object boundaries
- Could use this to weight gradient importance or ensure contrasting colors are included

### Further Enrichment
- **Diagonal directions**: Add 4 diagonal adjacency directions for more precision
- **Multi-scale flow**: Track flow at different neighborhood sizes
- **Combine flow with LAB constraints**: Follow flow AND require smooth LAB progression

### Gradient Merging
- Some gradients may be fragments of the same visual gradient
- Merge chains that share endpoints and have compatible directions/colors

## Notes

**Breakthrough**: Directional adjacency captures gradient flow that pure adjacency misses. A warm-to-cool lighting gradient appears as consistent upward flow (warm below, cool above).

**Remaining gap**: Very dark/high-contrast features need explicit handling. Current method finds them with relaxed parameters, but a significance weighting approach could make this more principled.
