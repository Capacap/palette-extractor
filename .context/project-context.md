# Color Palette Extractor

Tools for extracting and analyzing color relationships in images.

## Goal

Map the color space of an image: understand what colors exist and how they relate to each other spatially and perceptually. Gradients and palettes are views into this underlying structure.

## Approach

### Perceptual Quantization
Bin pixels into JND-sized buckets (2.3 LAB units) to get all perceptually distinct colors.

### Adjacency Graph
Build a graph where colors are nodes and edges represent spatial adjacency. Track directional flow (which color appears to the right/left/above/below of another).

### Salience Metrics
Measure how much each color "stands out":
- **Multi-hop contrast**: LAB distance to colors reachable within 3 hops
- **Global contrast**: Inverse density in LAB space (rare colors score high)
- **Local contrast**: Average LAB distance to spatial neighbors

### Color Relationships
Use adjacency + salience to understand:
- Which colors transition into each other (gradients)
- Which colors are isolated/accent colors
- Spatial structure of color distribution

## Key Design Decisions

- **LAB color space**: Perceptually uniform, separates lightness from chromaticity
- **JND binning**: 2.3 LAB units = Just Noticeable Difference, grounded in perception science
- **Graph-based analysis**: Colors understood through their relationships, not in isolation
- **Unified scoring**: Coverage and salience compete fairly for importance
