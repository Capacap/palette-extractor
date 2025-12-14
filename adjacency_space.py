#!/usr/bin/env python3
"""
Gradient extraction via adjacency space embedding.

Approach:
1. Quantize colors to bins
2. Build adjacency graph (which colors appear next to which)
3. Represent each color as a vector of its adjacency relationships
4. Apply dimensionality reduction - gradients should form continuous paths
5. Extract gradient chains by tracing paths in the embedded space
"""

import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from extract_colors import rgb_to_lab, lab_to_rgb, JND


def load_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load image as LAB pixels."""
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img)
    h, w = pixels.shape[:2]
    lab = rgb_to_lab(pixels.reshape(-1, 3)).reshape(h, w, 3)
    return lab, (h, w)


def quantize(lab: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Quantize LAB to JND-sized bins."""
    bin_size = scale * JND
    return np.round(lab / bin_size).astype(np.int32)


def bin_to_lab(bin_tuple: tuple, scale: float = 3.0) -> np.ndarray:
    """Convert bin indices back to LAB values."""
    return np.array(bin_tuple) * scale * JND


def build_adjacency_graph(binned: np.ndarray) -> tuple[dict, dict]:
    """
    Build full adjacency graph without any distance cutoff.

    Returns:
        adjacency: dict mapping (bin1, bin2) -> count
        coverage: dict mapping bin -> pixel count
    """
    h, w = binned.shape[:2]
    adjacency = {}

    def add_edge(b1, b2):
        if b1 == b2:
            return
        key = (b1, b2) if b1 < b2 else (b2, b1)
        adjacency[key] = adjacency.get(key, 0) + 1

    # Horizontal neighbors
    for y in range(h):
        for x in range(w - 1):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y, x + 1])
            add_edge(b1, b2)

    # Vertical neighbors
    for y in range(h - 1):
        for x in range(w):
            b1 = tuple(binned[y, x])
            b2 = tuple(binned[y + 1, x])
            add_edge(b1, b2)

    # Coverage
    flat = binned.reshape(-1, 3)
    unique, counts = np.unique(flat, axis=0, return_counts=True)
    coverage = {tuple(b): c for b, c in zip(unique, counts)}

    return adjacency, coverage


def build_directional_adjacency(binned: np.ndarray) -> dict:
    """
    Build directional adjacency graph - captures which direction colors flow.

    For each ordered pair (color_a, color_b), counts how often color_b appears
    in each direction relative to color_a.

    Returns:
        directional: dict mapping (bin1, bin2) -> {'right': n, 'left': n, 'below': n, 'above': n}

    The key insight: in a gradient flowing left-to-right from blue to pink,
    (blue, pink)['right'] >> (blue, pink)['left']
    This asymmetry reveals the gradient direction.
    """
    h, w = binned.shape[:2]
    directional = {}

    def add_directional(b1, b2, direction):
        """Add count: b2 is in 'direction' relative to b1"""
        if b1 == b2:
            return
        key = (b1, b2)
        if key not in directional:
            directional[key] = {'right': 0, 'left': 0, 'below': 0, 'above': 0}
        directional[key][direction] += 1

    # Horizontal neighbors: pixel at (y, x) and (y, x+1)
    for y in range(h):
        for x in range(w - 1):
            b_left = tuple(binned[y, x])
            b_right = tuple(binned[y, x + 1])
            # b_right is to the RIGHT of b_left
            add_directional(b_left, b_right, 'right')
            # b_left is to the LEFT of b_right
            add_directional(b_right, b_left, 'left')

    # Vertical neighbors: pixel at (y, x) and (y+1, x)
    for y in range(h - 1):
        for x in range(w):
            b_top = tuple(binned[y, x])
            b_bottom = tuple(binned[y + 1, x])
            # b_bottom is BELOW b_top
            add_directional(b_top, b_bottom, 'below')
            # b_top is ABOVE b_bottom
            add_directional(b_bottom, b_top, 'above')

    return directional


def analyze_gradient_flow(directional: dict, colors: list, coverage: dict,
                          scale: float = 3.0, min_flow: int = 50) -> dict:
    """
    Analyze directional adjacency to find gradient flow patterns.

    For each color pair, computes:
    - horizontal_flow: right - left (positive = flows right)
    - vertical_flow: below - above (positive = flows down)

    Returns dict with flow analysis for significant pairs.
    """
    flow_analysis = {}

    for (b1, b2), dirs in directional.items():
        total = dirs['right'] + dirs['left'] + dirs['below'] + dirs['above']
        if total < min_flow:
            continue

        h_flow = dirs['right'] - dirs['left']
        v_flow = dirs['below'] - dirs['above']

        # Compute flow strength (how asymmetric)
        h_total = dirs['right'] + dirs['left']
        v_total = dirs['below'] + dirs['above']

        h_asymmetry = abs(h_flow) / (h_total + 1) if h_total > 0 else 0
        v_asymmetry = abs(v_flow) / (v_total + 1) if v_total > 0 else 0

        flow_analysis[(b1, b2)] = {
            'h_flow': h_flow,
            'v_flow': v_flow,
            'h_asymmetry': h_asymmetry,
            'v_asymmetry': v_asymmetry,
            'total': total,
            'dirs': dirs
        }

    return flow_analysis


def find_flow_gradients(colors: list, directional: dict, coverage: dict,
                        scale: float = 3.0, min_chain_length: int = 5,
                        min_asymmetry: float = 0.3) -> list[dict]:
    """
    Find gradients by following directional flow through the adjacency graph.

    A gradient is a chain where colors consistently flow in one direction:
    - Horizontal gradient: each color has the next color predominantly to its right (or left)
    - Vertical gradient: each color has the next color predominantly below (or above)

    Returns list of gradient dicts with chain, direction, and metadata.
    """
    # Analyze flow patterns
    flow = analyze_gradient_flow(directional, colors, coverage, scale)

    # Build directed graph based on flow
    # Edge from A to B if B is predominantly in one direction from A
    flow_graph = {b: {'right': [], 'left': [], 'below': [], 'above': []} for b in colors}

    for (b1, b2), analysis in flow.items():
        if b1 not in flow_graph or b2 not in flow_graph:
            continue

        # Check horizontal flow
        if analysis['h_asymmetry'] > min_asymmetry:
            if analysis['h_flow'] > 0:
                # b2 is predominantly RIGHT of b1
                flow_graph[b1]['right'].append((b2, analysis['h_flow'], analysis['total']))
            else:
                # b2 is predominantly LEFT of b1
                flow_graph[b1]['left'].append((b2, -analysis['h_flow'], analysis['total']))

        # Check vertical flow
        if analysis['v_asymmetry'] > min_asymmetry:
            if analysis['v_flow'] > 0:
                # b2 is predominantly BELOW b1
                flow_graph[b1]['below'].append((b2, analysis['v_flow'], analysis['total']))
            else:
                # b2 is predominantly ABOVE b1
                flow_graph[b1]['above'].append((b2, -analysis['v_flow'], analysis['total']))

    # Sort neighbors by flow strength
    for b in flow_graph:
        for direction in flow_graph[b]:
            flow_graph[b][direction].sort(key=lambda x: x[1], reverse=True)

    def follow_flow(start, direction):
        """Follow flow in one direction to build a gradient chain."""
        chain = [start]
        current = start
        visited = {start}

        while True:
            candidates = flow_graph[current][direction]
            # Find best unvisited neighbor
            next_color = None
            for neighbor, strength, total in candidates:
                if neighbor not in visited:
                    next_color = neighbor
                    break

            if next_color is None:
                break

            chain.append(next_color)
            visited.add(next_color)
            current = next_color

        return chain

    # Find gradients starting from high-coverage colors
    sorted_colors = sorted(colors, key=lambda b: coverage.get(b, 0), reverse=True)

    all_gradients = []
    used = set()

    for direction in ['right', 'left', 'below', 'above']:
        for start in sorted_colors[:30]:
            if start in used:
                continue

            chain = follow_flow(start, direction)

            if len(chain) >= min_chain_length:
                chain_cov = sum(coverage.get(b, 0) for b in chain)

                # Compute LAB range
                labs = [bin_to_lab(b, scale) for b in chain]
                lab_array = np.array(labs)
                l_range = lab_array[:, 0].max() - lab_array[:, 0].min()
                a_range = lab_array[:, 1].max() - lab_array[:, 1].min()
                b_range = lab_array[:, 2].max() - lab_array[:, 2].min()

                all_gradients.append({
                    'chain': chain,
                    'direction': direction,
                    'coverage': chain_cov,
                    'l_range': l_range,
                    'a_range': a_range,
                    'b_range': b_range,
                    'score': len(chain) * chain_cov
                })

                used.update(chain)

    # Sort by score
    all_gradients.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate
    final_gradients = []
    for grad in all_gradients:
        chain_set = set(grad['chain'])
        is_dup = False
        for existing in final_gradients:
            overlap = len(chain_set & set(existing['chain']))
            if overlap > 0.5 * len(chain_set):
                is_dup = True
                break
        if not is_dup:
            final_gradients.append(grad)

    return final_gradients


def build_adjacency_matrix(adjacency: dict, coverage: dict,
                           min_coverage: float = 0.0001,
                           scale: float = 3.0,
                           lab_weight_sigma: float = 15.0) -> tuple[np.ndarray, list]:
    """
    Build adjacency matrix from graph, weighted by LAB similarity.

    Colors that are both spatially adjacent AND close in LAB space
    get stronger connections (smooth gradients). Adjacent but dissimilar
    colors (hard edges) get weaker connections.

    Args:
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        min_coverage: minimum coverage to include a color
        scale: JND scale factor for bin_to_lab
        lab_weight_sigma: sigma for Gaussian weighting by LAB distance
            - smaller = stronger preference for similar colors
            - larger = more uniform weighting

    Returns:
        matrix: NxN adjacency matrix (normalized by row)
        colors: list of bin tuples in matrix order
    """
    total_pixels = sum(coverage.values())

    # Filter to colors with minimum coverage
    colors = [b for b, c in coverage.items()
              if c / total_pixels >= min_coverage]
    color_to_idx = {b: i for i, b in enumerate(colors)}
    n = len(colors)

    print(f"Building {n}x{n} adjacency matrix (LAB-weighted, sigma={lab_weight_sigma})...")

    # Build matrix with LAB-weighted adjacency
    matrix = np.zeros((n, n), dtype=np.float32)
    for (b1, b2), count in adjacency.items():
        if b1 in color_to_idx and b2 in color_to_idx:
            i, j = color_to_idx[b1], color_to_idx[b2]

            # Calculate LAB distance
            lab1 = bin_to_lab(b1, scale)
            lab2 = bin_to_lab(b2, scale)
            lab_dist = np.linalg.norm(lab1 - lab2)

            # Weight by Gaussian: similar colors get weight ~1, dissimilar get weight ~0
            similarity = np.exp(-lab_dist / lab_weight_sigma)

            # Combined weight: count * similarity
            weight = count * similarity

            matrix[i, j] = weight
            matrix[j, i] = weight

    # Row-normalize (each color's adjacency vector sums to 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums

    return matrix, colors


def embed_adjacency_space(matrix: np.ndarray, method: str = 'pca',
                          n_components: int = 3) -> np.ndarray:
    """
    Embed adjacency vectors into lower-dimensional space.

    Args:
        matrix: NxN adjacency matrix
        method: 'pca' or 'tsne'
        n_components: dimensions to reduce to

    Returns:
        embedding: Nx(n_components) array
    """
    if method == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(matrix)
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    elif method == 'tsne':
        tsne = TSNE(n_components=min(n_components, 3), perplexity=30, random_state=42)
        embedding = tsne.fit_transform(matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding


def find_directional_gradients(embedding: np.ndarray, colors: list, adjacency: dict,
                                coverage: dict, scale: float = 3.0,
                                n_bins: int = 36, min_peak_prominence: float = 0.3,
                                min_colors_per_ray: int = 4) -> list[list[tuple]]:
    """
    Find gradients by detecting directional rays from the centroid.

    In adjacency space, gradients appear as rays emanating from a central cluster.
    Colors at the periphery are gradient endpoints; the center contains colors
    with mixed adjacency patterns.

    Args:
        embedding: Nx3 array of embedded coordinates
        colors: list of bin tuples
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        n_bins: number of angular bins for histogram (36 = 10 degree resolution)
        min_peak_prominence: minimum prominence for peak detection (relative to max)
        min_colors_per_ray: minimum colors to form a valid ray

    Returns:
        List of gradient chains, each ordered from center outward
    """
    n = len(colors)
    color_to_idx = {b: i for i, b in enumerate(colors)}

    # Build neighbor lookup
    neighbors = {b: set() for b in colors}
    for (b1, b2), count in adjacency.items():
        if b1 in neighbors and b2 in neighbors:
            neighbors[b1].add(b2)
            neighbors[b2].add(b1)

    # Step 1: Compute centroid (coverage-weighted)
    total_cov = sum(coverage.get(b, 1) for b in colors)
    weights = np.array([coverage.get(b, 1) / total_cov for b in colors])
    centroid = np.average(embedding, axis=0, weights=weights)

    print(f"Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

    # Step 2: Compute direction and distance from centroid for each color
    vectors = embedding - centroid
    distances = np.linalg.norm(vectors, axis=1)

    # Use PC1-PC2 plane for primary angular analysis
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # -pi to pi

    # Step 3: Build angular histogram to find ray directions
    # Weight by distance (peripheral colors define rays better)
    hist_weights = distances * weights  # distance * coverage

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(angles, bins=bin_edges, weights=hist_weights)

    # Smooth histogram to reduce noise
    hist_smooth = gaussian_filter1d(hist, sigma=1.5, mode='wrap')

    # Normalize for peak detection
    hist_norm = hist_smooth / (hist_smooth.max() + 1e-10)

    # Step 4: Find peaks (ray directions)
    # Handle wrap-around by extending the histogram
    hist_extended = np.concatenate([hist_norm, hist_norm, hist_norm])
    peaks, properties = find_peaks(hist_extended,
                                   prominence=min_peak_prominence,
                                   distance=2)

    # Filter to middle section and convert back to original indices
    valid_peaks = []
    for p in peaks:
        original_idx = p % n_bins
        if original_idx not in valid_peaks:
            valid_peaks.append(original_idx)

    # Convert peak indices to angles
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_angles = [bin_centers[p] for p in valid_peaks]

    print(f"Found {len(peak_angles)} directional rays:")
    for i, angle in enumerate(peak_angles):
        print(f"  Ray {i+1}: {np.degrees(angle):.0f}°")

    # Step 5: Assign colors to nearest ray
    ray_assignments = []
    for i, angle in enumerate(angles):
        # Find nearest peak angle (handling wrap-around)
        best_ray = 0
        best_dist = float('inf')
        for j, peak_angle in enumerate(peak_angles):
            # Angular distance with wrap-around
            diff = abs(angle - peak_angle)
            diff = min(diff, 2*np.pi - diff)
            if diff < best_dist:
                best_dist = diff
                best_ray = j
        ray_assignments.append(best_ray)

    # Step 6: For each ray, order colors by distance from centroid
    rays = []
    for ray_idx in range(len(peak_angles)):
        # Get colors in this ray
        ray_colors = [(colors[i], distances[i], coverage.get(colors[i], 0))
                      for i in range(n) if ray_assignments[i] == ray_idx]

        if len(ray_colors) < min_colors_per_ray:
            continue

        # Sort by distance from centroid (center to periphery)
        ray_colors.sort(key=lambda x: x[1])

        # Extract just the color tuples
        ordered_colors = [c for c, d, cov in ray_colors]

        rays.append({
            'angle': peak_angles[ray_idx],
            'colors': ordered_colors,
            'distances': [d for c, d, cov in ray_colors],
            'coverage': sum(cov for c, d, cov in ray_colors)
        })

    # Step 7: Validate connectivity - build connected chains within each ray
    validated_chains = []

    for ray in rays:
        ordered = ray['colors']

        # Find longest connected subsequence
        # Use greedy: start from center, extend outward following adjacency
        best_chain = []

        for start_idx in range(min(5, len(ordered))):  # Try starting from first few
            chain = [ordered[start_idx]]
            used = {ordered[start_idx]}

            # Extend outward (increasing distance)
            for color in ordered[start_idx + 1:]:
                # Check if connected to any color in chain
                if any(color in neighbors[c] for c in chain[-3:]):  # Check last 3
                    chain.append(color)
                    used.add(color)

            if len(chain) > len(best_chain):
                best_chain = chain

        if len(best_chain) >= min_colors_per_ray:
            validated_chains.append(best_chain)
            ray_cov = sum(coverage.get(b, 0) for b in best_chain)
            print(f"  Ray at {np.degrees(ray['angle']):.0f}°: {len(best_chain)} connected colors, {ray_cov} pixels")

    # Sort by coverage
    validated_chains.sort(key=lambda c: sum(coverage.get(b, 0) for b in c), reverse=True)

    return validated_chains


def find_graph_gradients(colors: list, adjacency: dict, coverage: dict,
                         scale: float = 3.0, min_chain_length: int = 5,
                         max_lab_step: float = 20.0) -> list[list[tuple]]:
    """
    Find gradients by tracing smooth paths through the adjacency graph.

    A gradient is a path where:
    1. Consecutive colors are spatially adjacent (connected in graph)
    2. LAB values change smoothly (small delta per step)
    3. Progression is monotonic in at least one LAB dimension

    Args:
        colors: list of bin tuples
        adjacency: dict of (bin1, bin2) -> count
        coverage: dict of bin -> pixel count
        scale: JND scale for bin_to_lab
        min_chain_length: minimum colors to form a valid gradient
        max_lab_step: maximum LAB distance per step

    Returns:
        List of gradient chains, sorted by quality (length * coverage)
    """
    # Build neighbor lookup with edge weights
    neighbors = {b: {} for b in colors}  # {color: {neighbor: weight}}

    for (b1, b2), count in adjacency.items():
        if b1 not in neighbors or b2 not in neighbors:
            continue

        # Compute LAB distance
        lab1 = bin_to_lab(b1, scale)
        lab2 = bin_to_lab(b2, scale)
        lab_dist = np.linalg.norm(lab1 - lab2)

        # Skip edges with large color jumps (not gradients)
        if lab_dist > max_lab_step:
            continue

        # Weight: prefer high adjacency count and small LAB distance
        # smoothness = count / (lab_dist + 1)
        weight = count

        neighbors[b1][b2] = {'count': count, 'lab_dist': lab_dist, 'weight': weight}
        neighbors[b2][b1] = {'count': count, 'lab_dist': lab_dist, 'weight': weight}

    # Get LAB values for all colors
    lab_values = {b: bin_to_lab(b, scale) for b in colors}

    def find_monotonic_path(start, lab_dim, direction):
        """
        Extend a path in one direction, requiring monotonic change in lab_dim.
        direction: +1 for increasing, -1 for decreasing
        """
        path = [start]
        current = start

        while True:
            current_lab = lab_values[current][lab_dim]

            # Find best neighbor that continues monotonic progression
            best_next = None
            best_score = -1

            for neighbor, edge in neighbors[current].items():
                if neighbor in path:
                    continue

                neighbor_lab = lab_values[neighbor][lab_dim]
                delta = (neighbor_lab - current_lab) * direction

                # Must progress in the right direction (or stay same)
                if delta < -0.5:  # Small tolerance for noise
                    continue

                # Score: prefer larger steps (more gradient) but smooth
                score = edge['count'] * (1 + delta / 10)

                if score > best_score:
                    best_score = score
                    best_next = neighbor

            if best_next is None:
                break

            path.append(best_next)
            current = best_next

        return path

    def extend_bidirectional(start, lab_dim):
        """Extend path in both directions along lab_dim."""
        # Extend in increasing direction
        forward = find_monotonic_path(start, lab_dim, +1)
        # Extend in decreasing direction
        backward = find_monotonic_path(start, lab_dim, -1)

        # Combine: backward (reversed, excluding start) + forward
        if len(backward) > 1:
            full_path = backward[::-1][:-1] + forward
        else:
            full_path = forward

        return full_path

    # Find gradient chains starting from high-coverage colors
    sorted_colors = sorted(colors, key=lambda b: coverage.get(b, 0), reverse=True)

    all_chains = []
    used_colors = set()

    # Try each LAB dimension (L=lightness, a=green-red, b=blue-yellow)
    for lab_dim, dim_name in [(0, 'L'), (1, 'a'), (2, 'b')]:
        for start in sorted_colors[:50]:  # Try top 50 colors as starting points
            if start in used_colors:
                continue

            chain = extend_bidirectional(start, lab_dim)

            if len(chain) >= min_chain_length:
                # Check that chain actually spans a meaningful range
                chain_labs = [lab_values[b][lab_dim] for b in chain]
                lab_range = max(chain_labs) - min(chain_labs)

                if lab_range < 10:  # Skip if not much color change
                    continue

                chain_cov = sum(coverage.get(b, 0) for b in chain)

                all_chains.append({
                    'chain': chain,
                    'dim': dim_name,
                    'range': lab_range,
                    'coverage': chain_cov,
                    'score': len(chain) * chain_cov
                })

                # Mark colors as used (but allow overlap between dimensions)
                if lab_dim == 0:  # Only mark for L dimension to allow a/b to reuse
                    used_colors.update(chain)

    # Sort by score and deduplicate
    all_chains.sort(key=lambda x: x['score'], reverse=True)

    # Remove chains that are subsets of better chains
    final_chains = []
    for chain_info in all_chains:
        chain_set = set(chain_info['chain'])

        # Check if >70% overlap with existing chain
        is_duplicate = False
        for existing in final_chains:
            existing_set = set(existing)
            overlap = len(chain_set & existing_set)
            if overlap > 0.7 * len(chain_set):
                is_duplicate = True
                break

        if not is_duplicate:
            final_chains.append(chain_info['chain'])
            print(f"  {chain_info['dim']}-gradient: {len(chain_info['chain'])} colors, "
                  f"range={chain_info['range']:.0f}, cov={chain_info['coverage']}")

    return final_chains


def visualize_directional_analysis(embedding: np.ndarray, colors: list,
                                    coverage: dict, chains: list[list[tuple]],
                                    output_path: str, scale: float = 3.0):
    """
    Visualize the directional ray analysis with rays highlighted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    total_pixels = sum(coverage.values())

    # Compute centroid
    weights = np.array([coverage.get(b, 1) for b in colors])
    weights = weights / weights.sum()
    centroid = np.average(embedding, axis=0, weights=weights)

    # Get RGB colors and sizes
    rgbs = []
    sizes = []
    for b in colors:
        lab = bin_to_lab(b, scale)
        rgb = lab_to_rgb(lab.reshape(1, -1))[0] / 255.0
        rgbs.append(rgb)
        sizes.append(coverage.get(b, 0) / total_pixels * 5000 + 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Assign colors to chains for highlighting
    color_to_chain = {}
    chain_colors_plt = plt.cm.Set1(np.linspace(0, 1, max(len(chains), 1)))
    for chain_idx, chain in enumerate(chains):
        for b in chain:
            color_to_chain[b] = chain_idx

    for ax_idx, (pc_x, pc_y, title) in enumerate([(0, 1, 'PC1 vs PC2'), (0, 2, 'PC1 vs PC3')]):
        ax = axes[ax_idx]

        # Plot all colors
        ax.scatter(embedding[:, pc_x], embedding[:, pc_y], c=rgbs, s=sizes, alpha=0.5)

        # Mark centroid
        ax.scatter([centroid[pc_x]], [centroid[pc_y]], c='black', s=100, marker='x', linewidths=2)

        # Draw rays for each chain
        for chain_idx, chain in enumerate(chains[:8]):  # Max 8 rays
            if not chain:
                continue

            # Get chain endpoints
            chain_indices = [colors.index(b) for b in chain if b in colors]
            if len(chain_indices) < 2:
                continue

            # Draw arrow from centroid toward chain direction
            chain_center = np.mean(embedding[chain_indices], axis=0)
            direction = chain_center - centroid
            direction = direction / (np.linalg.norm(direction) + 1e-10) * 0.15

            ax.annotate('', xy=(centroid[pc_x] + direction[pc_x], centroid[pc_y] + direction[pc_y]),
                       xytext=(centroid[pc_x], centroid[pc_y]),
                       arrowprops=dict(arrowstyle='->', color=chain_colors_plt[chain_idx], lw=2))

            # Highlight chain colors with edge
            for b in chain:
                if b in colors:
                    idx = colors.index(b)
                    ax.scatter([embedding[idx, pc_x]], [embedding[idx, pc_y]],
                             facecolors='none', edgecolors=chain_colors_plt[chain_idx],
                             s=sizes[idx]*1.5, linewidths=2, alpha=0.8)

        ax.set_xlabel(f'PC{pc_x+1}')
        ax.set_ylabel(f'PC{pc_y+1}')
        ax.set_title(f'Directional Analysis ({title})')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved directional analysis to {output_path}")


def visualize_embedding(embedding: np.ndarray, colors: list, coverage: dict,
                        output_path: str, scale: float = 3.0):
    """
    Visualize colors in their embedded adjacency space.

    Creates a scatter plot where position = embedding coordinates,
    color = actual RGB color, size = coverage.
    """
    import matplotlib.pyplot as plt

    total_pixels = sum(coverage.values())

    # Get RGB colors and sizes
    rgbs = []
    sizes = []
    for b in colors:
        lab = bin_to_lab(b, scale)
        rgb = lab_to_rgb(lab.reshape(1, -1))[0] / 255.0
        rgbs.append(rgb)
        sizes.append(coverage.get(b, 0) / total_pixels * 5000 + 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot PC1 vs PC2
    ax = axes[0]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=rgbs, s=sizes, alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Adjacency Space (PC1 vs PC2)')
    ax.set_aspect('equal')

    # Plot PC1 vs PC3 if available
    if embedding.shape[1] >= 3:
        ax = axes[1]
        ax.scatter(embedding[:, 0], embedding[:, 2], c=rgbs, s=sizes, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC3')
        ax.set_title('Adjacency Space (PC1 vs PC3)')
        ax.set_aspect('equal')
    else:
        ax = axes[1]
        ax.scatter(embedding[:, 0], embedding[:, 1], c=rgbs, s=sizes, alpha=0.7)
        ax.set_title('(Same as left)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved embedding visualization to {output_path}")


def find_gradient_paths(embedding: np.ndarray, colors: list, adjacency: dict,
                        coverage: dict, scale: float = 3.0) -> list[list[tuple]]:
    """
    Find gradient paths by tracing connected colors through embedding space.

    Uses shortest path between PC1 extremes to find the main gradient,
    then greedily extracts remaining chains.
    """
    from collections import deque

    color_to_idx = {b: i for i, b in enumerate(colors)}
    n = len(colors)

    # Build neighbor lookup from adjacency
    neighbors = {b: set() for b in colors}
    for (b1, b2), count in adjacency.items():
        if b1 in neighbors and b2 in neighbors:
            neighbors[b1].add(b2)
            neighbors[b2].add(b1)

    chains = []
    used = set()

    # Find main gradient: shortest path from PC1 min to PC1 max
    pc1_values = embedding[:, 0]
    start_idx = np.argmin(pc1_values)
    end_idx = np.argmax(pc1_values)
    start_color = colors[start_idx]
    end_color = colors[end_idx]

    print(f"Finding path from PC1 min ({pc1_values[start_idx]:.2f}) to max ({pc1_values[end_idx]:.2f})")

    # BFS to find shortest path
    def bfs_path(start, end):
        if start == end:
            return [start]
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            for neighbor in neighbors[current]:
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    main_path = bfs_path(start_color, end_color)
    if main_path:
        print(f"Main gradient path: {len(main_path)} colors")
        chains.append(main_path)
        used.update(main_path)

    # Try to build longest connected path along PC1
    def longest_pc1_path():
        """Find the longest path that generally follows PC1 direction."""
        # Sort all colors by PC1
        sorted_indices = np.argsort(pc1_values)
        sorted_colors = [colors[i] for i in sorted_indices]

        # Try to connect as many as possible in PC1 order
        # Use dynamic programming: for each color, find longest path ending there
        best_path = []

        for i, color in enumerate(sorted_colors):
            # Find the longest path that can reach this color
            best_prev_path = []

            for prev_color in neighbors[color]:
                if prev_color in sorted_colors[:i]:  # Must be earlier in PC1 order
                    # Check what path ends at prev_color
                    for path in [best_path]:  # simplified - just check best so far
                        if path and path[-1] == prev_color:
                            if len(path) > len(best_prev_path):
                                best_prev_path = path

            # Also check if we can extend any existing path
            extended = False
            if best_path and best_path[-1] in neighbors[color]:
                best_path = best_path + [color]
                extended = True

            if not extended:
                # Start fresh or continue
                if len(best_prev_path) > 0:
                    best_path = best_prev_path + [color]
                elif not best_path:
                    best_path = [color]

        return best_path

    # Simpler approach: greedy walk that allows small backwards steps
    def flexible_pc1_path(start, direction=1, max_backtrack=0.02):
        """Follow adjacency graph, allowing small backward steps to stay connected."""
        path = [start]
        current = start
        stuck_count = 0

        while stuck_count < 3:
            current_idx = color_to_idx[current]
            current_pc1 = embedding[current_idx, 0]

            # Find best neighbor - prefer forward progress, allow small backward
            candidates = []
            for neighbor in neighbors[current]:
                if neighbor in path:
                    continue
                neighbor_idx = color_to_idx[neighbor]
                neighbor_pc1 = embedding[neighbor_idx, 0]
                progress = (neighbor_pc1 - current_pc1) * direction
                candidates.append((neighbor, progress, neighbor_pc1))

            if not candidates:
                break

            # Sort by progress, take best that's not too far backward
            candidates.sort(key=lambda x: -x[1])
            best = None
            for cand, progress, _ in candidates:
                if progress >= -max_backtrack:
                    best = cand
                    break

            if best is None:
                stuck_count += 1
                # Take any neighbor to continue
                best = candidates[0][0]
            else:
                stuck_count = 0

            path.append(best)
            current = best

        return path

    # Try flexible greedy path for PC1
    greedy_path = flexible_pc1_path(start_color, direction=1)
    if len(greedy_path) > len(main_path or []):
        print(f"Flexible PC1 path: {len(greedy_path)} colors (vs BFS {len(main_path or [])})")
        chains = [greedy_path]
    else:
        print(f"BFS path kept: {len(main_path or [])} colors")
        if main_path:
            chains = [main_path]

    # Find gradients along other principal components too (chains CAN share colors)
    def find_gradient_along_pc(pc_idx, label):
        """Find gradient along a specific principal component."""
        pc_values = embedding[:, pc_idx]
        start_idx = np.argmin(pc_values)
        end_idx = np.argmax(pc_values)
        start_color = colors[start_idx]

        # Greedy path along this PC
        path = [start_color]
        current = start_color

        while True:
            current_idx = color_to_idx[current]
            current_pc = embedding[current_idx, pc_idx]

            best_neighbor = None
            best_progress = 0

            for neighbor in neighbors[current]:
                if neighbor in path:  # Avoid cycles within this chain only
                    continue
                neighbor_idx = color_to_idx[neighbor]
                neighbor_pc = embedding[neighbor_idx, pc_idx]
                progress = neighbor_pc - current_pc

                if progress > best_progress:
                    best_progress = progress
                    best_neighbor = neighbor

            if best_neighbor is None or best_progress < 0.001:
                break

            path.append(best_neighbor)
            current = best_neighbor

        if len(path) >= 5:
            print(f"{label} path: {len(path)} colors")
            return path
        return None

    # Try PC2 and PC3 for secondary gradients
    if embedding.shape[1] >= 2:
        pc2_path = find_gradient_along_pc(1, "PC2")
        if pc2_path:
            chains.append(pc2_path)

    if embedding.shape[1] >= 3:
        pc3_path = find_gradient_along_pc(2, "PC3")
        if pc3_path:
            chains.append(pc3_path)

    # Find colors with high coverage NOT in main chain
    main_chain_set = set(chains[0]) if chains else set()
    missing_high_cov = [
        (colors[i], coverage.get(colors[i], 0))
        for i in range(len(colors))
        if colors[i] not in main_chain_set
    ]
    missing_high_cov.sort(key=lambda x: -x[1])

    if missing_high_cov:
        print(f"Top colors not in main chain: {len(missing_high_cov)}")
        for color, cov in missing_high_cov[:5]:
            lab_val = bin_to_lab(color, scale)
            print(f"  L={lab_val[0]:.0f} a={lab_val[1]:.0f} b={lab_val[2]:.0f} cov={cov}")

    # Try building chains from high-coverage colors not in main chain
    for start_color, _ in missing_high_cov[:15]:
        # Build chain in BOTH directions from this color
        def extend_chain(start, pc_direction):
            path = [start]
            current = start

            while True:
                current_idx = color_to_idx[current]
                current_pc1 = embedding[current_idx, 0]

                best_neighbor = None
                best_progress = -999

                for neighbor in neighbors[current]:
                    if neighbor in path:
                        continue
                    neighbor_idx = color_to_idx[neighbor]
                    neighbor_pc1 = embedding[neighbor_idx, 0]
                    progress = (neighbor_pc1 - current_pc1) * pc_direction

                    if progress > best_progress:
                        best_progress = progress
                        best_neighbor = neighbor

                if best_neighbor is None:
                    break
                # Allow some backward movement to stay connected
                if best_progress < -0.05:
                    break

                path.append(best_neighbor)
                current = best_neighbor

            return path

        # Extend forward and backward
        forward = extend_chain(start_color, 1)
        backward = extend_chain(start_color, -1)

        # Combine (backward reversed + forward without start)
        full_path = backward[::-1] + forward[1:] if len(forward) > 1 else backward[::-1]

        if len(full_path) >= 5:
            # Check uniqueness
            is_unique = True
            for existing in chains:
                overlap = len(set(full_path) & set(existing))
                if overlap > len(full_path) * 0.5:  # >50% overlap
                    is_unique = False
                    break

            if is_unique:
                chains.append(full_path)

    # Sort chains by coverage (keep first chain in place if it's substantial)
    if len(chains) > 1:
        main_chain = chains[0]
        main_cov = sum(coverage.get(b, 0) for b in main_chain)
        other_chains = sorted(chains[1:],
                              key=lambda c: sum(coverage.get(b, 0) for b in c),
                              reverse=True)

        # Only keep main first if it has good coverage
        if main_cov >= sum(coverage.get(b, 0) for b in other_chains[0]) * 0.5:
            chains = [main_chain] + other_chains
        else:
            chains = sorted(chains, key=lambda c: sum(coverage.get(b, 0) for b in c), reverse=True)

    return chains


def visualize_chains(chains: list[list[tuple]], coverage: dict, total_pixels: int,
                     output_path: str, scale: float = 3.0, max_chains: int = 10):
    """Visualize gradient chains as color strips."""
    from PIL import ImageDraw

    swatch_size = 25
    padding = 10
    text_width = 120
    row_height = swatch_size + padding

    chains = chains[:max_chains]
    max_len = max(len(c) for c in chains) if chains else 1

    img_width = text_width + max_len * swatch_size + padding * 2
    img_height = len(chains) * row_height + padding * 2

    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    for row, chain in enumerate(chains):
        y = padding + row * row_height

        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        label = f"{chain_cov:.1f}% ({len(chain)} colors)"
        draw.text((padding, y + swatch_size // 2 - 5), label, fill=(0, 0, 0))

        for i, bin_tuple in enumerate(chain):
            x = text_width + i * swatch_size
            lab = bin_to_lab(bin_tuple, scale)
            rgb = lab_to_rgb(lab.reshape(1, -1))[0]
            draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(rgb))

    img.save(output_path)
    print(f"Saved {len(chains)} chains to {output_path}")


if __name__ == '__main__':
    source_dir = Path('source_images')
    test_image = source_dir / 'soft_gradients.jpeg'

    print(f"Analyzing: {test_image.name}")
    print("=" * 60)

    # Load and quantize
    lab, (h, w) = load_image(str(test_image))
    binned = quantize(lab, scale=3.0)
    total_pixels = h * w

    print(f"Image: {w}x{h}")

    # Build adjacency graph
    adjacency, coverage = build_adjacency_graph(binned)
    print(f"Unique colors: {len(coverage)}")
    print(f"Adjacency pairs: {len(adjacency)}")

    # Build adjacency matrix
    matrix, colors = build_adjacency_matrix(adjacency, coverage, min_coverage=0.0005,
                                            scale=3.0, lab_weight_sigma=15.0)
    print(f"Colors after filtering: {len(colors)}")

    # Embed in lower dimensions
    embedding = embed_adjacency_space(matrix, method='pca', n_components=3)

    # Visualize embedding
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    visualize_embedding(embedding, colors, coverage,
                        str(output_dir / f"{test_image.stem}_adjacency_space.png"))

    # Find gradient paths using original PC-following method
    print("\n--- PC-Following Method ---")
    chains = find_gradient_paths(embedding, colors, adjacency, coverage)
    print(f"Found {len(chains)} gradient chains")

    for i, chain in enumerate(chains[:5]):
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        print(f"  Chain {i+1}: {len(chain)} colors, {chain_cov:.1f}%")

    visualize_chains(chains, coverage, total_pixels,
                     str(output_dir / f"{test_image.stem}_adjacency_chains.png"))

    # Find gradients using graph-based method (traces actual adjacency paths)
    print("\n--- Graph-Based Method (LAB monotonic) ---")
    graph_chains = find_graph_gradients(
        colors, adjacency, coverage,
        scale=3.0, min_chain_length=5, max_lab_step=25.0
    )
    print(f"Found {len(graph_chains)} graph gradients")

    for i, chain in enumerate(graph_chains[:8]):
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        print(f"  Gradient {i+1}: {len(chain)} colors, {chain_cov:.1f}%")

    visualize_chains(graph_chains, coverage, total_pixels,
                     str(output_dir / f"{test_image.stem}_graph_gradients.png"),
                     max_chains=10)

    # Build directional adjacency and find flow-based gradients
    print("\n--- Directional Flow Method ---")
    directional = build_directional_adjacency(binned)
    print(f"Directional pairs: {len(directional)}")

    flow_gradients = find_flow_gradients(
        colors, directional, coverage,
        scale=3.0, min_chain_length=3, min_asymmetry=0.25
    )
    print(f"Found {len(flow_gradients)} flow gradients")

    for i, grad in enumerate(flow_gradients[:10]):
        chain = grad['chain']
        chain_cov = sum(coverage.get(b, 0) for b in chain) / total_pixels * 100
        print(f"  {grad['direction']:>5} gradient: {len(chain):2d} colors, {chain_cov:5.1f}%, "
              f"L={grad['l_range']:.0f} a={grad['a_range']:.0f} b={grad['b_range']:.0f}")

    # Visualize flow gradients
    flow_chains = [g['chain'] for g in flow_gradients]
    visualize_chains(flow_chains, coverage, total_pixels,
                     str(output_dir / f"{test_image.stem}_flow_gradients.png"),
                     max_chains=10)
