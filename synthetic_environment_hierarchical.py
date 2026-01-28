#!/usr/bin/env python3
"""
Phase B: Hierarchical Composition

Generates scenes with nested spatial relations:
- Level 1: "red circle above blue square"
- Level 2: "red circle above (blue square next_to green triangle)"
- Level 3: "(red circle left_of yellow square) above (blue square next_to green triangle)"

Tests if Distributed ATL can learn hierarchical groupings.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from synthetic_environment import (
    COLORS, SHAPES, create_stimulus_on_canvas, get_spatial_relation
)


@dataclass
class HierarchicalPair:
    """A scene with hierarchical composition."""
    image: np.ndarray
    label: str
    flat_label: str  # Without hierarchy markers
    hierarchy_depth: int
    objects: List[Dict]
    structure: Dict  # Tree structure

    def to_tensor(self) -> Tuple[torch.Tensor, str]:
        return torch.from_numpy(self.image).float(), self.label


def _place_object(canvas: np.ndarray, obj: Dict, canvas_size: int) -> np.ndarray:
    """Place a single object on canvas."""
    obj_img = create_stimulus_on_canvas(
        shape=obj['shape'],
        color=obj['color'],
        size=obj.get('size', 'small'),
        canvas_size=canvas_size,
        center=obj['center'],
        noise=0.0,
    )
    return np.clip(canvas + obj_img, 0.0, 1.0)


def create_hierarchical_scene_depth1(
    colors: List[str],
    shapes: List[str],
    canvas_size: int = 56,
) -> HierarchicalPair:
    """
    Depth 1: Simple relation (same as before)
    "red circle above blue square"
    """
    c1, c2 = np.random.choice(colors, 2, replace=False)
    s1, s2 = np.random.choice(shapes, 2, replace=True)
    
    relations = ['above', 'below', 'left_of', 'right_of']
    rel = np.random.choice(relations)
    
    margin = int(canvas_size * 0.25)
    centers = {
        'above': ((canvas_size//2, margin), (canvas_size//2, canvas_size-margin)),
        'below': ((canvas_size//2, canvas_size-margin), (canvas_size//2, margin)),
        'left_of': ((margin, canvas_size//2), (canvas_size-margin, canvas_size//2)),
        'right_of': ((canvas_size-margin, canvas_size//2), (margin, canvas_size//2)),
    }
    
    c1_pos, c2_pos = centers[rel]
    jitter = int(canvas_size * 0.05)
    c1_pos = (c1_pos[0] + np.random.randint(-jitter, jitter+1),
              c1_pos[1] + np.random.randint(-jitter, jitter+1))
    c2_pos = (c2_pos[0] + np.random.randint(-jitter, jitter+1),
              c2_pos[1] + np.random.randint(-jitter, jitter+1))
    
    obj1 = {'color': c1, 'shape': s1, 'center': c1_pos}
    obj2 = {'color': c2, 'shape': s2, 'center': c2_pos}
    
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    canvas = _place_object(canvas, obj1, canvas_size)
    canvas = _place_object(canvas, obj2, canvas_size)
    
    label = f"{c1} {s1} {rel} {c2} {s2}"
    
    return HierarchicalPair(
        image=canvas,
        label=label,
        flat_label=label,
        hierarchy_depth=1,
        objects=[obj1, obj2],
        structure={'rel': rel, 'left': obj1, 'right': obj2},
    )


def create_hierarchical_scene_depth2(
    colors: List[str],
    shapes: List[str],
    canvas_size: int = 56,
) -> HierarchicalPair:
    """
    Depth 2: One object relates to a GROUP
    "red circle above (blue square next_to green triangle)"
    
    Layout:
    - obj1 at top
    - (obj2, obj3) as group at bottom, horizontally arranged
    """
    c1, c2, c3 = np.random.choice(colors, 3, replace=True)
    # Ensure at least 2 different colors
    while c1 == c2 == c3:
        c3 = np.random.choice(colors)
    
    s1, s2, s3 = np.random.choice(shapes, 3, replace=True)
    
    # Main relation: obj1 vs (group)
    main_rels = ['above', 'below']
    main_rel = np.random.choice(main_rels)
    
    # Group relation: obj2 vs obj3
    group_rels = ['left_of', 'right_of', 'next_to']
    group_rel = np.random.choice(group_rels)
    
    # Position calculation
    margin = int(canvas_size * 0.2)
    group_y = canvas_size - margin if main_rel == 'above' else margin
    single_y = margin if main_rel == 'above' else canvas_size - margin
    
    # obj1 position (single, opposite side)
    c1_pos = (canvas_size // 2, single_y)
    
    # obj2, obj3 positions (group, side by side)
    group_spacing = int(canvas_size * 0.25)
    if group_rel == 'left_of':
        c2_pos = (canvas_size // 2 - group_spacing // 2, group_y)
        c3_pos = (canvas_size // 2 + group_spacing // 2, group_y)
    elif group_rel == 'right_of':
        c2_pos = (canvas_size // 2 + group_spacing // 2, group_y)
        c3_pos = (canvas_size // 2 - group_spacing // 2, group_y)
    else:  # next_to
        c2_pos = (canvas_size // 2 - group_spacing // 2, group_y)
        c3_pos = (canvas_size // 2 + group_spacing // 2, group_y)
    
    # Add jitter
    jitter = int(canvas_size * 0.03)
    def add_jitter(pos):
        return (pos[0] + np.random.randint(-jitter, jitter+1),
                pos[1] + np.random.randint(-jitter, jitter+1))
    
    c1_pos = add_jitter(c1_pos)
    c2_pos = add_jitter(c2_pos)
    c3_pos = add_jitter(c3_pos)
    
    obj1 = {'color': c1, 'shape': s1, 'center': c1_pos}
    obj2 = {'color': c2, 'shape': s2, 'center': c2_pos}
    obj3 = {'color': c3, 'shape': s3, 'center': c3_pos}
    
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    canvas = _place_object(canvas, obj1, canvas_size)
    canvas = _place_object(canvas, obj2, canvas_size)
    canvas = _place_object(canvas, obj3, canvas_size)
    
    # Hierarchical label with parentheses
    label = f"{c1} {s1} {main_rel} ({c2} {s2} {group_rel} {c3} {s3})"
    flat_label = f"{c1} {s1} {main_rel} {c2} {s2} {group_rel} {c3} {s3}"
    
    return HierarchicalPair(
        image=canvas,
        label=label,
        flat_label=flat_label,
        hierarchy_depth=2,
        objects=[obj1, obj2, obj3],
        structure={
            'rel': main_rel,
            'left': obj1,
            'right': {
                'rel': group_rel,
                'left': obj2,
                'right': obj3,
            }
        },
    )


def create_hierarchical_scene_depth3(
    colors: List[str],
    shapes: List[str],
    canvas_size: int = 56,
) -> HierarchicalPair:
    """
    Depth 3: Two groups relate to each other
    "(red circle left_of yellow square) above (blue square next_to green triangle)"
    
    Layout:
    - Top row: obj1, obj2 (group A)
    - Bottom row: obj3, obj4 (group B)
    """
    cs = np.random.choice(colors, 4, replace=True)
    # Ensure some variety
    while len(set(cs)) < 2:
        cs = np.random.choice(colors, 4, replace=True)
    c1, c2, c3, c4 = cs
    
    ss = np.random.choice(shapes, 4, replace=True)
    s1, s2, s3, s4 = ss
    
    # Main relation between groups
    main_rels = ['above', 'below']
    main_rel = np.random.choice(main_rels)
    
    # Group A relation (top or bottom depending on main_rel)
    group_a_rels = ['left_of', 'right_of', 'next_to']
    group_a_rel = np.random.choice(group_a_rels)
    
    # Group B relation
    group_b_rels = ['left_of', 'right_of', 'next_to']
    group_b_rel = np.random.choice(group_b_rels)
    
    # Positions
    margin = int(canvas_size * 0.2)
    spacing = int(canvas_size * 0.25)
    
    top_y = margin
    bottom_y = canvas_size - margin
    
    if main_rel == 'above':
        group_a_y, group_b_y = top_y, bottom_y
    else:
        group_a_y, group_b_y = bottom_y, top_y
    
    # Group A positions
    c1_pos = (canvas_size // 2 - spacing // 2, group_a_y)
    c2_pos = (canvas_size // 2 + spacing // 2, group_a_y)
    if group_a_rel == 'right_of':
        c1_pos, c2_pos = c2_pos, c1_pos
    
    # Group B positions
    c3_pos = (canvas_size // 2 - spacing // 2, group_b_y)
    c4_pos = (canvas_size // 2 + spacing // 2, group_b_y)
    if group_b_rel == 'right_of':
        c3_pos, c4_pos = c4_pos, c3_pos
    
    # Add jitter
    jitter = int(canvas_size * 0.02)
    def add_jitter(pos):
        return (pos[0] + np.random.randint(-jitter, jitter+1),
                pos[1] + np.random.randint(-jitter, jitter+1))
    
    c1_pos = add_jitter(c1_pos)
    c2_pos = add_jitter(c2_pos)
    c3_pos = add_jitter(c3_pos)
    c4_pos = add_jitter(c4_pos)
    
    obj1 = {'color': c1, 'shape': s1, 'center': c1_pos}
    obj2 = {'color': c2, 'shape': s2, 'center': c2_pos}
    obj3 = {'color': c3, 'shape': s3, 'center': c3_pos}
    obj4 = {'color': c4, 'shape': s4, 'center': c4_pos}
    
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    for obj in [obj1, obj2, obj3, obj4]:
        canvas = _place_object(canvas, obj, canvas_size)
    
    # Hierarchical label
    label = f"({c1} {s1} {group_a_rel} {c2} {s2}) {main_rel} ({c3} {s3} {group_b_rel} {c4} {s4})"
    flat_label = f"{c1} {s1} {group_a_rel} {c2} {s2} {main_rel} {c3} {s3} {group_b_rel} {c4} {s4}"
    
    return HierarchicalPair(
        image=canvas,
        label=label,
        flat_label=flat_label,
        hierarchy_depth=3,
        objects=[obj1, obj2, obj3, obj4],
        structure={
            'rel': main_rel,
            'left': {
                'rel': group_a_rel,
                'left': obj1,
                'right': obj2,
            },
            'right': {
                'rel': group_b_rel,
                'left': obj3,
                'right': obj4,
            }
        },
    )


def generate_hierarchical_pairs(
    colors: List[str] = None,
    shapes: List[str] = None,
    depths: List[int] = [1, 2, 3],
    n_per_depth: int = 200,
    canvas_size: int = 56,
) -> List[HierarchicalPair]:
    """Generate hierarchical composition dataset."""
    if colors is None:
        colors = ['red', 'blue', 'green', 'yellow']
    if shapes is None:
        shapes = ['circle', 'square', 'triangle']
    
    pairs = []
    
    generators = {
        1: create_hierarchical_scene_depth1,
        2: create_hierarchical_scene_depth2,
        3: create_hierarchical_scene_depth3,
    }
    
    for depth in depths:
        if depth not in generators:
            continue
        gen_fn = generators[depth]
        for _ in range(n_per_depth):
            pair = gen_fn(colors, shapes, canvas_size)
            pairs.append(pair)
    
    return pairs


if __name__ == '__main__':
    # Quick test
    import matplotlib.pyplot as plt
    
    pairs = generate_hierarchical_pairs(n_per_depth=3)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, depth in enumerate([1, 2, 3]):
        depth_pairs = [p for p in pairs if p.hierarchy_depth == depth]
        for j, pair in enumerate(depth_pairs[:3]):
            ax = axes[i, j]
            ax.imshow(pair.image)
            ax.set_title(f"Depth {depth}\n{pair.label}", fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('hierarchical_samples.png', dpi=150)
    print("Saved hierarchical_samples.png")
    
    print("\nSample labels:")
    for depth in [1, 2, 3]:
        depth_pairs = [p for p in pairs if p.hierarchy_depth == depth]
        if depth_pairs:
            print(f"Depth {depth}: {depth_pairs[0].label}")
