#!/usr/bin/env python3
"""
Synthetic Visual-Language Environment for Brain-Like Learning Research.

This is the scientifically valid approach:
- Controlled stimuli (not natural images)
- Small vocabulary (not 10k words)
- Ground truth known (not guessed from noisy labels)

The question: Can biological learning rules learn cross-modal binding from scratch?
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# ============================================================
# VISUAL STIMULI: Procedurally generated shapes
# ============================================================

SHAPES = ['circle', 'square', 'triangle', 'star', 'cross', 'diamond']
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
SIZES = ['small', 'large']

# RGB values for colors
COLOR_RGB = {
    'red': (255, 50, 50),
    'blue': (50, 50, 255),
    'green': (50, 200, 50),
    'yellow': (255, 255, 50),
    'purple': (180, 50, 180),
    'orange': (255, 150, 50),
}

# Size parameters (radius/half-width for 28x28 image)
SIZE_PARAMS = {
    'small': 5,
    'large': 10,
}


def draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw filled circle."""
    h, w = img.shape[0], img.shape[1]
    for y in range(max(0, cy-r), min(h, cy+r+1)):
        for x in range(max(0, cx-r), min(w, cx+r+1)):
            if (x - cx)**2 + (y - cy)**2 <= r**2:
                img[y, x] = color


def draw_square(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw filled square."""
    h, w = img.shape[0], img.shape[1]
    for y in range(max(0, cy-r), min(h, cy+r+1)):
        for x in range(max(0, cx-r), min(w, cx+r+1)):
            img[y, x] = color


def draw_triangle(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw filled triangle (pointing up)."""
    h, w = img.shape[0], img.shape[1]
    for y in range(max(0, cy-r), min(h, cy+r+1)):
        # Width at this height
        progress = (cy + r - y) / (2 * r) if r > 0 else 0
        half_width = int(r * progress)
        for x in range(max(0, cx-half_width), min(w, cx+half_width+1)):
            img[y, x] = color


def draw_star(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw 5-pointed star."""
    # Outer and inner radius
    r_outer = r
    r_inner = r // 2
    
    h, w = img.shape[0], img.shape[1]
    for y in range(max(0, cy-r), min(h, cy+r+1)):
        for x in range(max(0, cx-r), min(w, cx+r+1)):
            # Convert to polar
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)
            
            # 5-pointed star: radius varies with angle
            # 5 points = angle repeats every 2*pi/5
            angle_mod = (angle + math.pi) % (2 * math.pi / 5)
            # Interpolate between outer and inner
            t = abs(angle_mod - math.pi / 5) / (math.pi / 5)
            threshold = r_inner + t * (r_outer - r_inner)
            
            if dist <= threshold:
                img[y, x] = color


def draw_cross(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw plus/cross shape."""
    thickness = max(2, r // 3)
    # Horizontal bar
    h, w = img.shape[0], img.shape[1]
    for y in range(max(0, cy-thickness//2), min(h, cy+thickness//2+1)):
        for x in range(max(0, cx-r), min(w, cx+r+1)):
            img[y, x] = color
    # Vertical bar
    for y in range(max(0, cy-r), min(h, cy+r+1)):
        for x in range(max(0, cx-thickness//2), min(w, cx+thickness//2+1)):
            img[y, x] = color


def draw_diamond(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw diamond (rotated square)."""
    h, w = img.shape[0], img.shape[1]
    for y in range(max(0, cy-r), min(h, cy+r+1)):
        for x in range(max(0, cx-r), min(w, cx+r+1)):
            if abs(x - cx) + abs(y - cy) <= r:
                img[y, x] = color


DRAW_FUNCTIONS = {
    'circle': draw_circle,
    'square': draw_square,
    'triangle': draw_triangle,
    'star': draw_star,
    'cross': draw_cross,
    'diamond': draw_diamond,
}


def create_stimulus(shape: str, color: str, size: str, 
                   noise: float = 0.0, 
                   offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Create a 28x28 RGB image of a shape.
    
    Args:
        shape: One of SHAPES
        color: One of COLORS
        size: One of SIZES
        noise: Gaussian noise std (0-1)
        offset: (dx, dy) position offset from center
    
    Returns:
        28x28x3 numpy array (float32, 0-1 range)
    """
    img = np.zeros((28, 28, 3), dtype=np.uint8)
    
    # Center with offset
    cx = 14 + offset[0]
    cy = 14 + offset[1]
    
    # Size
    r = SIZE_PARAMS[size]
    
    # Color
    rgb = COLOR_RGB[color]
    
    # Draw shape
    draw_fn = DRAW_FUNCTIONS[shape]
    draw_fn(img, cx, cy, r, rgb)
    
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Add noise
    if noise > 0:
        img_float += np.random.randn(28, 28, 3).astype(np.float32) * noise
        img_float = np.clip(img_float, 0, 1)
    
    return img_float


def _scaled_radius(size: str, canvas_size: int) -> int:
    base = SIZE_PARAMS[size]
    return max(1, int(round(base * (canvas_size / 28.0))))


def create_stimulus_on_canvas(
    shape: str,
    color: str,
    size: str,
    canvas_size: int,
    center: Tuple[int, int],
    noise: float = 0.0,
) -> np.ndarray:
    """Create an RGB canvas with a single shape drawn at an explicit center."""
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    cx, cy = int(center[0]), int(center[1])
    r = _scaled_radius(size, canvas_size)
    rgb = COLOR_RGB[color]
    draw_fn = DRAW_FUNCTIONS[shape]
    draw_fn(img, cx, cy, r, rgb)

    img_float = img.astype(np.float32) / 255.0
    if noise > 0:
        img_float += np.random.randn(canvas_size, canvas_size, 3).astype(np.float32) * noise
        img_float = np.clip(img_float, 0, 1)

    return img_float


def get_spatial_relation(
    center1: Tuple[int, int],
    center2: Tuple[int, int],
    next_to_threshold: int = 12,
) -> str:
    """Infer a coarse spatial relation between object 1 and object 2."""
    x1, y1 = center1
    x2, y2 = center2
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) <= next_to_threshold and abs(dy) <= next_to_threshold:
        return 'next_to'

    if abs(dy) >= abs(dx):
        return 'above' if y1 < y2 else 'below'
    return 'left_of' if x1 < x2 else 'right_of'


def downsample_scene(scene: np.ndarray, out_size: int = 28) -> np.ndarray:
    """Downsample a scene (e.g., 56x56x3) to out_size x out_size x 3."""
    if scene.ndim != 3 or scene.shape[-1] != 3:
        raise ValueError(f"Expected scene [H, W, 3], got {tuple(scene.shape)}")

    x = torch.from_numpy(scene).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    x = F.interpolate(x, size=(out_size, out_size), mode='bilinear', align_corners=False)
    y = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return y.astype(np.float32)


# ============================================================
# VOCABULARY: 50 words for synthetic domain
# ============================================================

VOCABULARY = (
    # Shapes (6)
    SHAPES +
    # Colors (6)
    COLORS +
    # Sizes (2)
    SIZES +
    # Compositional words (10)
    ['object', 'thing', 'shape', 'colored', 'bright', 'dark', 'the', 'a', 'is', 'and'] +
    # Spatial (6)
    ['left', 'right', 'up', 'down', 'center', 'middle'] +
    # Descriptors (10)
    ['round', 'angular', 'pointed', 'flat', 'thick', 'thin', 'big', 'tiny', 'same', 'different'] +
    # Actions (for future) (10)
    ['see', 'look', 'find', 'point', 'show', 'name', 'describe', 'match', 'compare', 'identify'] +
    ['above', 'below', 'left_of', 'right_of', 'next_to']
)

# Map words to indices
WORD_TO_IDX = {word: i for i, word in enumerate(VOCABULARY)}
IDX_TO_WORD = {i: word for i, word in enumerate(VOCABULARY)}


# ============================================================
# CROSS-MODAL PAIRS: Controlled training data
# ============================================================

@dataclass
class CrossModalPair:
    """A single training example."""
    image: np.ndarray          # 28x28x3
    label: str                 # e.g., "red circle"
    shape: str                 # e.g., "circle"
    color: str                 # e.g., "red"
    size: str                  # e.g., "small"
    
    def to_tensor(self) -> Tuple[torch.Tensor, str]:
        """Convert image to tensor."""
        img_t = torch.from_numpy(self.image).float()
        return img_t, self.label


def generate_training_pairs(n_per_combination: int = 10,
                           noise_range: Tuple[float, float] = (0.0, 0.05),
                           offset_range: int = 2) -> List[CrossModalPair]:
    """
    Generate training pairs with controlled variation.
    
    For each (shape, color, size) combination:
    - Generate n_per_combination images with slight variations
    - Pair with label "{size} {color} {shape}" or "{color} {shape}"
    """
    pairs = []
    
    for shape in SHAPES:
        for color in COLORS:
            for size in SIZES:
                for _ in range(n_per_combination):
                    # Random noise and offset
                    noise = np.random.uniform(*noise_range)
                    offset = (
                        np.random.randint(-offset_range, offset_range + 1),
                        np.random.randint(-offset_range, offset_range + 1)
                    )
                    
                    img = create_stimulus(shape, color, size, noise=noise, offset=offset)
                    
                    # Label: sometimes include size, sometimes not
                    if np.random.random() < 0.5:
                        label = f"{size} {color} {shape}"
                    else:
                        label = f"{color} {shape}"
                    
                    pairs.append(CrossModalPair(
                        image=img,
                        label=label,
                        shape=shape,
                        color=color,
                        size=size,
                    ))
    
    return pairs


def generate_test_pairs(include_unseen_combinations: bool = True) -> Dict[str, List[CrossModalPair]]:
    """
    Generate test pairs for scientific evaluation.
    
    Returns dict with:
    - 'basic': Simple pairs (same as training distribution)
    - 'unseen_combinations': Novel shape+color combinations not in training
    - 'noisy': High noise versions
    - 'shifted': Off-center versions
    """
    tests = {
        'basic': [],
        'unseen_combinations': [],
        'noisy': [],
        'shifted': [],
    }
    
    # Basic test: same as training
    for shape in SHAPES[:3]:  # circle, square, triangle
        for color in COLORS[:3]:  # red, blue, green
            img = create_stimulus(shape, color, 'small')
            tests['basic'].append(CrossModalPair(
                image=img,
                label=f"{color} {shape}",
                shape=shape,
                color=color,
                size='small',
            ))
    
    # Unseen combinations (for generalization test)
    # Training might have: red circle, blue square
    # Test: red square, blue circle (novel combinations)
    if include_unseen_combinations:
        unseen = [
            ('circle', 'purple'),
            ('square', 'orange'),
            ('triangle', 'yellow'),
            ('star', 'blue'),
        ]
        for shape, color in unseen:
            img = create_stimulus(shape, color, 'large')
            tests['unseen_combinations'].append(CrossModalPair(
                image=img,
                label=f"{color} {shape}",
                shape=shape,
                color=color,
                size='large',
            ))
    
    # Noisy versions (robustness test)
    for shape in SHAPES[:2]:
        for color in COLORS[:2]:
            img = create_stimulus(shape, color, 'small', noise=0.15)
            tests['noisy'].append(CrossModalPair(
                image=img,
                label=f"{color} {shape}",
                shape=shape,
                color=color,
                size='small',
            ))
    
    # Shifted versions (position invariance)
    for shape in SHAPES[:2]:
        for color in COLORS[:2]:
            img = create_stimulus(shape, color, 'small', offset=(4, -3))
            tests['shifted'].append(CrossModalPair(
                image=img,
                label=f"{color} {shape}",
                shape=shape,
                color=color,
                size='small',
            ))
    
    return tests


@dataclass
class TwoObjectPair:
    """A two-object relational training example."""
    image: np.ndarray
    label: str
    relation: str
    obj1: Dict[str, object]
    obj2: Dict[str, object]

    def to_tensor(self) -> Tuple[torch.Tensor, str]:
        img_t = torch.from_numpy(self.image).float()
        return img_t, self.label


def create_two_object_scene(
    obj1: Dict[str, object],
    obj2: Dict[str, object],
    canvas_size: int = 56,
    noise: float = 0.0,
) -> Tuple[np.ndarray, str, str]:
    """Create a 56x56 (default) scene with two objects and an inferred relation."""
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)

    img1 = create_stimulus_on_canvas(
        shape=str(obj1['shape']),
        color=str(obj1['color']),
        size=str(obj1.get('size', 'small')),
        canvas_size=canvas_size,
        center=tuple(obj1['center']),
        noise=0.0,
    )
    img2 = create_stimulus_on_canvas(
        shape=str(obj2['shape']),
        color=str(obj2['color']),
        size=str(obj2.get('size', 'small')),
        canvas_size=canvas_size,
        center=tuple(obj2['center']),
        noise=0.0,
    )

    img = np.clip(img1 + img2, 0.0, 1.0)
    if noise > 0:
        img += np.random.randn(canvas_size, canvas_size, 3).astype(np.float32) * noise
        img = np.clip(img, 0.0, 1.0)

    relation = get_spatial_relation(tuple(obj1['center']), tuple(obj2['center']))
    label = f"{obj1['color']} {obj1['shape']} {relation} {obj2['color']} {obj2['shape']}"
    return img.astype(np.float32), label, relation


def generate_two_object_pairs(
    colors: List[str],
    shapes: List[str],
    relations: List[str],
    n_per_combination: int = 10,
    canvas_size: int = 56,
    noise_range: Tuple[float, float] = (0.0, 0.05),
) -> List[TwoObjectPair]:
    """Generate a small relational dataset: (obj1, relation, obj2) on a 56x56 canvas."""
    pairs: List[TwoObjectPair] = []

    margin = int(round(canvas_size * 0.25))
    c_low = margin
    c_high = canvas_size - margin

    centers_for_relation = {
        'above': ((canvas_size // 2, c_low), (canvas_size // 2, c_high)),
        'below': ((canvas_size // 2, c_high), (canvas_size // 2, c_low)),
        'left_of': ((c_low, canvas_size // 2), (c_high, canvas_size // 2)),
        'right_of': ((c_high, canvas_size // 2), (c_low, canvas_size // 2)),
        'next_to': ((canvas_size // 2 - margin // 2, canvas_size // 2), (canvas_size // 2 + margin // 2, canvas_size // 2)),
    }

    for rel in relations:
        if rel not in centers_for_relation:
            continue
        c1_base, c2_base = centers_for_relation[rel]

        for color1 in colors:
            for shape1 in shapes:
                for color2 in colors:
                    for shape2 in shapes:
                        if (color1 == color2) and (shape1 == shape2):
                            continue

                        for _ in range(n_per_combination):
                            jitter = int(round(canvas_size * 0.03))
                            c1 = (
                                int(c1_base[0] + np.random.randint(-jitter, jitter + 1)),
                                int(c1_base[1] + np.random.randint(-jitter, jitter + 1)),
                            )
                            c2 = (
                                int(c2_base[0] + np.random.randint(-jitter, jitter + 1)),
                                int(c2_base[1] + np.random.randint(-jitter, jitter + 1)),
                            )

                            noise = float(np.random.uniform(*noise_range))
                            obj1 = {'shape': shape1, 'color': color1, 'size': 'small', 'center': c1}
                            obj2 = {'shape': shape2, 'color': color2, 'size': 'small', 'center': c2}
                            img, label, inferred = create_two_object_scene(
                                obj1=obj1,
                                obj2=obj2,
                                canvas_size=canvas_size,
                                noise=noise,
                            )

                            if inferred != rel:
                                continue

                            pairs.append(TwoObjectPair(
                                image=img,
                                label=label,
                                relation=inferred,
                                obj1=obj1,
                                obj2=obj2,
                            ))

    return pairs


# ============================================================
# VISUALIZATION (for debugging)
# ============================================================

def visualize_samples(pairs: List[CrossModalPair], n: int = 8, save_path: str = None):
    """Visualize sample pairs."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, n // 2, figsize=(12, 6))
        axes = axes.flatten()
        
        for i, (ax, pair) in enumerate(zip(axes, pairs[:n])):
            ax.imshow(pair.image)
            ax.set_title(pair.label, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    except ImportError:
        print("matplotlib not available for visualization")


# ============================================================
# MAIN: Test the environment
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("SYNTHETIC VISUAL-LANGUAGE ENVIRONMENT")
    print("="*60)
    
    print(f"\nShapes: {SHAPES}")
    print(f"Colors: {COLORS}")
    print(f"Sizes: {SIZES}")
    print(f"Vocabulary: {len(VOCABULARY)} words")
    
    # Generate training pairs
    print("\n" + "-"*40)
    print("Generating training pairs...")
    train_pairs = generate_training_pairs(n_per_combination=10)
    print(f"  Total training pairs: {len(train_pairs)}")
    print(f"  Unique combinations: {len(SHAPES) * len(COLORS) * len(SIZES)}")
    
    # Sample labels
    print("\n  Sample pairs:")
    for pair in train_pairs[:5]:
        print(f"    {pair.label} → image {pair.image.shape}")
    
    # Generate test pairs
    print("\n" + "-"*40)
    print("Generating test pairs...")
    test_pairs = generate_test_pairs()
    for name, pairs in test_pairs.items():
        print(f"  {name}: {len(pairs)} pairs")
    
    # Visualize
    print("\n" + "-"*40)
    print("Visualizing samples...")
    visualize_samples(train_pairs[:8], save_path="synthetic_samples.png")
    
    print("\n" + "="*60)
    print("ENVIRONMENT READY FOR BRAIN-LIKE LEARNING")
    print("="*60)
