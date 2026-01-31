#!/usr/bin/env python3
"""
Synthetic Causal Interaction Dataset

Generates sequences showing:
1. Push interactions (object A pushes object B)
2. Block interactions (object A blocks object B)
3. Independent motion (no interaction - control)
4. Chain reactions (A pushes B which pushes C)

For training CausalATL to understand physical causality.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

from synthetic_environment import (
    create_stimulus_on_canvas, SHAPES, COLORS
)


class CausalInteractionType(Enum):
    """Types of causal interactions."""
    PUSH = 'push'
    BLOCK = 'block'
    INDEPENDENT = 'independent'
    CHAIN = 'chain'


@dataclass
class CausalFrame:
    """Single frame in a causal sequence."""
    image: np.ndarray
    objects: List[Dict]  # List of {shape, color, position, velocity}
    timestep: int


@dataclass
class CausalSequence:
    """Sequence showing a causal interaction."""
    frames: List[CausalFrame]
    interaction_type: CausalInteractionType
    cause_object: Optional[str]  # e.g., "red circle"
    effect_object: Optional[str]  # e.g., "blue square"
    cause_frame: int  # Frame where cause occurs
    effect_frame: int  # Frame where effect is visible


def draw_object(
    canvas: np.ndarray,
    shape: str,
    color: str,
    position: Tuple[int, int],
    size: str = 'small',
) -> np.ndarray:
    """Draw an object on the canvas by creating stimulus and compositing."""
    # Create stimulus on white background
    stimulus = create_stimulus_on_canvas(
        shape=shape,
        color=color,
        size=size,
        canvas_size=canvas.shape[0],
        center=position,
        noise=0.01,
    )
    
    # Composite: where stimulus is not white (background), use stimulus
    # Detect non-white pixels (object pixels)
    is_object = np.any(stimulus < 0.95, axis=-1)
    
    # Copy object pixels to canvas
    result = canvas.copy()
    result[is_object] = stimulus[is_object]
    
    return result


def create_push_sequence(
    pusher_shape: str = 'circle',
    pusher_color: str = 'red',
    pushed_shape: str = 'square',
    pushed_color: str = 'blue',
    canvas_size: int = 56,
    n_frames: int = 6,
) -> CausalSequence:
    """
    Create sequence where object A pushes object B.
    
    Timeline:
    - Frame 0-1: A approaches B
    - Frame 2: A touches B (cause)
    - Frame 3-5: B moves away, A stops (effect)
    """
    frames = []
    
    # Initial positions
    pusher_x = 12
    pusher_y = 28
    pushed_x = 28
    pushed_y = 28
    
    pusher_vel = 5  # Pixels per frame
    
    for t in range(n_frames):
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        
        if t < 3:
            # Pusher approaches
            curr_pusher_x = pusher_x + t * pusher_vel
            curr_pushed_x = pushed_x
        else:
            # Contact made, pusher stops, pushed moves
            curr_pusher_x = pusher_x + 2 * pusher_vel  # Stopped at contact
            curr_pushed_x = pushed_x + (t - 2) * pusher_vel  # Now moving
        
        objects = [
            {'shape': pusher_shape, 'color': pusher_color, 
             'position': (int(curr_pusher_x), pusher_y), 'velocity': (pusher_vel if t < 3 else 0, 0)},
            {'shape': pushed_shape, 'color': pushed_color,
             'position': (int(curr_pushed_x), pushed_y), 'velocity': (0 if t < 3 else pusher_vel, 0)},
        ]
        
        # Draw objects
        for obj in objects:
            canvas = draw_object(
                canvas, obj['shape'], obj['color'], obj['position']
            )
        
        frames.append(CausalFrame(
            image=canvas,
            objects=objects,
            timestep=t,
        ))
    
    return CausalSequence(
        frames=frames,
        interaction_type=CausalInteractionType.PUSH,
        cause_object=f"{pusher_color} {pusher_shape}",
        effect_object=f"{pushed_color} {pushed_shape}",
        cause_frame=2,
        effect_frame=3,
    )


def create_block_sequence(
    blocker_shape: str = 'square',
    blocker_color: str = 'green',
    blocked_shape: str = 'circle',
    blocked_color: str = 'red',
    canvas_size: int = 56,
    n_frames: int = 6,
) -> CausalSequence:
    """
    Create sequence where stationary object A blocks moving object B.
    
    Timeline:
    - Frame 0-2: B approaches stationary A
    - Frame 3-5: B stops at A (blocked)
    """
    frames = []
    
    # Blocker is stationary
    blocker_x = 28
    blocker_y = 28
    
    # Blocked starts moving toward blocker
    blocked_x = 10
    blocked_y = 28
    blocked_vel = 5
    
    for t in range(n_frames):
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        
        # Blocked moves until hitting blocker
        if t < 3:
            curr_blocked_x = blocked_x + t * blocked_vel
        else:
            curr_blocked_x = blocked_x + 2 * blocked_vel  # Stopped at blocker
        
        objects = [
            {'shape': blocker_shape, 'color': blocker_color,
             'position': (blocker_x, blocker_y), 'velocity': (0, 0)},
            {'shape': blocked_shape, 'color': blocked_color,
             'position': (int(curr_blocked_x), blocked_y), 'velocity': (blocked_vel if t < 3 else 0, 0)},
        ]
        
        # Draw objects
        for obj in objects:
            canvas = draw_object(
                canvas, obj['shape'], obj['color'], obj['position']
            )
        
        frames.append(CausalFrame(
            image=canvas,
            objects=objects,
            timestep=t,
        ))
    
    return CausalSequence(
        frames=frames,
        interaction_type=CausalInteractionType.BLOCK,
        cause_object=f"{blocker_color} {blocker_shape}",
        effect_object=f"{blocked_color} {blocked_shape}",
        cause_frame=2,
        effect_frame=3,
    )


def create_independent_sequence(
    obj1_shape: str = 'circle',
    obj1_color: str = 'red',
    obj2_shape: str = 'square',
    obj2_color: str = 'blue',
    canvas_size: int = 56,
    n_frames: int = 6,
) -> CausalSequence:
    """
    Create sequence where two objects move independently (no interaction).
    
    Control condition: objects don't touch, move on parallel paths.
    """
    frames = []
    
    # Object 1 moves right
    obj1_x = 10
    obj1_y = 15
    obj1_vel = 4
    
    # Object 2 moves left (parallel, no intersection)
    obj2_x = 45
    obj2_y = 40
    obj2_vel = -3
    
    for t in range(n_frames):
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        
        curr_obj1_x = obj1_x + t * obj1_vel
        curr_obj2_x = obj2_x + t * obj2_vel
        
        objects = [
            {'shape': obj1_shape, 'color': obj1_color,
             'position': (int(curr_obj1_x), obj1_y), 'velocity': (obj1_vel, 0)},
            {'shape': obj2_shape, 'color': obj2_color,
             'position': (int(curr_obj2_x), obj2_y), 'velocity': (obj2_vel, 0)},
        ]
        
        for obj in objects:
            canvas = draw_object(
                canvas, obj['shape'], obj['color'], obj['position']
            )
        
        frames.append(CausalFrame(
            image=canvas,
            objects=objects,
            timestep=t,
        ))
    
    return CausalSequence(
        frames=frames,
        interaction_type=CausalInteractionType.INDEPENDENT,
        cause_object=None,
        effect_object=None,
        cause_frame=-1,
        effect_frame=-1,
    )


def create_chain_sequence(
    obj1_shape: str = 'circle',
    obj1_color: str = 'red',
    obj2_shape: str = 'square',
    obj2_color: str = 'blue',
    obj3_shape: str = 'triangle',
    obj3_color: str = 'green',
    canvas_size: int = 56,
    n_frames: int = 8,
) -> CausalSequence:
    """
    Create chain reaction: A pushes B, B pushes C.
    
    Timeline:
    - Frame 0-2: A approaches B
    - Frame 3: A hits B
    - Frame 4-5: B moves toward C, A stops
    - Frame 6: B hits C
    - Frame 7: C moves, B stops
    """
    frames = []
    
    # Initial positions in a line
    obj1_x = 8
    obj2_x = 22
    obj3_x = 36
    y = 28
    vel = 4
    
    for t in range(n_frames):
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        
        if t < 3:
            # A approaching B
            curr_obj1_x = obj1_x + t * vel
            curr_obj2_x = obj2_x
            curr_obj3_x = obj3_x
        elif t < 6:
            # A stopped, B moving toward C
            curr_obj1_x = obj1_x + 2 * vel
            curr_obj2_x = obj2_x + (t - 2) * vel
            curr_obj3_x = obj3_x
        else:
            # A and B stopped, C moving
            curr_obj1_x = obj1_x + 2 * vel
            curr_obj2_x = obj2_x + 3 * vel
            curr_obj3_x = obj3_x + (t - 5) * vel
        
        objects = [
            {'shape': obj1_shape, 'color': obj1_color,
             'position': (int(curr_obj1_x), y), 'velocity': (vel if t < 3 else 0, 0)},
            {'shape': obj2_shape, 'color': obj2_color,
             'position': (int(curr_obj2_x), y), 'velocity': (vel if 3 <= t < 6 else 0, 0)},
            {'shape': obj3_shape, 'color': obj3_color,
             'position': (int(curr_obj3_x), y), 'velocity': (vel if t >= 6 else 0, 0)},
        ]
        
        for obj in objects:
            canvas = draw_object(
                canvas, obj['shape'], obj['color'], obj['position']
            )
        
        frames.append(CausalFrame(
            image=canvas,
            objects=objects,
            timestep=t,
        ))
    
    return CausalSequence(
        frames=frames,
        interaction_type=CausalInteractionType.CHAIN,
        cause_object=f"{obj1_color} {obj1_shape}",
        effect_object=f"{obj3_color} {obj3_shape}",  # Ultimate effect
        cause_frame=3,
        effect_frame=7,
    )


def generate_causal_dataset(
    n_push: int = 200,
    n_block: int = 200,
    n_independent: int = 200,
    n_chain: int = 100,
) -> List[CausalSequence]:
    """
    Generate full causal interaction dataset.
    
    Returns list of CausalSequence objects with varied object combinations.
    """
    sequences = []
    
    shapes = ['circle', 'square', 'triangle', 'star']
    colors = ['red', 'blue', 'green', 'yellow']
    
    # Push sequences
    print(f"Generating {n_push} push sequences...")
    for _ in range(n_push):
        pusher_shape = np.random.choice(shapes)
        pushed_shape = np.random.choice(shapes)
        pusher_color = np.random.choice(colors)
        pushed_color = np.random.choice([c for c in colors if c != pusher_color])
        
        seq = create_push_sequence(
            pusher_shape=pusher_shape,
            pusher_color=pusher_color,
            pushed_shape=pushed_shape,
            pushed_color=pushed_color,
        )
        sequences.append(seq)
    
    # Block sequences
    print(f"Generating {n_block} block sequences...")
    for _ in range(n_block):
        blocker_shape = np.random.choice(shapes)
        blocked_shape = np.random.choice(shapes)
        blocker_color = np.random.choice(colors)
        blocked_color = np.random.choice([c for c in colors if c != blocker_color])
        
        seq = create_block_sequence(
            blocker_shape=blocker_shape,
            blocker_color=blocker_color,
            blocked_shape=blocked_shape,
            blocked_color=blocked_color,
        )
        sequences.append(seq)
    
    # Independent sequences
    print(f"Generating {n_independent} independent sequences...")
    for _ in range(n_independent):
        obj1_shape = np.random.choice(shapes)
        obj2_shape = np.random.choice(shapes)
        obj1_color = np.random.choice(colors)
        obj2_color = np.random.choice([c for c in colors if c != obj1_color])
        
        seq = create_independent_sequence(
            obj1_shape=obj1_shape,
            obj1_color=obj1_color,
            obj2_shape=obj2_shape,
            obj2_color=obj2_color,
        )
        sequences.append(seq)
    
    # Chain sequences
    print(f"Generating {n_chain} chain sequences...")
    for _ in range(n_chain):
        obj1_shape = np.random.choice(shapes)
        obj2_shape = np.random.choice(shapes)
        obj3_shape = np.random.choice(shapes)
        obj1_color = np.random.choice(colors)
        remaining_colors = [c for c in colors if c != obj1_color]
        obj2_color = np.random.choice(remaining_colors)
        remaining_colors = [c for c in remaining_colors if c != obj2_color]
        obj3_color = np.random.choice(remaining_colors) if remaining_colors else 'purple'
        
        seq = create_chain_sequence(
            obj1_shape=obj1_shape,
            obj1_color=obj1_color,
            obj2_shape=obj2_shape,
            obj2_color=obj2_color,
            obj3_shape=obj3_shape,
            obj3_color=obj3_color,
        )
        sequences.append(seq)
    
    print(f"Total: {len(sequences)} causal sequences")
    return sequences


def get_interaction_label(interaction_type: CausalInteractionType) -> int:
    """Convert interaction type to integer label."""
    mapping = {
        CausalInteractionType.PUSH: 0,
        CausalInteractionType.BLOCK: 1,
        CausalInteractionType.INDEPENDENT: 2,
        CausalInteractionType.CHAIN: 3,
    }
    return mapping.get(interaction_type, 2)


if __name__ == "__main__":
    print("Testing causal sequence generation...")
    
    # Test each type
    push_seq = create_push_sequence()
    print(f"Push sequence: {len(push_seq.frames)} frames")
    print(f"  Cause: {push_seq.cause_object} at frame {push_seq.cause_frame}")
    print(f"  Effect: {push_seq.effect_object} at frame {push_seq.effect_frame}")
    
    block_seq = create_block_sequence()
    print(f"\nBlock sequence: {len(block_seq.frames)} frames")
    print(f"  Cause: {block_seq.cause_object}")
    print(f"  Effect: {block_seq.effect_object}")
    
    indep_seq = create_independent_sequence()
    print(f"\nIndependent sequence: {len(indep_seq.frames)} frames")
    print(f"  No causal interaction")
    
    chain_seq = create_chain_sequence()
    print(f"\nChain sequence: {len(chain_seq.frames)} frames")
    print(f"  Initial cause: {chain_seq.cause_object}")
    print(f"  Final effect: {chain_seq.effect_object}")
    
    # Generate small dataset
    print("\nGenerating test dataset...")
    dataset = generate_causal_dataset(n_push=10, n_block=10, n_independent=10, n_chain=5)
    print(f"Generated {len(dataset)} sequences")
    
    # Count types
    type_counts = {}
    for seq in dataset:
        t = seq.interaction_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Type distribution: {type_counts}")
    
    print("\n✓ Causal dataset generation working!")
