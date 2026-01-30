#!/usr/bin/env python3
"""
PredictiveATL: Temporal prediction for CHPL cognitive development.

Phase 1.1 of the Cognitive Development Roadmap:
- Predict future states from current activation patterns
- Learn temporal dynamics of moving objects
- Foundation for imagination and planning

Scientific question: Can distributed semantic codes learn temporal structure?
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# Import base CHPL components
from brain_crossmodal_learner import (
    DistributedATL, SimpleVisualCortex56, SimpleLanguageCortex,
    Hippocampus, BrainCrossModalLearner, DEVICE
)
from synthetic_environment import (
    SHAPES, COLORS, COLOR_RGB, SIZE_PARAMS, DRAW_FUNCTIONS,
    create_stimulus_on_canvas
)


class PredictiveATL(DistributedATL):
    """
    Distributed ATL + world model for temporal prediction.
    
    Predicts next activation pattern from current state.
    This is the foundation for:
    - Future state prediction
    - Counterfactual imagination
    - Planning
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        n_concepts: int = 200,
        temperature: float = 0.1,
        activation_threshold: float = 0.01,
        base_lr: float = 0.01,
        n_actions: int = 8,  # move directions + stay
    ):
        super().__init__(
            feature_dim=feature_dim,
            n_concepts=n_concepts,
            temperature=temperature,
            activation_threshold=activation_threshold,
            base_lr=base_lr,
        )
        
        self.n_actions = n_actions
        
        # Temporal prediction network (predicts next activation from current)
        self.predictor = nn.Sequential(
            nn.Linear(n_concepts, n_concepts * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_concepts * 2, n_concepts * 2),
            nn.ReLU(),
            nn.Linear(n_concepts * 2, n_concepts),
        ).to(DEVICE)
        
        # Action encoder (for conditioned prediction)
        self.action_encoder = nn.Sequential(
            nn.Linear(n_actions, n_concepts // 2),
            nn.ReLU(),
            nn.Linear(n_concepts // 2, n_concepts),
        ).to(DEVICE)
        
        # Velocity encoder (continuous movement)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, 32),  # dx, dy
            nn.ReLU(),
            nn.Linear(32, n_concepts),
        ).to(DEVICE)
        
        # Prediction optimizer
        self.pred_optimizer = torch.optim.Adam(
            list(self.predictor.parameters()) +
            list(self.action_encoder.parameters()) +
            list(self.velocity_encoder.parameters()),
            lr=1e-3,
            weight_decay=1e-5,
        )
        
        # Statistics
        self.prediction_losses = []
        
    def predict_next(self, current_activation: torch.Tensor) -> torch.Tensor:
        """
        Given current scene activation, predict next timestep.
        
        Args:
            current_activation: [n_concepts] activation pattern
            
        Returns:
            predicted_activation: [n_concepts] predicted next pattern
        """
        if current_activation.dim() == 1:
            current_activation = current_activation.unsqueeze(0)
        
        predicted = self.predictor(current_activation)
        return torch.softmax(predicted / self.temperature, dim=-1).squeeze(0)
    
    def predict_with_velocity(
        self,
        current_activation: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state conditioned on object velocity.
        
        Args:
            current_activation: [n_concepts] current activation
            velocity: [2] tensor (dx, dy) normalized velocity
            
        Returns:
            predicted_activation: [n_concepts] predicted pattern
        """
        if current_activation.dim() == 1:
            current_activation = current_activation.unsqueeze(0)
        if velocity.dim() == 1:
            velocity = velocity.unsqueeze(0)
        
        # Encode velocity
        vel_encoding = self.velocity_encoder(velocity.to(DEVICE))
        
        # Condition prediction on velocity
        conditioned = current_activation + vel_encoding
        
        predicted = self.predictor(conditioned)
        return torch.softmax(predicted / self.temperature, dim=-1).squeeze(0)
    
    def imagine(
        self,
        current_activation: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        """
        Imagine result of action without executing.
        
        Args:
            current_activation: current visual/language state
            action: action index (0-7 for directions, 8 for stay)
            
        Returns:
            imagined: predicted future activation pattern
        """
        if current_activation.dim() == 1:
            current_activation = current_activation.unsqueeze(0)
        
        # One-hot action encoding
        action_onehot = torch.zeros(1, self.n_actions, device=DEVICE)
        action_onehot[0, action] = 1.0
        
        action_embedding = self.action_encoder(action_onehot)
        
        # Condition prediction on action
        conditioned = current_activation + action_embedding
        
        imagined = self.predictor(conditioned)
        return torch.softmax(imagined / self.temperature, dim=-1).squeeze(0)
    
    def train_prediction(
        self,
        current_act: torch.Tensor,
        next_act: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Train predictor on a transition pair.
        
        Args:
            current_act: activation at time t
            next_act: ground truth activation at time t+1
            velocity: optional velocity for conditioned prediction
            
        Returns:
            loss value
        """
        self.pred_optimizer.zero_grad()
        
        if velocity is not None:
            predicted = self.predict_with_velocity(current_act, velocity)
        else:
            predicted = self.predict_next(current_act)
        
        # KL divergence loss (comparing probability distributions)
        loss = F.kl_div(
            (predicted + 1e-10).log(),
            next_act,
            reduction='batchmean'
        )
        
        # Also add cosine similarity loss for pattern matching
        cos_loss = 1 - F.cosine_similarity(
            predicted.unsqueeze(0),
            next_act.unsqueeze(0)
        ).mean()
        
        total_loss = loss + 0.5 * cos_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.pred_optimizer.step()
        
        self.prediction_losses.append(total_loss.item())
        return total_loss.item()
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction training statistics."""
        if not self.prediction_losses:
            return {'mean_loss': 0.0, 'recent_loss': 0.0}
        
        return {
            'mean_loss': np.mean(self.prediction_losses),
            'recent_loss': np.mean(self.prediction_losses[-100:]) if len(self.prediction_losses) >= 100 else np.mean(self.prediction_losses),
            'n_updates': len(self.prediction_losses),
        }


@dataclass
class TemporalFrame:
    """A single frame in a temporal sequence."""
    image: np.ndarray  # canvas_size x canvas_size x 3
    label: str
    position: Tuple[int, int]  # object center
    velocity: Tuple[float, float]  # dx, dy per frame
    timestep: int


@dataclass 
class TemporalSequence:
    """A sequence of frames showing object motion."""
    frames: List[TemporalFrame]
    object_shape: str
    object_color: str
    motion_type: str  # 'linear', 'circular', 'bounce', 'random'
    
    def __len__(self):
        return len(self.frames)


def create_moving_object_sequence(
    shape: str,
    color: str,
    start_pos: Tuple[int, int],
    velocity: Tuple[float, float],
    n_frames: int = 10,
    canvas_size: int = 56,
    motion_type: str = 'linear',
    noise: float = 0.02,
) -> TemporalSequence:
    """
    Create a sequence of frames showing an object in motion.
    
    Args:
        shape: object shape
        color: object color
        start_pos: starting (x, y) position
        velocity: (dx, dy) per frame
        n_frames: number of frames
        canvas_size: image size
        motion_type: 'linear', 'bounce', or 'circular'
        noise: image noise level
        
    Returns:
        TemporalSequence with frames
    """
    frames = []
    pos = list(start_pos)
    vel = list(velocity)
    
    margin = 10  # boundary margin
    
    for t in range(n_frames):
        # Create frame
        img = create_stimulus_on_canvas(
            shape=shape,
            color=color,
            size='small',
            canvas_size=canvas_size,
            center=(int(pos[0]), int(pos[1])),
            noise=noise,
        )
        
        label = f"{color} {shape} at ({int(pos[0])}, {int(pos[1])})"
        
        frames.append(TemporalFrame(
            image=img,
            label=label,
            position=(int(pos[0]), int(pos[1])),
            velocity=(vel[0], vel[1]),
            timestep=t,
        ))
        
        # Update position
        pos[0] += vel[0]
        pos[1] += vel[1]
        
        # Handle boundary conditions based on motion type
        if motion_type == 'bounce':
            if pos[0] <= margin or pos[0] >= canvas_size - margin:
                vel[0] = -vel[0]
                pos[0] = max(margin, min(canvas_size - margin, pos[0]))
            if pos[1] <= margin or pos[1] >= canvas_size - margin:
                vel[1] = -vel[1]
                pos[1] = max(margin, min(canvas_size - margin, pos[1]))
        elif motion_type == 'circular':
            # Circular motion around center
            center = canvas_size // 2
            radius = 15
            angle = (t + 1) * (2 * math.pi / n_frames)
            pos[0] = center + radius * math.cos(angle)
            pos[1] = center + radius * math.sin(angle)
            vel[0] = -radius * math.sin(angle) * (2 * math.pi / n_frames)
            vel[1] = radius * math.cos(angle) * (2 * math.pi / n_frames)
        else:  # linear - just clip to bounds
            pos[0] = max(margin, min(canvas_size - margin, pos[0]))
            pos[1] = max(margin, min(canvas_size - margin, pos[1]))
    
    return TemporalSequence(
        frames=frames,
        object_shape=shape,
        object_color=color,
        motion_type=motion_type,
    )


def generate_temporal_dataset(
    n_sequences: int = 500,
    frames_per_seq: int = 10,
    canvas_size: int = 56,
    motion_types: List[str] = ['linear', 'bounce'],
) -> List[TemporalSequence]:
    """
    Generate a dataset of temporal sequences for prediction training.
    
    Args:
        n_sequences: number of sequences to generate
        frames_per_seq: frames per sequence
        canvas_size: image size
        motion_types: types of motion to include
        
    Returns:
        List of TemporalSequence objects
    """
    sequences = []
    
    shapes = SHAPES[:4]  # circle, square, triangle, star
    colors = COLORS[:4]  # red, blue, green, yellow
    
    for _ in range(n_sequences):
        shape = np.random.choice(shapes)
        color = np.random.choice(colors)
        motion_type = np.random.choice(motion_types)
        
        # Random starting position (avoid edges)
        margin = 15
        start_x = np.random.randint(margin, canvas_size - margin)
        start_y = np.random.randint(margin, canvas_size - margin)
        
        # Random velocity
        speed = np.random.uniform(2, 5)
        angle = np.random.uniform(0, 2 * math.pi)
        vel_x = speed * math.cos(angle)
        vel_y = speed * math.sin(angle)
        
        seq = create_moving_object_sequence(
            shape=shape,
            color=color,
            start_pos=(start_x, start_y),
            velocity=(vel_x, vel_y),
            n_frames=frames_per_seq,
            canvas_size=canvas_size,
            motion_type=motion_type,
            noise=np.random.uniform(0.01, 0.03),
        )
        
        sequences.append(seq)
    
    return sequences


def create_occlusion_sequence(
    occluded_shape: str,
    occluded_color: str,
    occluder_shape: str = 'square',
    occluder_color: str = 'blue',
    n_frames: int = 15,
    canvas_size: int = 56,
) -> TemporalSequence:
    """
    Create a sequence where an object is temporarily occluded.
    
    Phases:
    1. Object visible, moving
    2. Object hidden behind occluder
    3. Object emerges on other side
    
    This tests object permanence.
    """
    frames = []
    
    center = canvas_size // 2
    occluder_pos = (center, center)
    
    # Object moves from left to right, passing behind occluder
    start_x = 10
    end_x = canvas_size - 10
    y = center
    
    for t in range(n_frames):
        progress = t / (n_frames - 1)
        obj_x = int(start_x + progress * (end_x - start_x))
        
        # Create canvas
        img = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
        
        # Draw occluder first (it's behind in layer order but we draw it first)
        # Actually we want occluder in front, so draw object first
        obj_img = create_stimulus_on_canvas(
            shape=occluded_shape,
            color=occluded_color,
            size='small',
            canvas_size=canvas_size,
            center=(obj_x, y),
            noise=0.01,
        )
        img = np.clip(img + obj_img, 0, 1)
        
        # Draw occluder on top
        occ_img = create_stimulus_on_canvas(
            shape=occluder_shape,
            color=occluder_color,
            size='large',
            canvas_size=canvas_size,
            center=occluder_pos,
            noise=0.0,
        )
        
        # Occluder overwrites where it exists
        occ_mask = occ_img.sum(axis=-1) > 0.1
        for c in range(3):
            img[:, :, c][occ_mask] = occ_img[:, :, c][occ_mask]
        
        # Determine visibility
        dist_to_occluder = abs(obj_x - center)
        is_visible = dist_to_occluder > 12  # object visible if far from occluder center
        
        if is_visible:
            label = f"{occluded_color} {occluded_shape} visible at ({obj_x}, {y})"
        else:
            label = f"{occluded_color} {occluded_shape} hidden behind {occluder_color} {occluder_shape}"
        
        vel_x = (end_x - start_x) / (n_frames - 1)
        
        frames.append(TemporalFrame(
            image=img.astype(np.float32),
            label=label,
            position=(obj_x, y),
            velocity=(vel_x, 0),
            timestep=t,
        ))
    
    return TemporalSequence(
        frames=frames,
        object_shape=occluded_shape,
        object_color=occluded_color,
        motion_type='occlusion',
    )


def generate_occlusion_dataset(
    n_sequences: int = 100,
    canvas_size: int = 56,
) -> List[TemporalSequence]:
    """Generate sequences for object permanence testing."""
    sequences = []
    
    shapes = SHAPES[:4]
    colors = COLORS[:4]
    occluder_colors = ['blue', 'green', 'purple']
    
    for _ in range(n_sequences):
        obj_shape = np.random.choice(shapes)
        obj_color = np.random.choice(colors)
        occ_color = np.random.choice(occluder_colors)
        
        # Ensure occluder is different from object
        while occ_color == obj_color:
            occ_color = np.random.choice(occluder_colors)
        
        seq = create_occlusion_sequence(
            occluded_shape=obj_shape,
            occluded_color=obj_color,
            occluder_color=occ_color,
            n_frames=15,
            canvas_size=canvas_size,
        )
        sequences.append(seq)
    
    return sequences


class PredictiveBrain(BrainCrossModalLearner):
    """
    Brain with predictive capabilities - extends BrainCrossModalLearner.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        n_concepts: int = 200,
        visual_input_size: int = 56,
    ):
        # Initialize base brain but we'll replace ATL
        super().__init__(
            feature_dim=feature_dim,
            n_concepts=n_concepts,
            visual_input_size=visual_input_size,
            use_distributed_atl=True,  # We'll replace with PredictiveATL
        )
        
        # Replace ATL with PredictiveATL
        self.atl = PredictiveATL(
            feature_dim=feature_dim,
            n_concepts=n_concepts,
        )
        
    def get_scene_activation(self, image: torch.Tensor) -> torch.Tensor:
        """Get activation pattern for a scene."""
        vis_features = self.visual(image)
        activation, _ = self.atl.get_activation_pattern(vis_features, 'visual')
        return activation
    
    def predict_next_scene(
        self,
        current_image: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict activation pattern for next timestep."""
        current_act = self.get_scene_activation(current_image)
        
        if velocity is not None:
            return self.atl.predict_with_velocity(current_act, velocity)
        else:
            return self.atl.predict_next(current_act)
    
    def train_on_transition(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
    ) -> float:
        """Train predictor on an image transition."""
        current_act = self.get_scene_activation(current_image)
        next_act = self.get_scene_activation(next_image)
        
        return self.atl.train_prediction(
            current_act.detach(),
            next_act.detach(),
            velocity,
        )
    
    def evaluate_prediction(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Evaluate prediction quality without training.
        
        Returns similarity metrics between predicted and actual next state.
        """
        with torch.no_grad():
            current_act = self.get_scene_activation(current_image)
            actual_next_act = self.get_scene_activation(next_image)
            
            if velocity is not None:
                predicted_next_act = self.atl.predict_with_velocity(current_act, velocity)
            else:
                predicted_next_act = self.atl.predict_next(current_act)
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(
                predicted_next_act.unsqueeze(0),
                actual_next_act.unsqueeze(0)
            ).item()
            
            # KL divergence
            kl_div = F.kl_div(
                (predicted_next_act + 1e-10).log(),
                actual_next_act,
                reduction='sum'
            ).item()
            
            # Top-k overlap (are the most active concepts the same?)
            k = 10
            pred_topk = torch.topk(predicted_next_act, k).indices
            actual_topk = torch.topk(actual_next_act, k).indices
            
            overlap = len(set(pred_topk.cpu().numpy()) & set(actual_topk.cpu().numpy())) / k
            
        return {
            'cosine_similarity': cos_sim,
            'kl_divergence': kl_div,
            'topk_overlap': overlap,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTIVE ATL - Temporal Learning Module")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Quick test
    print("\nInitializing PredictiveATL...")
    pred_atl = PredictiveATL()
    print(f"  n_concepts: {pred_atl.n_concepts}")
    print(f"  n_actions: {pred_atl.n_actions}")
    
    # Test prediction
    test_act = torch.softmax(torch.randn(pred_atl.n_concepts, device=DEVICE), dim=0)
    predicted = pred_atl.predict_next(test_act)
    print(f"  Prediction shape: {predicted.shape}")
    
    # Test velocity-conditioned prediction
    velocity = torch.tensor([3.0, -2.0], device=DEVICE)
    predicted_vel = pred_atl.predict_with_velocity(test_act, velocity)
    print(f"  Velocity-conditioned prediction shape: {predicted_vel.shape}")
    
    # Test imagination
    imagined = pred_atl.imagine(test_act, action=0)
    print(f"  Imagined state shape: {imagined.shape}")
    
    # Test sequence generation
    print("\nGenerating test sequence...")
    seq = create_moving_object_sequence(
        shape='circle',
        color='red',
        start_pos=(15, 28),
        velocity=(4, 0),
        n_frames=8,
    )
    print(f"  Sequence length: {len(seq)}")
    print(f"  First frame label: {seq.frames[0].label}")
    print(f"  Last frame label: {seq.frames[-1].label}")
    
    # Test occlusion sequence
    print("\nGenerating occlusion sequence...")
    occ_seq = create_occlusion_sequence(
        occluded_shape='circle',
        occluded_color='red',
        n_frames=15,
    )
    print(f"  Occlusion sequence length: {len(occ_seq)}")
    for i in [0, 7, 14]:
        print(f"  Frame {i}: {occ_seq.frames[i].label}")
    
    print("\n" + "=" * 60)
    print("PredictiveATL module ready for training")
    print("=" * 60)
