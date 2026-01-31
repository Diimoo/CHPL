#!/usr/bin/env python3
"""
Phase 6: Video Understanding

Learn domain knowledge from video sequences.

Capabilities:
1. Extract temporal patterns from video frames
2. Predict future frames and detect prediction errors
3. Learn physical laws from observations
4. Associate visual patterns with language explanations

This enables learning physics, biology, etc. from educational videos.
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from hierarchical_atl import AbstractBrain, DEVICE
from predictive_atl import PredictiveATL


@dataclass
class VideoFrame:
    """A single frame from a video."""
    image: np.ndarray  # [H, W, 3]
    timestamp: float   # seconds
    description: str = ""  # optional narration


@dataclass
class VideoClip:
    """A sequence of video frames."""
    frames: List[VideoFrame]
    title: str = ""
    domain: str = ""  # physics, biology, etc.


@dataclass
class LearnedPattern:
    """A pattern learned from video observation."""
    before_activation: torch.Tensor
    after_activation: torch.Tensor
    change_vector: torch.Tensor
    description: str
    domain: str
    confidence: float


class VideoLearner:
    """
    Learn domain knowledge from video sequences.
    
    Uses prediction errors to identify novel patterns worth learning.
    Associates visual transitions with language descriptions.
    """
    
    def __init__(self, brain: AbstractBrain):
        self.brain = brain
        self.n_concepts = brain.atl.n_concepts
        
        # Learned patterns library
        self.patterns: List[LearnedPattern] = []
        
        # Domain knowledge bases
        self.domain_knowledge: Dict[str, List[LearnedPattern]] = {
            'physics': [],
            'biology': [],
            'chemistry': [],
            'general': [],
        }
        
        # Statistics
        self.frames_processed = 0
        self.patterns_learned = 0
        self.prediction_errors: List[float] = []
    
    def process_frame(self, frame: VideoFrame) -> torch.Tensor:
        """Convert video frame to activation pattern."""
        image = torch.from_numpy(frame.image).float()
        
        # Handle different image formats
        if image.dim() == 2:  # Grayscale
            image = image.unsqueeze(-1).repeat(1, 1, 3)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        image = image.permute(2, 0, 1).to(DEVICE)
        
        # Resize if needed
        if image.shape[1] != 56 or image.shape[2] != 56:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(56, 56), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Get activation
        activation = self.brain.get_scene_activation(image)
        return activation
    
    def predict_next_frame(
        self, 
        current_activation: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict activation for next frame."""
        # Simple prediction: assume similar to current with small noise
        # This allows us to detect significant changes as prediction errors
        noise = torch.randn_like(current_activation) * 0.05
        predicted = current_activation + noise
        predicted = F.softmax(predicted / 0.2, dim=-1)
        return predicted
    
    def compute_prediction_error(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Compute error between predicted and actual activation."""
        similarity = F.cosine_similarity(
            predicted.unsqueeze(0),
            actual.unsqueeze(0)
        ).item()
        
        error = 1.0 - similarity
        return error
    
    def watch_video(
        self,
        clip: VideoClip,
        learning_threshold: float = 0.3,
    ) -> Dict:
        """
        Watch a video clip and learn from it.
        
        Args:
            clip: Video clip to process
            learning_threshold: Prediction error threshold for learning
        
        Returns:
            Statistics about what was learned
        """
        print(f"  Watching: {clip.title} ({len(clip.frames)} frames)")
        
        stats = {
            'frames': len(clip.frames),
            'patterns_learned': 0,
            'mean_error': 0,
            'max_error': 0,
        }
        
        errors = []
        previous_activation = None
        
        for i, frame in enumerate(clip.frames):
            # Process current frame
            current_activation = self.process_frame(frame)
            self.frames_processed += 1
            
            if previous_activation is not None:
                # Predict what we expected
                predicted = self.predict_next_frame(previous_activation)
                
                # Compute error
                error = self.compute_prediction_error(predicted, current_activation)
                errors.append(error)
                self.prediction_errors.append(error)
                
                # If error is high, this is a novel pattern worth learning
                if error > learning_threshold:
                    pattern = self.learn_pattern(
                        previous_activation,
                        current_activation,
                        frame.description,
                        clip.domain,
                    )
                    if pattern is not None:
                        stats['patterns_learned'] += 1
            
            previous_activation = current_activation
        
        if errors:
            stats['mean_error'] = np.mean(errors)
            stats['max_error'] = np.max(errors)
        
        return stats
    
    def learn_pattern(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        description: str,
        domain: str,
    ) -> Optional[LearnedPattern]:
        """
        Learn a new pattern from a transition.
        
        Args:
            before: Activation before the transition
            after: Activation after the transition
            description: Optional language description
            domain: Domain category
        
        Returns:
            Learned pattern or None if redundant
        """
        # Compute change vector
        change = after - before
        
        # Check if this pattern is novel (not too similar to existing)
        for existing in self.patterns:
            similarity = F.cosine_similarity(
                change.unsqueeze(0),
                existing.change_vector.unsqueeze(0)
            ).item()
            
            if similarity > 0.9:  # Too similar, skip
                return None
        
        # Create new pattern
        pattern = LearnedPattern(
            before_activation=before.detach().clone(),
            after_activation=after.detach().clone(),
            change_vector=change.detach().clone(),
            description=description,
            domain=domain,
            confidence=1.0,
        )
        
        self.patterns.append(pattern)
        self.patterns_learned += 1
        
        # Add to domain knowledge
        if domain in self.domain_knowledge:
            self.domain_knowledge[domain].append(pattern)
        else:
            self.domain_knowledge['general'].append(pattern)
        
        return pattern
    
    def explain_transition(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> str:
        """
        Explain what happened between two states.
        
        Finds the closest learned pattern and returns its description.
        """
        change = after - before
        
        best_pattern = None
        best_similarity = -1
        
        for pattern in self.patterns:
            similarity = F.cosine_similarity(
                change.unsqueeze(0),
                pattern.change_vector.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = pattern
        
        if best_pattern is None:
            return "Unknown transition"
        
        if best_similarity > 0.5:
            return f"{best_pattern.description} (confidence: {best_similarity:.2f})"
        else:
            return f"Possibly: {best_pattern.description} (low confidence: {best_similarity:.2f})"
    
    def query_domain(self, domain: str) -> List[str]:
        """List all learned patterns in a domain."""
        patterns = self.domain_knowledge.get(domain, [])
        return [p.description for p in patterns if p.description]


def create_physics_video() -> VideoClip:
    """Create a synthetic physics video: ball falling and bouncing."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    # Ball falling sequence
    ball_y_positions = [10, 15, 22, 31, 42, 50, 42, 35, 42, 48, 50]
    
    for i, y in enumerate(ball_y_positions):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Draw ball
        ball = create_stimulus_on_canvas(
            shape='circle',
            color='red',
            size='small',
            canvas_size=56,
            center=(28, y),
        )
        
        # Composite
        is_ball = np.any(ball < 0.95, axis=-1)
        canvas[is_ball] = ball[is_ball]
        
        # Description based on phase
        if i < 5:
            desc = "ball falls down due to gravity"
        elif i == 5:
            desc = "ball hits ground"
        elif i < 8:
            desc = "ball bounces up after hitting ground"
        else:
            desc = "ball falls again after bounce"
        
        frames.append(VideoFrame(
            image=canvas,
            timestamp=i * 0.1,
            description=desc,
        ))
    
    return VideoClip(
        frames=frames,
        title="Ball Falling and Bouncing",
        domain="physics",
    )


def create_collision_video() -> VideoClip:
    """Create a synthetic physics video: two balls colliding."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    # Ball positions over time
    red_x = [10, 15, 20, 25, 28, 25, 20, 15, 10]
    blue_x = [46, 41, 36, 31, 28, 31, 36, 41, 46]
    
    for i in range(len(red_x)):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Draw red ball
        red_ball = create_stimulus_on_canvas(
            shape='circle',
            color='red',
            size='small',
            canvas_size=56,
            center=(red_x[i], 28),
        )
        
        # Draw blue ball
        blue_ball = create_stimulus_on_canvas(
            shape='circle',
            color='blue',
            size='small',
            canvas_size=56,
            center=(blue_x[i], 28),
        )
        
        # Composite
        is_red = np.any(red_ball < 0.95, axis=-1)
        is_blue = np.any(blue_ball < 0.95, axis=-1)
        canvas[is_red] = red_ball[is_red]
        canvas[is_blue] = blue_ball[is_blue]
        
        # Description based on phase
        if i < 4:
            desc = "two balls move toward each other"
        elif i == 4:
            desc = "balls collide and transfer momentum"
        else:
            desc = "balls bounce apart after collision"
        
        frames.append(VideoFrame(
            image=canvas,
            timestamp=i * 0.1,
            description=desc,
        ))
    
    return VideoClip(
        frames=frames,
        title="Two Ball Collision",
        domain="physics",
    )


def create_growth_video() -> VideoClip:
    """Create a synthetic biology video: object growing."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    # Circle growing
    sizes = ['small'] * 3 + ['medium'] * 3 + ['large'] * 3
    
    for i, size in enumerate(sizes):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Map size to actual radius
        if size == 'small':
            radius = 5
        elif size == 'medium':
            radius = 8
        else:
            radius = 12
        
        # Draw growing circle
        y, x = np.ogrid[:56, :56]
        dist = np.sqrt((x - 28)**2 + (y - 28)**2)
        mask = dist <= radius
        
        canvas[mask] = [0.2, 0.8, 0.2]  # Green
        
        # Description
        if i < 3:
            desc = "organism is small"
        elif i < 6:
            desc = "organism grows larger"
        else:
            desc = "organism reaches full size"
        
        frames.append(VideoFrame(
            image=canvas,
            timestamp=i * 0.5,
            description=desc,
        ))
    
    return VideoClip(
        frames=frames,
        title="Organism Growth",
        domain="biology",
    )


def create_rotation_video() -> VideoClip:
    """Create a synthetic video: object rotating."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    # Triangle rotating (simulated by position changes)
    positions = [
        (28, 20),  # Top
        (35, 25),  # Top-right
        (38, 32),  # Right
        (35, 39),  # Bottom-right
        (28, 42),  # Bottom
        (21, 39),  # Bottom-left
        (18, 32),  # Left
        (21, 25),  # Top-left
    ]
    
    for i, (x, y) in enumerate(positions):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Draw triangle at different positions to simulate rotation
        tri = create_stimulus_on_canvas(
            shape='triangle',
            color='blue',
            size='small',
            canvas_size=56,
            center=(x, y),
        )
        
        is_tri = np.any(tri < 0.95, axis=-1)
        canvas[is_tri] = tri[is_tri]
        
        desc = "object rotates in a circle"
        
        frames.append(VideoFrame(
            image=canvas,
            timestamp=i * 0.1,
            description=desc,
        ))
    
    return VideoClip(
        frames=frames,
        title="Object Rotation",
        domain="physics",
    )


def run_video_learning_experiment():
    """Run the video learning experiment."""
    print("=" * 70)
    print("PHASE 6: VIDEO UNDERSTANDING")
    print("Learning domain knowledge from video sequences")
    print("=" * 70)
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    
    # Load brain
    brain = None
    for results_dir in ['../abstraction_results', '../language_results']:
        if Path(results_dir).exists():
            model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]
            if model_files:
                model_files.sort(reverse=True)
                model_path = os.path.join(results_dir, model_files[0])
                print(f"Loading brain from: {model_path}")
                
                brain = AbstractBrain(
                    feature_dim=64,
                    n_concepts=200,
                    visual_input_size=56,
                )
                
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
                brain.visual.load_state_dict(checkpoint['visual_state'])
                print("Loaded brain weights")
                break
    
    if brain is None:
        print("No pre-trained brain found. Creating new brain...")
        brain = AbstractBrain(feature_dim=64, n_concepts=200, visual_input_size=56)
    
    # Create video learner
    learner = VideoLearner(brain)
    
    # Create synthetic videos
    print("\n" + "=" * 60)
    print("Creating synthetic educational videos...")
    print("=" * 60)
    
    videos = [
        create_physics_video(),
        create_collision_video(),
        create_growth_video(),
        create_rotation_video(),
    ]
    
    print(f"Created {len(videos)} videos")
    
    # Watch and learn from videos
    print("\n" + "=" * 60)
    print("Learning from videos...")
    print("=" * 60)
    
    all_stats = []
    for video in videos:
        stats = learner.watch_video(video, learning_threshold=0.1)
        all_stats.append(stats)
        print(f"    Learned {stats['patterns_learned']} patterns, "
              f"mean error: {stats['mean_error']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VIDEO LEARNING SUMMARY")
    print("=" * 60)
    
    print(f"\n  Frames processed: {learner.frames_processed}")
    print(f"  Patterns learned: {learner.patterns_learned}")
    
    print("\n  Domain knowledge:")
    for domain, patterns in learner.domain_knowledge.items():
        if patterns:
            print(f"    {domain}: {len(patterns)} patterns")
            for p in patterns[:3]:
                if p.description:
                    print(f"      - {p.description}")
    
    # Test explanation
    print("\n" + "=" * 60)
    print("TESTING EXPLANATIONS")
    print("=" * 60)
    
    # Create test transitions
    test_video = create_physics_video()
    
    print("\n  Testing on falling ball video:")
    for i in range(len(test_video.frames) - 1):
        before = learner.process_frame(test_video.frames[i])
        after = learner.process_frame(test_video.frames[i + 1])
        
        explanation = learner.explain_transition(before, after)
        print(f"    Frame {i}→{i+1}: {explanation}")
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} seconds")
    print(f"  Videos watched: {len(videos)}")
    print(f"  Frames processed: {learner.frames_processed}")
    print(f"  Patterns learned: {learner.patterns_learned}")
    
    # Save results
    results_dir = Path('../video_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'frames_processed': learner.frames_processed,
        'patterns_learned': learner.patterns_learned,
        'domain_counts': {d: len(p) for d, p in learner.domain_knowledge.items()},
        'video_stats': all_stats,
        'duration_seconds': duration,
    }
    
    results_path = results_dir / f'video_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    return learner, results


if __name__ == "__main__":
    learner, results = run_video_learning_experiment()
