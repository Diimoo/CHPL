#!/usr/bin/env python3
"""
Phase 7: Real-World Observation

Learn from continuous observation of real-world scenes.

Capabilities:
1. Detect events (motion, appearance, disappearance)
2. Cluster similar events into patterns
3. Build model of "normal" behavior
4. Detect anomalies (unusual events)
5. Learn temporal routines

This enables unsupervised learning from webcams/CCTV streams.
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
from dataclasses import dataclass, field
from collections import deque

from hierarchical_atl import AbstractBrain, DEVICE


@dataclass
class Event:
    """A detected event in the observation stream."""
    timestamp: float
    event_type: str  # appeared, disappeared, moved, changed
    activation_before: torch.Tensor
    activation_after: torch.Tensor
    change_magnitude: float
    description: str = ""


@dataclass
class EventCluster:
    """A cluster of similar events."""
    prototype: torch.Tensor
    examples: List[Event] = field(default_factory=list)
    count: int = 0
    label: str = ""


@dataclass
class TemporalPattern:
    """A temporal pattern (routine)."""
    event_sequence: List[str]  # sequence of event types
    times: List[float]  # typical times this occurs
    frequency: int = 0
    description: str = ""


class RealWorldObserver:
    """
    Learn from continuous real-world observation.
    
    Detects events, clusters them, learns routines, and flags anomalies.
    """
    
    def __init__(self, brain: AbstractBrain):
        self.brain = brain
        self.n_concepts = brain.atl.n_concepts
        
        # Event detection
        self.frame_buffer: deque = deque(maxlen=100)
        self.event_history: List[Event] = []
        
        # Event clustering
        self.event_clusters: List[EventCluster] = []
        self.cluster_threshold = 0.7  # Similarity threshold for clustering
        
        # Temporal patterns
        self.temporal_patterns: List[TemporalPattern] = []
        self.recent_events: deque = deque(maxlen=10)
        
        # Anomaly detection
        self.anomaly_threshold = 0.3  # Below this similarity = anomaly
        self.anomalies: List[Event] = []
        
        # Statistics
        self.frames_observed = 0
        self.events_detected = 0
        self.anomalies_detected = 0
    
    def process_frame(self, image: np.ndarray, timestamp: float) -> torch.Tensor:
        """Process a single frame and return activation."""
        image_t = torch.from_numpy(image).float()
        
        if image_t.dim() == 2:
            image_t = image_t.unsqueeze(-1).repeat(1, 1, 3)
        
        if image_t.max() > 1.0:
            image_t = image_t / 255.0
        
        image_t = image_t.permute(2, 0, 1).to(DEVICE)
        
        # Resize if needed
        if image_t.shape[1] != 56 or image_t.shape[2] != 56:
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(56, 56),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        activation = self.brain.get_scene_activation(image_t)
        
        # Store in buffer
        self.frame_buffer.append({
            'activation': activation,
            'timestamp': timestamp,
        })
        
        self.frames_observed += 1
        return activation
    
    def detect_event(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        timestamp: float,
    ) -> Optional[Event]:
        """Detect if an event occurred between frames."""
        # Compute change
        change = current - previous
        change_magnitude = torch.norm(change).item()
        
        # Threshold for event detection (lowered for sensitivity)
        if change_magnitude < 0.01:
            return None  # No significant event
        
        # Classify event type based on activation patterns
        current_sum = current.sum().item()
        previous_sum = previous.sum().item()
        
        if current_sum > previous_sum * 1.2:
            event_type = "appeared"
        elif current_sum < previous_sum * 0.8:
            event_type = "disappeared"
        else:
            # Check for movement vs. change
            similarity = F.cosine_similarity(
                current.unsqueeze(0),
                previous.unsqueeze(0)
            ).item()
            
            if similarity > 0.7:
                event_type = "moved"
            else:
                event_type = "changed"
        
        event = Event(
            timestamp=timestamp,
            event_type=event_type,
            activation_before=previous.detach().clone(),
            activation_after=current.detach().clone(),
            change_magnitude=change_magnitude,
        )
        
        self.events_detected += 1
        self.event_history.append(event)
        self.recent_events.append(event)
        
        return event
    
    def cluster_event(self, event: Event) -> EventCluster:
        """Assign event to a cluster or create new cluster."""
        change = event.activation_after - event.activation_before
        
        # Find best matching cluster
        best_cluster = None
        best_similarity = self.cluster_threshold
        
        for cluster in self.event_clusters:
            similarity = F.cosine_similarity(
                change.unsqueeze(0),
                cluster.prototype.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        if best_cluster is not None:
            # Add to existing cluster
            best_cluster.examples.append(event)
            best_cluster.count += 1
            
            # Update prototype (running average)
            alpha = 0.1
            best_cluster.prototype = (
                (1 - alpha) * best_cluster.prototype + 
                alpha * change
            )
            
            return best_cluster
        else:
            # Create new cluster
            new_cluster = EventCluster(
                prototype=change.detach().clone(),
                examples=[event],
                count=1,
                label=f"cluster_{len(self.event_clusters)}",
            )
            self.event_clusters.append(new_cluster)
            return new_cluster
    
    def detect_anomaly(self, event: Event) -> bool:
        """Check if event is anomalous."""
        if len(self.event_clusters) == 0:
            return False  # No baseline yet
        
        change = event.activation_after - event.activation_before
        
        # Find closest cluster
        best_similarity = -1
        for cluster in self.event_clusters:
            similarity = F.cosine_similarity(
                change.unsqueeze(0),
                cluster.prototype.unsqueeze(0)
            ).item()
            best_similarity = max(best_similarity, similarity)
        
        if best_similarity < self.anomaly_threshold:
            self.anomalies.append(event)
            self.anomalies_detected += 1
            return True
        
        return False
    
    def observe_stream(
        self,
        frames: List[Tuple[np.ndarray, float]],
        verbose: bool = True,
    ) -> Dict:
        """
        Observe a stream of frames.
        
        Args:
            frames: List of (image, timestamp) tuples
            verbose: Print progress
        
        Returns:
            Statistics about observation
        """
        if verbose:
            print(f"  Observing {len(frames)} frames...")
        
        stats = {
            'frames': len(frames),
            'events': 0,
            'anomalies': 0,
            'clusters': 0,
        }
        
        previous_activation = None
        
        for i, (image, timestamp) in enumerate(frames):
            current_activation = self.process_frame(image, timestamp)
            
            if previous_activation is not None:
                # Detect event
                event = self.detect_event(
                    current_activation,
                    previous_activation,
                    timestamp,
                )
                
                if event is not None:
                    stats['events'] += 1
                    
                    # Cluster event
                    self.cluster_event(event)
                    
                    # Check for anomaly
                    if self.detect_anomaly(event):
                        stats['anomalies'] += 1
                        if verbose:
                            print(f"    ! Anomaly at t={timestamp:.1f}: {event.event_type}")
            
            previous_activation = current_activation
        
        stats['clusters'] = len(self.event_clusters)
        
        if verbose:
            print(f"    Events: {stats['events']}, Clusters: {stats['clusters']}, "
                  f"Anomalies: {stats['anomalies']}")
        
        return stats
    
    def get_baseline_summary(self) -> Dict:
        """Summarize learned baseline behavior."""
        summary = {
            'total_frames': self.frames_observed,
            'total_events': self.events_detected,
            'event_clusters': len(self.event_clusters),
            'anomalies': self.anomalies_detected,
            'cluster_sizes': [c.count for c in self.event_clusters],
        }
        
        # Event type distribution
        event_types = {}
        for event in self.event_history:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        summary['event_types'] = event_types
        
        return summary


def create_traffic_simulation() -> List[Tuple[np.ndarray, float]]:
    """Create simulated traffic camera footage."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    # Simulate cars passing
    # Car 1: moves left to right
    # Car 2: moves right to left
    
    for t in range(50):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Road (gray stripe)
        canvas[24:32, :] = [0.5, 0.5, 0.5]
        
        # Car 1 (red square, moves left to right)
        if 0 <= t < 20:
            car1_x = 5 + t * 2
            if car1_x < 50:
                car1 = create_stimulus_on_canvas(
                    shape='square', color='red', size='small',
                    canvas_size=56, center=(car1_x, 28),
                )
                is_car = np.any(car1 < 0.95, axis=-1)
                canvas[is_car] = car1[is_car]
        
        # Car 2 (blue square, moves right to left)
        if 15 <= t < 40:
            car2_x = 50 - (t - 15) * 2
            if car2_x > 5:
                car2 = create_stimulus_on_canvas(
                    shape='square', color='blue', size='small',
                    canvas_size=56, center=(car2_x, 28),
                )
                is_car = np.any(car2 < 0.95, axis=-1)
                canvas[is_car] = car2[is_car]
        
        # Anomaly: pedestrian crossing (rare event)
        if t == 35:
            ped = create_stimulus_on_canvas(
                shape='circle', color='green', size='small',
                canvas_size=56, center=(28, 20),
            )
            is_ped = np.any(ped < 0.95, axis=-1)
            canvas[is_ped] = ped[is_ped]
        
        frames.append((canvas, t * 0.5))  # 0.5 second intervals
    
    return frames


def create_nature_simulation() -> List[Tuple[np.ndarray, float]]:
    """Create simulated nature camera footage."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    for t in range(40):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Sky (light blue top)
        canvas[:20, :] = [0.6, 0.8, 1.0]
        
        # Ground (green bottom)
        canvas[40:, :] = [0.3, 0.6, 0.3]
        
        # Sun (yellow circle, stays mostly still)
        sun = create_stimulus_on_canvas(
            shape='circle', color='yellow', size='small',
            canvas_size=56, center=(45, 10),
        )
        is_sun = np.any(sun < 0.95, axis=-1)
        canvas[is_sun] = sun[is_sun]
        
        # Bird occasionally flies by
        if 10 <= t < 20:
            bird_x = 5 + (t - 10) * 4
            bird = create_stimulus_on_canvas(
                shape='triangle', color='blue', size='small',
                canvas_size=56, center=(bird_x, 15),
            )
            is_bird = np.any(bird < 0.95, axis=-1)
            canvas[is_bird] = bird[is_bird]
        
        # Rare: animal appears (anomaly)
        if t == 30:
            animal = create_stimulus_on_canvas(
                shape='circle', color='orange', size='small',
                canvas_size=56, center=(28, 45),
            )
            is_animal = np.any(animal < 0.95, axis=-1)
            canvas[is_animal] = animal[is_animal]
        
        frames.append((canvas, t * 1.0))  # 1 second intervals
    
    return frames


def create_lobby_simulation() -> List[Tuple[np.ndarray, float]]:
    """Create simulated building lobby footage."""
    from synthetic_environment import create_stimulus_on_canvas
    
    frames = []
    
    for t in range(60):
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Floor
        canvas[45:, :] = [0.7, 0.6, 0.5]
        
        # Door (right side)
        canvas[10:45, 48:56] = [0.4, 0.3, 0.2]
        
        # Person 1 enters, walks across, exits
        if 5 <= t < 25:
            person_x = 50 - (t - 5) * 2
            if person_x > 5:
                person = create_stimulus_on_canvas(
                    shape='circle', color='blue', size='small',
                    canvas_size=56, center=(person_x, 35),
                )
                is_person = np.any(person < 0.95, axis=-1)
                canvas[is_person] = person[is_person]
        
        # Person 2 enters later
        if 35 <= t < 55:
            person_x = 50 - (t - 35) * 2
            if person_x > 5:
                person = create_stimulus_on_canvas(
                    shape='circle', color='red', size='small',
                    canvas_size=56, center=(person_x, 35),
                )
                is_person = np.any(person < 0.95, axis=-1)
                canvas[is_person] = person[is_person]
        
        # Anomaly: unusual object left behind
        if 45 <= t < 55:
            obj = create_stimulus_on_canvas(
                shape='square', color='green', size='small',
                canvas_size=56, center=(28, 42),
            )
            is_obj = np.any(obj < 0.95, axis=-1)
            canvas[is_obj] = obj[is_obj]
        
        frames.append((canvas, t * 0.5))
    
    return frames


def run_observation_experiment():
    """Run the real-world observation experiment."""
    print("=" * 70)
    print("PHASE 7: REAL-WORLD OBSERVATION")
    print("Learning from continuous observation streams")
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
    
    # Create observer
    observer = RealWorldObserver(brain)
    
    # Create simulated streams
    print("\n" + "=" * 60)
    print("Creating simulated observation streams...")
    print("=" * 60)
    
    streams = [
        ("Traffic Camera", create_traffic_simulation()),
        ("Nature Camera", create_nature_simulation()),
        ("Lobby Camera", create_lobby_simulation()),
    ]
    
    # Observe streams
    print("\n" + "=" * 60)
    print("Observing streams...")
    print("=" * 60)
    
    all_stats = []
    for name, frames in streams:
        print(f"\n  {name}:")
        stats = observer.observe_stream(frames)
        stats['name'] = name
        all_stats.append(stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("OBSERVATION SUMMARY")
    print("=" * 60)
    
    baseline = observer.get_baseline_summary()
    
    print(f"\n  Total frames observed: {baseline['total_frames']}")
    print(f"  Total events detected: {baseline['total_events']}")
    print(f"  Event clusters formed: {baseline['event_clusters']}")
    print(f"  Anomalies detected: {baseline['anomalies']}")
    
    print("\n  Event type distribution:")
    for event_type, count in baseline['event_types'].items():
        print(f"    {event_type}: {count}")
    
    print("\n  Cluster sizes:")
    for i, size in enumerate(baseline['cluster_sizes']):
        print(f"    Cluster {i}: {size} events")
    
    # Final stats
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} seconds")
    print(f"  Streams observed: {len(streams)}")
    print(f"  Learned to detect {baseline['event_clusters']} distinct event patterns")
    print(f"  Identified {baseline['anomalies']} anomalous events")
    
    # Save results
    results_dir = Path('../observation_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'baseline': {k: v for k, v in baseline.items() if k != 'cluster_sizes'},
        'cluster_sizes': baseline['cluster_sizes'],
        'stream_stats': all_stats,
        'duration_seconds': duration,
    }
    
    results_path = results_dir / f'observation_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    return observer, results


if __name__ == "__main__":
    observer, results = run_observation_experiment()
