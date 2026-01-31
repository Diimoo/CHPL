#!/usr/bin/env python3
"""
Phase 8: Self-Directed Learning

Curiosity-driven exploration and goal-setting.

Capabilities:
1. Compute curiosity/uncertainty for states
2. Select observations that maximize learning
3. Set and decompose learning goals
4. Track knowledge gaps
5. Meta-learning: learn how to learn better

This is the final phase toward adult-level cognition.
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
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
import heapq

from hierarchical_atl import AbstractBrain, DEVICE


@dataclass
class LearningGoal:
    """A learning goal to pursue."""
    description: str
    sub_goals: List[str] = field(default_factory=list)
    progress: float = 0.0
    completed: bool = False
    priority: float = 1.0


@dataclass
class KnowledgeGap:
    """An identified gap in knowledge."""
    domain: str
    description: str
    uncertainty: float
    examples_needed: int = 5


@dataclass
class LearningExperience:
    """Record of a learning experience."""
    observation: torch.Tensor
    prediction_error: float
    knowledge_gained: float
    timestamp: float


class CuriousLearner:
    """
    Self-directed learning agent with curiosity.
    
    Uses prediction error as intrinsic motivation.
    Actively seeks high-uncertainty states for learning.
    """
    
    def __init__(self, brain: AbstractBrain):
        self.brain = brain
        self.n_concepts = brain.atl.n_concepts
        
        # Curiosity system
        self.uncertainty_threshold = 0.3
        self.curiosity_buffer: List[Tuple[float, torch.Tensor]] = []  # (curiosity, state)
        
        # Knowledge tracking
        self.known_states: List[torch.Tensor] = []
        self.knowledge_gaps: List[KnowledgeGap] = []
        self.learning_history: List[LearningExperience] = []
        
        # Goal system
        self.current_goals: List[LearningGoal] = []
        self.completed_goals: List[LearningGoal] = []
        
        # Meta-learning
        self.learning_strategies: Dict[str, Dict] = {
            'prediction': {'success_rate': 0.5, 'uses': 0},
            'analogy': {'success_rate': 0.5, 'uses': 0},
            'definition': {'success_rate': 0.5, 'uses': 0},
            'observation': {'success_rate': 0.5, 'uses': 0},
        }
        
        # Statistics
        self.total_observations = 0
        self.knowledge_gained = 0.0
        self.goals_completed = 0
    
    def compute_curiosity(self, state: torch.Tensor) -> float:
        """
        Compute curiosity/uncertainty for a state.
        
        High curiosity = high entropy OR far from known states.
        """
        state = state.to(DEVICE)
        
        # Component 1: Activation entropy (uncertainty in representation)
        entropy = -torch.sum(state * torch.log(state + 1e-10)).item()
        normalized_entropy = entropy / np.log(self.n_concepts)  # 0 to 1
        
        # Component 2: Distance from known states (novelty)
        if len(self.known_states) == 0:
            novelty = 1.0  # Everything is novel at first
        else:
            similarities = []
            for known in self.known_states[-100:]:  # Check recent states
                sim = F.cosine_similarity(
                    state.unsqueeze(0),
                    known.unsqueeze(0)
                ).item()
                similarities.append(sim)
            
            max_similarity = max(similarities)
            novelty = 1.0 - max_similarity
        
        # Combine: curiosity = novelty * uncertainty
        curiosity = 0.5 * normalized_entropy + 0.5 * novelty
        
        return curiosity
    
    def should_explore(self, state: torch.Tensor) -> bool:
        """Decide whether to explore this state further."""
        curiosity = self.compute_curiosity(state)
        return curiosity > self.uncertainty_threshold
    
    def select_next_observation(
        self,
        candidates: List[torch.Tensor],
    ) -> Tuple[int, float]:
        """
        Select which observation to make next.
        
        Returns index of most curious candidate and its curiosity score.
        """
        if len(candidates) == 0:
            return -1, 0.0
        
        curiosities = []
        for i, candidate in enumerate(candidates):
            curiosity = self.compute_curiosity(candidate)
            curiosities.append((curiosity, i))
        
        # Select highest curiosity
        curiosities.sort(reverse=True)
        best_curiosity, best_idx = curiosities[0]
        
        return best_idx, best_curiosity
    
    def learn_from_observation(
        self,
        observation: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Learn from an observation.
        
        Returns knowledge gained (based on prediction error).
        """
        observation = observation.to(DEVICE)
        self.total_observations += 1
        
        # Compute prediction error if we have a prediction
        if prediction is not None:
            prediction = prediction.to(DEVICE)
            error = 1.0 - F.cosine_similarity(
                observation.unsqueeze(0),
                prediction.unsqueeze(0)
            ).item()
        else:
            # Use novelty as proxy for learning
            error = self.compute_curiosity(observation)
        
        # Knowledge gained is proportional to error (surprising = informative)
        knowledge = min(error, 1.0)
        self.knowledge_gained += knowledge
        
        # Add to known states
        self.known_states.append(observation.detach().clone())
        
        # Keep buffer manageable
        if len(self.known_states) > 1000:
            self.known_states = self.known_states[-500:]
        
        # Record experience
        experience = LearningExperience(
            observation=observation.detach().clone(),
            prediction_error=error,
            knowledge_gained=knowledge,
            timestamp=datetime.now().timestamp(),
        )
        self.learning_history.append(experience)
        
        return knowledge
    
    def set_learning_goal(self, description: str) -> LearningGoal:
        """
        Set a high-level learning goal.
        
        Decomposes goal into sub-goals.
        """
        # Simple goal decomposition based on keywords
        sub_goals = []
        
        description_lower = description.lower()
        
        if 'physics' in description_lower:
            sub_goals = [
                "observe objects falling",
                "observe objects colliding",
                "observe objects bouncing",
                "learn gravity patterns",
                "learn momentum transfer",
            ]
        elif 'color' in description_lower:
            sub_goals = [
                "identify primary colors",
                "learn color names",
                "understand color mixing",
            ]
        elif 'shape' in description_lower:
            sub_goals = [
                "identify basic shapes",
                "learn shape names",
                "understand shape properties",
            ]
        elif 'language' in description_lower:
            sub_goals = [
                "learn word meanings",
                "understand word relationships",
                "practice descriptions",
            ]
        else:
            sub_goals = [
                f"explore {description}",
                f"identify patterns in {description}",
                f"consolidate knowledge of {description}",
            ]
        
        goal = LearningGoal(
            description=description,
            sub_goals=sub_goals,
            progress=0.0,
            priority=1.0,
        )
        
        self.current_goals.append(goal)
        return goal
    
    def update_goal_progress(self, goal: LearningGoal, amount: float):
        """Update progress on a goal."""
        goal.progress = min(goal.progress + amount, 1.0)
        
        if goal.progress >= 1.0:
            goal.completed = True
            self.current_goals.remove(goal)
            self.completed_goals.append(goal)
            self.goals_completed += 1
    
    def identify_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Identify gaps in current knowledge."""
        gaps = []
        
        # Analyze learning history for high-error domains
        if len(self.learning_history) < 10:
            return gaps
        
        recent = self.learning_history[-50:]
        avg_error = np.mean([exp.prediction_error for exp in recent])
        
        if avg_error > 0.5:
            gaps.append(KnowledgeGap(
                domain="general",
                description="High prediction errors suggest knowledge gaps",
                uncertainty=avg_error,
                examples_needed=20,
            ))
        
        self.knowledge_gaps = gaps
        return gaps
    
    def select_learning_strategy(self, task_type: str) -> str:
        """
        Meta-learning: select best strategy for a task type.
        
        Uses success rates from past experience.
        """
        # Get strategies sorted by success rate
        sorted_strategies = sorted(
            self.learning_strategies.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True,
        )
        
        # Epsilon-greedy: usually pick best, sometimes explore
        if np.random.random() < 0.2:  # Explore
            strategy = np.random.choice(list(self.learning_strategies.keys()))
        else:  # Exploit
            strategy = sorted_strategies[0][0]
        
        # Update usage count
        self.learning_strategies[strategy]['uses'] += 1
        
        return strategy
    
    def update_strategy_success(self, strategy: str, success: bool):
        """Update success rate for a learning strategy."""
        if strategy not in self.learning_strategies:
            return
        
        current = self.learning_strategies[strategy]
        alpha = 0.1  # Learning rate for meta-learning
        
        if success:
            current['success_rate'] = (1 - alpha) * current['success_rate'] + alpha * 1.0
        else:
            current['success_rate'] = (1 - alpha) * current['success_rate'] + alpha * 0.0
    
    def get_learning_summary(self) -> Dict:
        """Summarize learning progress."""
        return {
            'total_observations': self.total_observations,
            'knowledge_gained': self.knowledge_gained,
            'known_states': len(self.known_states),
            'goals_completed': self.goals_completed,
            'current_goals': len(self.current_goals),
            'knowledge_gaps': len(self.knowledge_gaps),
            'strategies': {
                k: v['success_rate'] 
                for k, v in self.learning_strategies.items()
            },
        }


def create_exploration_environment():
    """Create an environment to explore with various stimuli."""
    from synthetic_environment import create_stimulus_on_canvas
    
    stimuli = []
    
    # Various shapes and colors
    shapes = ['circle', 'square', 'triangle', 'star']
    colors = ['red', 'blue', 'green', 'yellow']
    
    for shape in shapes:
        for color in colors:
            img = create_stimulus_on_canvas(
                shape=shape,
                color=color,
                size='small',
                canvas_size=56,
                center=(28, 28),
            )
            stimuli.append({
                'image': img,
                'description': f"{color} {shape}",
            })
    
    # Some combinations
    for i in range(10):
        color1, color2 = np.random.choice(colors, 2, replace=False)
        shape1, shape2 = np.random.choice(shapes, 2, replace=False)
        
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        obj1 = create_stimulus_on_canvas(shape1, color1, 'small', 56, (20, 28))
        obj2 = create_stimulus_on_canvas(shape2, color2, 'small', 56, (38, 28))
        
        is_obj1 = np.any(obj1 < 0.95, axis=-1)
        is_obj2 = np.any(obj2 < 0.95, axis=-1)
        canvas[is_obj1] = obj1[is_obj1]
        canvas[is_obj2] = obj2[is_obj2]
        
        stimuli.append({
            'image': canvas,
            'description': f"{color1} {shape1} and {color2} {shape2}",
        })
    
    return stimuli


def run_curious_learning_experiment():
    """Run the self-directed learning experiment."""
    print("=" * 70)
    print("PHASE 8: SELF-DIRECTED LEARNING")
    print("Curiosity-driven exploration and goal-setting")
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
    
    # Create curious learner
    learner = CuriousLearner(brain)
    
    # Create exploration environment
    print("\n" + "=" * 60)
    print("Creating exploration environment...")
    print("=" * 60)
    
    stimuli = create_exploration_environment()
    print(f"Created {len(stimuli)} stimuli to explore")
    
    # Set learning goals
    print("\n" + "=" * 60)
    print("Setting learning goals...")
    print("=" * 60)
    
    goals = [
        learner.set_learning_goal("understand shapes"),
        learner.set_learning_goal("understand colors"),
        learner.set_learning_goal("understand physics"),
    ]
    
    for goal in goals:
        print(f"  Goal: {goal.description}")
        for sub in goal.sub_goals[:2]:
            print(f"    - {sub}")
    
    # Curiosity-driven exploration
    print("\n" + "=" * 60)
    print("Curiosity-driven exploration...")
    print("=" * 60)
    
    exploration_rounds = 3
    stimuli_per_round = 10
    
    for round_num in range(exploration_rounds):
        print(f"\n  Round {round_num + 1}/{exploration_rounds}:")
        
        # Get candidate stimuli
        candidates = np.random.choice(len(stimuli), min(stimuli_per_round, len(stimuli)), replace=False)
        
        candidate_activations = []
        for idx in candidates:
            img = stimuli[idx]['image']
            img_t = torch.from_numpy(img).float().permute(2, 0, 1).to(DEVICE)
            activation = brain.get_scene_activation(img_t)
            candidate_activations.append(activation)
        
        # Select based on curiosity
        best_idx, curiosity = learner.select_next_observation(candidate_activations)
        
        if best_idx >= 0:
            selected = stimuli[candidates[best_idx]]
            knowledge = learner.learn_from_observation(candidate_activations[best_idx])
            
            print(f"    Selected: {selected['description']}")
            print(f"    Curiosity: {curiosity:.3f}, Knowledge gained: {knowledge:.3f}")
            
            # Update goal progress
            for goal in learner.current_goals:
                if any(kw in selected['description'].lower() for kw in goal.description.lower().split()):
                    learner.update_goal_progress(goal, 0.1)
        
        # Learn from all candidates
        for activation in candidate_activations:
            learner.learn_from_observation(activation)
    
    # Test curiosity selection
    print("\n" + "=" * 60)
    print("Testing curiosity-based selection...")
    print("=" * 60)
    
    # Create novel vs. familiar stimuli
    from synthetic_environment import create_stimulus_on_canvas
    
    # Familiar: red circle (should have low curiosity)
    familiar = create_stimulus_on_canvas('circle', 'red', 'small', 56, (28, 28))
    familiar_t = torch.from_numpy(familiar).float().permute(2, 0, 1).to(DEVICE)
    familiar_act = brain.get_scene_activation(familiar_t)
    familiar_curiosity = learner.compute_curiosity(familiar_act)
    
    # Novel: unusual combination (should have higher curiosity)
    novel = np.ones((56, 56, 3), dtype=np.float32)
    novel[10:46, 10:46] = [0.5, 0.0, 0.5]  # Purple square (unusual)
    novel_t = torch.from_numpy(novel).float().permute(2, 0, 1).to(DEVICE)
    novel_act = brain.get_scene_activation(novel_t)
    novel_curiosity = learner.compute_curiosity(novel_act)
    
    print(f"  Familiar stimulus (red circle) curiosity: {familiar_curiosity:.3f}")
    print(f"  Novel stimulus (purple square) curiosity: {novel_curiosity:.3f}")
    print(f"  Novel > Familiar: {novel_curiosity > familiar_curiosity}")
    
    # Summary
    print("\n" + "=" * 60)
    print("LEARNING SUMMARY")
    print("=" * 60)
    
    summary = learner.get_learning_summary()
    
    print(f"\n  Total observations: {summary['total_observations']}")
    print(f"  Knowledge gained: {summary['knowledge_gained']:.2f}")
    print(f"  Known states: {summary['known_states']}")
    print(f"  Goals completed: {summary['goals_completed']}")
    
    print("\n  Learning strategies:")
    for strategy, rate in summary['strategies'].items():
        print(f"    {strategy}: {rate:.2f} success rate")
    
    # Identify knowledge gaps
    gaps = learner.identify_knowledge_gaps()
    if gaps:
        print("\n  Knowledge gaps identified:")
        for gap in gaps:
            print(f"    - {gap.description} (uncertainty: {gap.uncertainty:.2f})")
    
    # Final stats
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} seconds")
    print(f"  Observations made: {summary['total_observations']}")
    print(f"  Knowledge gained: {summary['knowledge_gained']:.2f}")
    
    curiosity_works = novel_curiosity > familiar_curiosity
    print(f"\n  Curiosity-based selection: {'WORKING' if curiosity_works else 'NEEDS WORK'}")
    
    # Save results
    results_dir = Path('../curiosity_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'summary': summary,
        'curiosity_test': {
            'familiar_curiosity': float(familiar_curiosity),
            'novel_curiosity': float(novel_curiosity),
            'working': bool(curiosity_works),
        },
        'duration_seconds': float(duration),
    }
    
    results_path = results_dir / f'curiosity_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    return learner, results


if __name__ == "__main__":
    learner, results = run_curious_learning_experiment()
