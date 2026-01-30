#!/usr/bin/env python3
"""
Phase 2: Causal Reasoning & Planning

CausalATL extends PredictiveATL with:
- Causal inference: infer what caused observed changes
- Physical constraints: learn what interactions are possible
- Goal-directed planning: plan actions to reach goals

This builds on Phase 1's temporal prediction capabilities.
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from predictive_atl import PredictiveATL, DEVICE


class InteractionType(Enum):
    """Types of physical interactions between objects."""
    PUSH = 'push'
    BLOCK = 'block'
    PASS_THROUGH = 'pass_through'
    NONE = 'none'


class ActionType(Enum):
    """Available actions for planning."""
    MOVE_RIGHT = 'move_right'
    MOVE_LEFT = 'move_left'
    MOVE_UP = 'move_up'
    MOVE_DOWN = 'move_down'
    PUSH = 'push'
    STAY = 'stay'


@dataclass
class CausalEvent:
    """Represents a causal event with cause, effect, and mechanism."""
    cause_activation: torch.Tensor
    effect_activation: torch.Tensor
    mechanism: InteractionType
    cause_label: str
    effect_label: str
    confidence: float = 1.0


class CausalATL(PredictiveATL):
    """
    Causal ATL: Temporal prediction + causal reasoning.
    
    Key capabilities:
    1. Infer causes from observed effects
    2. Learn physical constraints (what can interact with what)
    3. Plan action sequences to reach goals
    4. Counterfactual reasoning (what would happen if...)
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_concepts: int = 200,
        temperature: float = 0.2,
    ):
        super().__init__(feature_dim, n_concepts, temperature)
        
        # Causal inference network
        # Input: (state_before, state_after)
        # Output: causal graph edges
        self.causal_encoder = nn.Sequential(
            nn.Linear(n_concepts * 2, n_concepts * 2),
            nn.LayerNorm(n_concepts * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_concepts * 2, n_concepts),
            nn.ReLU(),
        ).to(DEVICE)
        
        # Causal graph decoder: predicts edge probabilities
        self.causal_decoder = nn.Sequential(
            nn.Linear(n_concepts, n_concepts * n_concepts // 4),
            nn.ReLU(),
            nn.Linear(n_concepts * n_concepts // 4, n_concepts),
            nn.Sigmoid(),
        ).to(DEVICE)
        
        # Interaction classifier: what type of interaction occurred?
        self.interaction_classifier = nn.Sequential(
            nn.Linear(n_concepts * 2, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, len(InteractionType)),
        ).to(DEVICE)
        
        # Physical constraint network
        # Learns: objects must touch to push, heavy > light, etc.
        self.constraint_net = nn.Sequential(
            nn.Linear(n_concepts * 2, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, len(InteractionType)),
            nn.Sigmoid(),
        ).to(DEVICE)
        
        # Goal-directed planner
        # Input: (current_state, goal_state)
        # Output: action probabilities
        self.planner = nn.Sequential(
            nn.Linear(n_concepts * 2, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, n_concepts // 2),
            nn.ReLU(),
            nn.Linear(n_concepts // 2, len(ActionType)),
        ).to(DEVICE)
        
        # Action encoder for planning
        self.action_encoder = nn.Embedding(len(ActionType), 16).to(DEVICE)
        
        # Value network for planning (predicts how close to goal)
        self.value_net = nn.Sequential(
            nn.Linear(n_concepts, n_concepts // 2),
            nn.ReLU(),
            nn.Linear(n_concepts // 2, 1),
            nn.Sigmoid(),
        ).to(DEVICE)
        
        self.action_types = list(ActionType)
        self.interaction_types = list(InteractionType)
    
    def infer_causality(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Infer what caused the change between states.
        
        Args:
            state_before: [n_concepts] activation before event
            state_after: [n_concepts] activation after event
        
        Returns:
            dict with:
                - causal_strength: [n_concepts] how much each concept contributed to change
                - interaction_probs: [n_interaction_types] probabilities
                - change_magnitude: scalar indicating overall change
        """
        # Ensure on device
        state_before = state_before.to(DEVICE)
        state_after = state_after.to(DEVICE)
        
        # Combine states
        combined = torch.cat([state_before, state_after])
        
        # Encode causal information
        causal_features = self.causal_encoder(combined)
        
        # Decode causal strengths per concept
        causal_strength = self.causal_decoder(causal_features)
        
        # Classify interaction type
        interaction_logits = self.interaction_classifier(combined)
        interaction_probs = F.softmax(interaction_logits, dim=-1)
        
        # Compute change magnitude
        change = state_after - state_before
        change_magnitude = torch.norm(change)
        
        return {
            'causal_strength': causal_strength,
            'interaction_probs': interaction_probs,
            'change_magnitude': change_magnitude,
            'change_vector': change,
        }
    
    def check_interaction_feasibility(
        self,
        obj1_activation: torch.Tensor,
        obj2_activation: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Check if interactions between two objects are physically possible.
        
        Args:
            obj1_activation: [n_concepts] first object
            obj2_activation: [n_concepts] second object
        
        Returns:
            dict mapping interaction type to feasibility probability
        """
        obj1_activation = obj1_activation.to(DEVICE)
        obj2_activation = obj2_activation.to(DEVICE)
        
        combined = torch.cat([obj1_activation, obj2_activation])
        feasibilities = self.constraint_net(combined)
        
        return {
            itype.value: feasibilities[i].item()
            for i, itype in enumerate(self.interaction_types)
        }
    
    def plan_to_goal(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        max_steps: int = 10,
        exploration_rate: float = 0.1,
    ) -> Tuple[List[ActionType], List[torch.Tensor], float]:
        """
        Plan action sequence to reach goal state.
        
        Uses:
        - Temporal prediction (from parent)
        - Value estimation (this class)
        - Goal-directed action selection
        
        Args:
            current_state: [n_concepts] current activation
            goal_state: [n_concepts] target activation
            max_steps: maximum planning horizon
            exploration_rate: probability of random action
        
        Returns:
            - action_sequence: list of actions
            - state_trajectory: list of predicted states
            - final_similarity: how close we got to goal
        """
        current_state = current_state.to(DEVICE)
        goal_state = goal_state.to(DEVICE)
        
        plan = []
        states = [current_state.clone()]
        state = current_state.clone()
        
        for step in range(max_steps):
            # Check if goal reached
            similarity = F.cosine_similarity(
                state.unsqueeze(0),
                goal_state.unsqueeze(0)
            ).item()
            
            if similarity > 0.9:
                break
            
            # Get action probabilities
            combined = torch.cat([state, goal_state])
            action_logits = self.planner(combined)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Select action
            if np.random.rand() < exploration_rate:
                action_idx = np.random.randint(len(self.action_types))
            else:
                action_idx = torch.argmax(action_probs).item()
            
            action = self.action_types[action_idx]
            
            # Convert action to velocity for prediction
            velocity = self._action_to_velocity(action)
            
            # Predict next state
            with torch.no_grad():
                next_state = self.predict_with_velocity(state, velocity)
            
            plan.append(action)
            states.append(next_state.clone())
            state = next_state
        
        # Final similarity to goal
        final_similarity = F.cosine_similarity(
            state.unsqueeze(0),
            goal_state.unsqueeze(0)
        ).item()
        
        return plan, states, final_similarity
    
    def _action_to_velocity(self, action: ActionType) -> torch.Tensor:
        """Convert action to velocity vector for prediction."""
        velocity_map = {
            ActionType.MOVE_RIGHT: (0.4, 0.0),
            ActionType.MOVE_LEFT: (-0.4, 0.0),
            ActionType.MOVE_UP: (0.0, -0.4),
            ActionType.MOVE_DOWN: (0.0, 0.4),
            ActionType.PUSH: (0.3, 0.0),
            ActionType.STAY: (0.0, 0.0),
        }
        vx, vy = velocity_map.get(action, (0.0, 0.0))
        return torch.tensor([vx, vy], device=DEVICE)
    
    def estimate_value(self, state: torch.Tensor, goal_state: torch.Tensor) -> float:
        """
        Estimate how valuable a state is relative to goal.
        
        Higher value = closer to goal.
        """
        state = state.to(DEVICE)
        goal_state = goal_state.to(DEVICE)
        
        # Simple heuristic: cosine similarity + learned value
        cos_sim = F.cosine_similarity(
            state.unsqueeze(0),
            goal_state.unsqueeze(0)
        ).item()
        
        learned_value = self.value_net(state).item()
        
        return 0.5 * cos_sim + 0.5 * learned_value
    
    def counterfactual_imagine(
        self,
        state: torch.Tensor,
        actual_action: ActionType,
        counterfactual_action: ActionType,
    ) -> Dict[str, torch.Tensor]:
        """
        Counterfactual imagination: what would have happened with different action?
        
        Args:
            state: current state
            actual_action: action that was taken
            counterfactual_action: alternative action to imagine
        
        Returns:
            dict with actual and counterfactual outcomes
        """
        state = state.to(DEVICE)
        
        actual_vel = self._action_to_velocity(actual_action)
        counter_vel = self._action_to_velocity(counterfactual_action)
        
        with torch.no_grad():
            actual_next = self.predict_with_velocity(state, actual_vel)
            counter_next = self.predict_with_velocity(state, counter_vel)
        
        difference = torch.norm(actual_next - counter_next).item()
        
        return {
            'actual_outcome': actual_next,
            'counterfactual_outcome': counter_next,
            'outcome_difference': difference,
        }
    
    def train_causal_step(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        interaction_label: int,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step for causal inference.
        
        Args:
            state_before: activation before event
            state_after: activation after event
            interaction_label: ground truth interaction type index
            optimizer: optimizer for causal networks
        
        Returns:
            dict with loss values
        """
        state_before = state_before.to(DEVICE)
        state_after = state_after.to(DEVICE)
        
        # Forward pass
        causal_result = self.infer_causality(state_before, state_after)
        
        # Interaction classification loss
        target = torch.tensor([interaction_label], device=DEVICE)
        interaction_loss = F.cross_entropy(
            causal_result['interaction_probs'].unsqueeze(0),
            target
        )
        
        # Causal strength should correlate with change magnitude
        # Higher change → higher causal strength for changed concepts
        change = torch.abs(state_after - state_before)
        change_norm = change / (change.max() + 1e-8)
        
        causal_correlation_loss = F.mse_loss(
            causal_result['causal_strength'],
            change_norm
        )
        
        total_loss = interaction_loss + 0.5 * causal_correlation_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'interaction_loss': interaction_loss.item(),
            'causal_correlation_loss': causal_correlation_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    def train_planning_step(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        optimal_action: int,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step for goal-directed planning.
        
        Args:
            current_state: current activation
            goal_state: target activation
            optimal_action: ground truth best action index
            optimizer: optimizer for planner
        
        Returns:
            dict with loss values
        """
        current_state = current_state.to(DEVICE)
        goal_state = goal_state.to(DEVICE)
        
        # Get action probabilities
        combined = torch.cat([current_state, goal_state])
        action_logits = self.planner(combined)
        
        # Classification loss
        target = torch.tensor([optimal_action], device=DEVICE)
        action_loss = F.cross_entropy(action_logits.unsqueeze(0), target)
        
        # Value estimation loss
        # After taking action, should be closer to goal
        action = self.action_types[optimal_action]
        velocity = self._action_to_velocity(action)
        
        with torch.no_grad():
            next_state = self.predict_with_velocity(current_state, velocity)
            next_similarity = F.cosine_similarity(
                next_state.unsqueeze(0),
                goal_state.unsqueeze(0)
            )
        
        predicted_value = self.value_net(current_state)
        value_loss = F.mse_loss(predicted_value.squeeze(), next_similarity.squeeze())
        
        total_loss = action_loss + 0.3 * value_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'action_loss': action_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
        }


class CausalBrain:
    """
    Brain with causal reasoning capabilities.
    
    Extends PredictiveBrain with CausalATL.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        n_concepts: int = 200,
        visual_input_size: int = 56,
    ):
        from brain_crossmodal_learner import SimpleVisualCortex56, SimpleLanguageCortex
        
        self.visual = SimpleVisualCortex56(feature_dim=feature_dim).to(DEVICE)
        self.language = SimpleLanguageCortex(feature_dim=feature_dim).to(DEVICE)
        self.atl = CausalATL(
            feature_dim=feature_dim,
            n_concepts=n_concepts,
            temperature=0.2,
        ).to(DEVICE)
        
        self.feature_dim = feature_dim
        self.n_concepts = n_concepts
    
    def load_from_predictive(self, checkpoint_path: str):
        """Load weights from trained PredictiveBrain."""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        
        self.visual.load_state_dict(checkpoint['visual_state'])
        self.language.load_state_dict(checkpoint['language_state'])
        
        # Load ATL weights (compatible subset)
        atl_state = checkpoint['atl_state']
        current_state = self.atl.state_dict()
        
        # Only load matching keys
        for key in atl_state:
            if key in current_state and atl_state[key].shape == current_state[key].shape:
                current_state[key] = atl_state[key]
        
        self.atl.load_state_dict(current_state)
        print(f"Loaded weights from {checkpoint_path}")
    
    def get_scene_activation(self, image: torch.Tensor) -> torch.Tensor:
        """Get distributed activation pattern for scene."""
        image = image.to(DEVICE)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.visual(image)  # Use forward() not encode()
            if features.dim() == 2:
                features = features.squeeze(0)
            activation, _ = self.atl.get_activation_pattern(features, 'visual')
        
        return activation
    
    def infer_causality_from_images(
        self,
        image_before: torch.Tensor,
        image_after: torch.Tensor,
    ) -> Dict:
        """Infer causality between two images."""
        act_before = self.get_scene_activation(image_before)
        act_after = self.get_scene_activation(image_after)
        
        return self.atl.infer_causality(act_before, act_after)
    
    def plan_from_images(
        self,
        current_image: torch.Tensor,
        goal_image: torch.Tensor,
        max_steps: int = 10,
    ) -> Tuple[List[ActionType], float]:
        """Plan actions to get from current to goal image."""
        current_act = self.get_scene_activation(current_image)
        goal_act = self.get_scene_activation(goal_image)
        
        plan, states, final_sim = self.atl.plan_to_goal(
            current_act, goal_act, max_steps
        )
        
        return plan, final_sim


if __name__ == "__main__":
    print("Testing CausalATL...")
    print(f"Device: {DEVICE}")
    
    # Initialize
    causal_atl = CausalATL(feature_dim=64, n_concepts=200)
    print(f"CausalATL parameters:")
    print(f"  Causal encoder: {sum(p.numel() for p in causal_atl.causal_encoder.parameters()):,}")
    print(f"  Causal decoder: {sum(p.numel() for p in causal_atl.causal_decoder.parameters()):,}")
    print(f"  Interaction classifier: {sum(p.numel() for p in causal_atl.interaction_classifier.parameters()):,}")
    print(f"  Constraint net: {sum(p.numel() for p in causal_atl.constraint_net.parameters()):,}")
    print(f"  Planner: {sum(p.numel() for p in causal_atl.planner.parameters()):,}")
    
    # Test causal inference
    state_before = torch.randn(200, device=DEVICE)
    state_after = state_before + torch.randn(200, device=DEVICE) * 0.3
    
    result = causal_atl.infer_causality(state_before, state_after)
    print(f"\nCausal inference test:")
    print(f"  Causal strength shape: {result['causal_strength'].shape}")
    print(f"  Interaction probs: {result['interaction_probs']}")
    print(f"  Change magnitude: {result['change_magnitude']:.4f}")
    
    # Test planning
    current = torch.randn(200, device=DEVICE)
    goal = torch.randn(200, device=DEVICE)
    
    plan, states, final_sim = causal_atl.plan_to_goal(current, goal)
    print(f"\nPlanning test:")
    print(f"  Plan length: {len(plan)}")
    print(f"  Actions: {[a.value for a in plan[:5]]}...")
    print(f"  Final similarity to goal: {final_sim:.4f}")
    
    # Test counterfactual
    cf_result = causal_atl.counterfactual_imagine(
        current,
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_LEFT
    )
    print(f"\nCounterfactual test:")
    print(f"  Outcome difference: {cf_result['outcome_difference']:.4f}")
    
    print("\n✓ CausalATL module working!")
