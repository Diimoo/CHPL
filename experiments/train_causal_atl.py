#!/usr/bin/env python3
"""
Phase 2: Causal Reasoning & Planning - Training Script

Trains CausalATL on:
1. Causal inference (identifying what caused observed changes)
2. Interaction classification (push, block, independent, chain)
3. Goal-directed planning (action selection to reach goals)

Builds on Phase 1's temporal prediction capabilities.

Success Criteria (from roadmap):
- Causal inference accuracy > 0.7
- Interaction classification > 0.8
- Goal completion rate > 0.6
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

from causal_atl import CausalATL, CausalBrain, ActionType, InteractionType
from synthetic_causal import (
    generate_causal_dataset, CausalSequence, CausalInteractionType,
    get_interaction_label, create_push_sequence, create_independent_sequence
)
from predictive_atl import DEVICE


def ensure_cuda():
    """Ensure we're running on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training. No GPU detected.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return torch.device('cuda')


def find_latest_model(results_dir: str = '../prediction_results') -> str:
    """Find the most recent trained model from Phase 1."""
    model_files = [f for f in os.listdir(results_dir) 
                   if f.startswith('predictive_brain_') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No Phase 1 models found in {results_dir}")
    
    model_files.sort(reverse=True)
    return os.path.join(results_dir, model_files[0])


def train_causal_inference(
    brain: CausalBrain,
    sequences: List[CausalSequence],
    n_epochs: int = 20,
) -> Dict:
    """
    Phase 2.1: Train causal inference network.
    
    Learn to:
    - Detect when causal interaction occurred
    - Identify cause and effect
    - Classify interaction type
    """
    print("\n" + "=" * 60)
    print("PHASE 2.1: Causal Inference Training")
    print("Learning to identify causes and effects")
    print("=" * 60)
    
    # Optimizer for causal networks only
    causal_params = list(brain.atl.causal_encoder.parameters()) + \
                    list(brain.atl.causal_decoder.parameters()) + \
                    list(brain.atl.interaction_classifier.parameters())
    
    optimizer = torch.optim.Adam(causal_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    stats = {'losses': [], 'accuracies': [], 'per_class_acc': []}
    
    for epoch in range(n_epochs):
        np.random.shuffle(sequences)
        
        epoch_losses = []
        correct = 0
        total = 0
        class_correct = {t.value: 0 for t in CausalInteractionType}
        class_total = {t.value: 0 for t in CausalInteractionType}
        
        for seq in sequences:
            if len(seq.frames) < 2:
                continue
            
            # Get before/after frames around causal event
            if seq.cause_frame >= 0 and seq.effect_frame >= 0:
                before_idx = max(0, seq.cause_frame - 1)
                after_idx = min(len(seq.frames) - 1, seq.effect_frame)
            else:
                before_idx = 0
                after_idx = len(seq.frames) - 1
            
            before_img = torch.from_numpy(seq.frames[before_idx].image).float()
            after_img = torch.from_numpy(seq.frames[after_idx].image).float()
            
            # Transpose to CHW format
            before_img = before_img.permute(2, 0, 1).to(DEVICE)
            after_img = after_img.permute(2, 0, 1).to(DEVICE)
            
            # Get activations
            with torch.no_grad():
                before_act = brain.get_scene_activation(before_img)
                after_act = brain.get_scene_activation(after_img)
            
            # Forward pass
            combined = torch.cat([before_act, after_act])
            
            # Causal encoding
            causal_features = brain.atl.causal_encoder(combined)
            causal_strength = brain.atl.causal_decoder(causal_features)
            
            # Interaction classification
            interaction_logits = brain.atl.interaction_classifier(combined)
            
            # Ground truth
            target_label = get_interaction_label(seq.interaction_type)
            target = torch.tensor([target_label], device=DEVICE)
            
            # Classification loss
            classification_loss = F.cross_entropy(
                interaction_logits.unsqueeze(0), target
            )
            
            # Causal strength loss: should correlate with actual change
            change = torch.abs(after_act - before_act)
            change_norm = change / (change.max() + 1e-8)
            
            # For interactions, expect high causal strength; for independent, low
            if seq.interaction_type == CausalInteractionType.INDEPENDENT:
                # Should predict low causality
                causal_target = torch.zeros_like(causal_strength)
            else:
                # Should predict high causality where change occurred
                causal_target = change_norm
            
            causal_loss = F.mse_loss(causal_strength, causal_target)
            
            total_loss = classification_loss + 0.3 * causal_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # Accuracy tracking
            predicted = torch.argmax(interaction_logits).item()
            is_correct = (predicted == target_label)
            correct += int(is_correct)
            total += 1
            
            itype = seq.interaction_type.value
            class_correct[itype] += int(is_correct)
            class_total[itype] += 1
        
        scheduler.step()
        
        epoch_loss = np.mean(epoch_losses)
        epoch_acc = correct / total if total > 0 else 0
        
        stats['losses'].append(epoch_loss)
        stats['accuracies'].append(epoch_acc)
        
        # Per-class accuracy
        per_class = {k: class_correct[k] / max(class_total[k], 1) 
                     for k in class_correct}
        stats['per_class_acc'].append(per_class)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.3f}")
            for itype, acc in per_class.items():
                print(f"    {itype}: {acc:.3f}")
    
    final_acc = stats['accuracies'][-1]
    print(f"\n  Final causal inference accuracy: {final_acc:.3f}")
    print(f"  Target: > 0.7 | Status: {'PASS ✓' if final_acc > 0.7 else 'FAIL ✗'}")
    
    return stats


def train_planning(
    brain: CausalBrain,
    n_episodes: int = 500,
    max_steps: int = 10,
) -> Dict:
    """
    Phase 2.2: Train goal-directed planning.
    
    Learn to select actions that move toward goal states.
    """
    print("\n" + "=" * 60)
    print("PHASE 2.2: Goal-Directed Planning Training")
    print(f"Training on {n_episodes} episodes, max {max_steps} steps each")
    print("=" * 60)
    
    # Optimizer for planner and value networks
    planner_params = list(brain.atl.planner.parameters()) + \
                     list(brain.atl.value_net.parameters())
    
    optimizer = torch.optim.Adam(planner_params, lr=5e-4)
    
    stats = {'rewards': [], 'success_rates': [], 'avg_steps': []}
    
    # Generate start/goal pairs from push sequences
    print("  Generating planning scenarios...")
    
    for episode in range(n_episodes):
        # Random start and goal states
        # Use push sequences: goal is the final state after push
        seq = create_push_sequence(
            pusher_shape=np.random.choice(['circle', 'square', 'triangle']),
            pusher_color=np.random.choice(['red', 'blue', 'green']),
            pushed_shape=np.random.choice(['circle', 'square', 'triangle']),
            pushed_color=np.random.choice(['red', 'blue', 'green', 'yellow']),
        )
        
        # Start: first frame, Goal: last frame
        start_img = torch.from_numpy(seq.frames[0].image).float().permute(2, 0, 1).to(DEVICE)
        goal_img = torch.from_numpy(seq.frames[-1].image).float().permute(2, 0, 1).to(DEVICE)
        
        with torch.no_grad():
            start_act = brain.get_scene_activation(start_img)
            goal_act = brain.get_scene_activation(goal_img)
        
        current_act = start_act.clone()
        
        episode_reward = 0
        steps_taken = 0
        
        for step in range(max_steps):
            # Get current similarity to goal
            current_sim = F.cosine_similarity(
                current_act.unsqueeze(0),
                goal_act.unsqueeze(0)
            ).item()
            
            if current_sim > 0.9:
                # Goal reached
                episode_reward += 1.0
                break
            
            # Get action from planner
            combined = torch.cat([current_act, goal_act])
            action_logits = brain.atl.planner(combined)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample action
            if np.random.rand() < 0.2:  # Exploration
                action_idx = np.random.randint(len(brain.atl.action_types))
            else:
                action_idx = torch.argmax(action_probs).item()
            
            action = brain.atl.action_types[action_idx]
            velocity = brain.atl._action_to_velocity(action)
            
            # Predict next state
            with torch.no_grad():
                next_act = brain.atl.predict_with_velocity(current_act, velocity)
            
            # Compute reward: improvement in similarity
            next_sim = F.cosine_similarity(
                next_act.unsqueeze(0),
                goal_act.unsqueeze(0)
            ).item()
            
            reward = next_sim - current_sim  # Reward for getting closer
            
            # Training step
            # Target: action that improves similarity most
            best_action_idx = action_idx
            best_improvement = reward
            
            for a_idx, a_type in enumerate(brain.atl.action_types):
                a_vel = brain.atl._action_to_velocity(a_type)
                with torch.no_grad():
                    a_next = brain.atl.predict_with_velocity(current_act, a_vel)
                    a_sim = F.cosine_similarity(
                        a_next.unsqueeze(0),
                        goal_act.unsqueeze(0)
                    ).item()
                    a_improvement = a_sim - current_sim
                    if a_improvement > best_improvement:
                        best_improvement = a_improvement
                        best_action_idx = a_idx
            
            # Train planner to select best action
            target = torch.tensor([best_action_idx], device=DEVICE)
            action_loss = F.cross_entropy(action_logits.unsqueeze(0), target)
            
            # Train value network
            predicted_value = brain.atl.value_net(current_act)
            target_value = torch.tensor([[next_sim]], device=DEVICE)
            value_loss = F.mse_loss(predicted_value, target_value)
            
            total_loss = action_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            episode_reward += reward
            current_act = next_act
            steps_taken += 1
        
        stats['rewards'].append(episode_reward)
        stats['avg_steps'].append(steps_taken)
        
        # Success = reached goal
        success = current_sim > 0.9 if 'current_sim' in dir() else F.cosine_similarity(
            current_act.unsqueeze(0), goal_act.unsqueeze(0)
        ).item() > 0.9
        stats['success_rates'].append(float(success))
        
        if (episode + 1) % 100 == 0:
            recent_reward = np.mean(stats['rewards'][-100:])
            recent_success = np.mean(stats['success_rates'][-100:])
            print(f"  Episode {episode+1}: reward={recent_reward:.3f}, success={recent_success:.3f}")
    
    final_success = np.mean(stats['success_rates'][-100:])
    print(f"\n  Final planning success rate: {final_success:.3f}")
    print(f"  Target: > 0.6 | Status: {'PASS ✓' if final_success > 0.6 else 'FAIL ✗'}")
    
    return stats


def test_causal_reasoning(brain: CausalBrain, n_test: int = 100) -> Dict:
    """
    Test causal reasoning capabilities.
    """
    print("\n" + "=" * 60)
    print("CAUSAL REASONING TEST")
    print("=" * 60)
    
    # Generate test sequences
    test_sequences = generate_causal_dataset(
        n_push=n_test // 4,
        n_block=n_test // 4,
        n_independent=n_test // 4,
        n_chain=n_test // 4,
    )
    
    correct = 0
    total = 0
    confusion = {}
    
    for seq in test_sequences:
        if len(seq.frames) < 2:
            continue
        
        before_idx = max(0, seq.cause_frame - 1) if seq.cause_frame >= 0 else 0
        after_idx = min(len(seq.frames) - 1, seq.effect_frame) if seq.effect_frame >= 0 else len(seq.frames) - 1
        
        before_img = torch.from_numpy(seq.frames[before_idx].image).float().permute(2, 0, 1).to(DEVICE)
        after_img = torch.from_numpy(seq.frames[after_idx].image).float().permute(2, 0, 1).to(DEVICE)
        
        with torch.no_grad():
            result = brain.infer_causality_from_images(before_img, after_img)
        
        predicted_idx = torch.argmax(result['interaction_probs']).item()
        true_idx = get_interaction_label(seq.interaction_type)
        
        interaction_types = ['push', 'block', 'independent', 'chain']
        predicted_type = interaction_types[predicted_idx] if predicted_idx < len(interaction_types) else 'unknown'
        true_type = seq.interaction_type.value
        
        if predicted_idx == true_idx:
            correct += 1
        
        key = (true_type, predicted_type)
        confusion[key] = confusion.get(key, 0) + 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n  Causal inference accuracy: {accuracy:.3f}")
    print(f"  Target: > 0.7 | Status: {'PASS ✓' if accuracy > 0.7 else 'FAIL ✗'}")
    
    print("\n  Confusion (true → predicted):")
    for (true_t, pred_t), count in sorted(confusion.items()):
        print(f"    {true_t} → {pred_t}: {count}")
    
    return {
        'accuracy': accuracy,
        'confusion': confusion,
        'passed': accuracy > 0.7,
    }


def test_planning(brain: CausalBrain, n_episodes: int = 50) -> Dict:
    """
    Test goal-directed planning.
    """
    print("\n" + "=" * 60)
    print("PLANNING TEST")
    print("=" * 60)
    
    successes = []
    path_lengths = []
    
    for _ in range(n_episodes):
        seq = create_push_sequence()
        
        start_img = torch.from_numpy(seq.frames[0].image).float().permute(2, 0, 1).to(DEVICE)
        goal_img = torch.from_numpy(seq.frames[-1].image).float().permute(2, 0, 1).to(DEVICE)
        
        plan, final_sim = brain.plan_from_images(start_img, goal_img, max_steps=15)
        
        success = final_sim > 0.8
        successes.append(float(success))
        path_lengths.append(len(plan))
    
    success_rate = np.mean(successes)
    avg_path_length = np.mean(path_lengths)
    
    print(f"\n  Planning success rate: {success_rate:.3f}")
    print(f"  Average path length: {avg_path_length:.1f} steps")
    print(f"  Target: > 0.6 | Status: {'PASS ✓' if success_rate > 0.6 else 'FAIL ✗'}")
    
    return {
        'success_rate': success_rate,
        'avg_path_length': avg_path_length,
        'passed': success_rate > 0.6,
    }


def test_counterfactual(brain: CausalBrain, n_tests: int = 30) -> Dict:
    """
    Test counterfactual reasoning.
    """
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL REASONING TEST")
    print("What would have happened with different action?")
    print("=" * 60)
    
    differences = []
    
    for _ in range(n_tests):
        seq = create_push_sequence()
        img = torch.from_numpy(seq.frames[1].image).float().permute(2, 0, 1).to(DEVICE)
        
        with torch.no_grad():
            state = brain.get_scene_activation(img)
        
        # Compare: what if we pushed right vs left?
        result = brain.atl.counterfactual_imagine(
            state,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_LEFT
        )
        
        differences.append(result['outcome_difference'])
    
    mean_diff = np.mean(differences)
    
    print(f"\n  Mean counterfactual difference: {mean_diff:.4f}")
    print(f"  (Higher = better discrimination between actions)")
    
    passed = mean_diff > 0.05
    print(f"  Target: > 0.05 | Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return {
        'mean_difference': mean_diff,
        'passed': passed,
    }


def run_full_experiment():
    """Run complete Phase 2 experiment."""
    print("=" * 70)
    print("CHPL COGNITIVE DEVELOPMENT - PHASE 2: CAUSAL REASONING & PLANNING")
    print("=" * 70)
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ensure_cuda()
    
    # Initialize brain from Phase 1
    print("\nLoading Phase 1 brain...")
    model_path = find_latest_model()
    print(f"Found model: {model_path}")
    
    brain = CausalBrain(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=56,
    )
    brain.load_from_predictive(model_path)
    
    print(f"\nCausal ATL parameters:")
    print(f"  Causal encoder: {sum(p.numel() for p in brain.atl.causal_encoder.parameters()):,}")
    print(f"  Causal decoder: {sum(p.numel() for p in brain.atl.causal_decoder.parameters()):,}")
    print(f"  Interaction classifier: {sum(p.numel() for p in brain.atl.interaction_classifier.parameters()):,}")
    print(f"  Planner: {sum(p.numel() for p in brain.atl.planner.parameters()):,}")
    
    results = {}
    
    # Generate training data
    print("\n" + "=" * 60)
    print("Generating causal interaction dataset...")
    print("=" * 60)
    train_sequences = generate_causal_dataset(
        n_push=300,
        n_block=300,
        n_independent=300,
        n_chain=100,
    )
    
    # Phase 2.1: Causal Inference
    causal_stats = train_causal_inference(brain, train_sequences, n_epochs=20)
    results['causal_training'] = {
        'final_accuracy': causal_stats['accuracies'][-1],
        'final_loss': causal_stats['losses'][-1],
    }
    
    # Phase 2.2: Planning
    planning_stats = train_planning(brain, n_episodes=500, max_steps=10)
    results['planning_training'] = {
        'final_success_rate': np.mean(planning_stats['success_rates'][-100:]),
        'final_reward': np.mean(planning_stats['rewards'][-100:]),
    }
    
    # Testing
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    causal_test = test_causal_reasoning(brain, n_test=200)
    results['causal_test'] = causal_test
    
    planning_test = test_planning(brain, n_episodes=50)
    results['planning_test'] = planning_test
    
    counterfactual_test = test_counterfactual(brain, n_tests=50)
    results['counterfactual_test'] = counterfactual_test
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} minutes")
    
    print(f"\n  Phase 2.1 Results (Causal Inference):")
    print(f"    - Interaction classification: {causal_test['accuracy']:.3f}")
    print(f"      Target: > 0.7 | Status: {'PASS ✓' if causal_test['passed'] else 'FAIL ✗'}")
    
    print(f"\n  Phase 2.2 Results (Planning):")
    print(f"    - Goal completion rate: {planning_test['success_rate']:.3f}")
    print(f"      Target: > 0.6 | Status: {'PASS ✓' if planning_test['passed'] else 'FAIL ✗'}")
    
    print(f"\n  Counterfactual Reasoning:")
    print(f"    - Action discrimination: {counterfactual_test['mean_difference']:.4f}")
    print(f"      Status: {'PASS ✓' if counterfactual_test['passed'] else 'FAIL ✗'}")
    
    # Overall assessment
    all_passed = causal_test['passed'] and planning_test['passed']
    
    print("\n" + "-" * 70)
    if all_passed:
        print("  ★ EARLY CHILD LEVEL REACHED ★")
        print("  CHPL can now reason about causes and plan actions!")
    else:
        print("  Phase 2 partially complete. More training may be needed.")
    print("-" * 70)
    
    # Save results
    results_dir = Path('../causal_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = results_dir / f'causal_brain_{timestamp}.pt'
    torch.save({
        'visual_state': brain.visual.state_dict(),
        'language_state': brain.language.state_dict(),
        'atl_state': brain.atl.state_dict(),
    }, model_path)
    print(f"\n  Model saved to: {model_path}")
    
    # Save results JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    results_path = results_dir / f'causal_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    return brain, results


if __name__ == "__main__":
    brain, results = run_full_experiment()
