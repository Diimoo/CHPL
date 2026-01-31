#!/usr/bin/env python3
"""
Rigorous, honest tests for PredictiveATL.

These tests are designed to be HARD and HONEST for research purposes:
1. Out-of-distribution generalization
2. Novel object combinations never seen in training
3. Different motion patterns than training
4. Long-horizon prediction (multiple steps ahead)
5. Adversarial cases

If the model passes these, it's genuinely learning temporal structure.
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
import math

from predictive_atl import (
    PredictiveBrain, PredictiveATL,
    create_moving_object_sequence, generate_temporal_dataset,
    generate_occlusion_dataset, TemporalSequence, DEVICE
)
from synthetic_environment import (
    SHAPES, COLORS, create_stimulus_on_canvas
)


def load_trained_model(model_dir: str = '../prediction_results') -> PredictiveBrain:
    """Load the most recently trained model."""
    # Find most recent model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith('predictive_brain_') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}")
    
    model_files.sort(reverse=True)  # Most recent first
    model_path = os.path.join(model_dir, model_files[0])
    
    print(f"Loading model: {model_path}")
    
    brain = PredictiveBrain(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=56,
    )
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    brain.visual.load_state_dict(checkpoint['visual_state'])
    brain.language.load_state_dict(checkpoint['language_state'])
    brain.atl.load_state_dict(checkpoint['atl_state'])
    
    return brain


def test_novel_shapes(brain: PredictiveBrain) -> Dict:
    """
    Test 1: Novel shapes not seen during training.
    
    Training used: circle, square, triangle, star
    Test with: cross, diamond (held out)
    """
    print("\n" + "=" * 60)
    print("TEST 1: Novel Shapes (Out-of-Distribution)")
    print("Shapes not seen during training: cross, diamond")
    print("=" * 60)
    
    novel_shapes = ['cross', 'diamond']
    colors = ['red', 'blue', 'green']
    
    results = {'cos_sims': [], 'sequences': []}
    
    for shape in novel_shapes:
        for color in colors:
            # Create moving sequence with novel shape
            seq = create_moving_object_sequence(
                shape=shape,
                color=color,
                start_pos=(15, 28),
                velocity=(4, 0),
                n_frames=8,
                canvas_size=56,
            )
            
            # Test prediction on consecutive frames
            cos_sims = []
            with torch.no_grad():
                for t in range(len(seq) - 1):
                    current_img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                    next_img = torch.from_numpy(seq.frames[t + 1].image).float().to(DEVICE)
                    
                    vel = torch.tensor([
                        seq.frames[t].velocity[0] / 10.0,
                        seq.frames[t].velocity[1] / 10.0,
                    ], device=DEVICE)
                    
                    result = brain.evaluate_prediction(current_img, next_img, vel)
                    cos_sims.append(result['cosine_similarity'])
            
            mean_sim = np.mean(cos_sims)
            results['cos_sims'].append(mean_sim)
            results['sequences'].append(f"{color} {shape}")
            print(f"  {color} {shape}: cos_sim = {mean_sim:.4f}")
    
    overall_mean = np.mean(results['cos_sims'])
    print(f"\n  Overall novel shape accuracy: {overall_mean:.4f}")
    print(f"  Target: > 0.5 (lower bar for novel shapes)")
    print(f"  Status: {'PASS ✓' if overall_mean > 0.5 else 'FAIL ✗'}")
    
    return {
        'test': 'novel_shapes',
        'mean_cos_sim': overall_mean,
        'passed': overall_mean > 0.5,
        'details': results,
    }


def test_novel_colors(brain: PredictiveBrain) -> Dict:
    """
    Test 2: Novel colors not seen during training.
    
    Training used: red, blue, green, yellow
    Test with: purple, orange (held out)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Novel Colors (Out-of-Distribution)")
    print("Colors not seen during training: purple, orange")
    print("=" * 60)
    
    novel_colors = ['purple', 'orange']
    shapes = ['circle', 'square', 'triangle']
    
    results = {'cos_sims': [], 'sequences': []}
    
    for color in novel_colors:
        for shape in shapes:
            seq = create_moving_object_sequence(
                shape=shape,
                color=color,
                start_pos=(15, 28),
                velocity=(3, 2),
                n_frames=8,
                canvas_size=56,
            )
            
            cos_sims = []
            with torch.no_grad():
                for t in range(len(seq) - 1):
                    current_img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                    next_img = torch.from_numpy(seq.frames[t + 1].image).float().to(DEVICE)
                    
                    vel = torch.tensor([
                        seq.frames[t].velocity[0] / 10.0,
                        seq.frames[t].velocity[1] / 10.0,
                    ], device=DEVICE)
                    
                    result = brain.evaluate_prediction(current_img, next_img, vel)
                    cos_sims.append(result['cosine_similarity'])
            
            mean_sim = np.mean(cos_sims)
            results['cos_sims'].append(mean_sim)
            results['sequences'].append(f"{color} {shape}")
            print(f"  {color} {shape}: cos_sim = {mean_sim:.4f}")
    
    overall_mean = np.mean(results['cos_sims'])
    print(f"\n  Overall novel color accuracy: {overall_mean:.4f}")
    print(f"  Target: > 0.5 (lower bar for novel colors)")
    print(f"  Status: {'PASS ✓' if overall_mean > 0.5 else 'FAIL ✗'}")
    
    return {
        'test': 'novel_colors',
        'mean_cos_sim': overall_mean,
        'passed': overall_mean > 0.5,
        'details': results,
    }


def test_novel_motion_patterns(brain: PredictiveBrain) -> Dict:
    """
    Test 3: Novel motion patterns.
    
    Training used: linear, bounce
    Test with: circular, zigzag, acceleration
    """
    print("\n" + "=" * 60)
    print("TEST 3: Novel Motion Patterns")
    print("Patterns not in training: circular, zigzag, acceleration")
    print("=" * 60)
    
    results = {'patterns': {}}
    
    # Test 1: Circular motion
    print("\n  Testing circular motion...")
    circular_sims = []
    for _ in range(10):
        shape = np.random.choice(['circle', 'square', 'triangle'])
        color = np.random.choice(['red', 'blue', 'green'])
        
        seq = create_moving_object_sequence(
            shape=shape,
            color=color,
            start_pos=(28, 28),
            velocity=(0, 0),  # Will be overridden by circular
            n_frames=12,
            canvas_size=56,
            motion_type='circular',
        )
        
        with torch.no_grad():
            for t in range(len(seq) - 1):
                current_img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                next_img = torch.from_numpy(seq.frames[t + 1].image).float().to(DEVICE)
                
                vel = torch.tensor([
                    seq.frames[t].velocity[0] / 10.0,
                    seq.frames[t].velocity[1] / 10.0,
                ], device=DEVICE)
                
                result = brain.evaluate_prediction(current_img, next_img, vel)
                circular_sims.append(result['cosine_similarity'])
    
    circular_mean = np.mean(circular_sims)
    results['patterns']['circular'] = circular_mean
    print(f"    Circular motion: {circular_mean:.4f}")
    
    # Test 2: Zigzag motion (alternating direction each frame)
    print("\n  Testing zigzag motion...")
    zigzag_sims = []
    for _ in range(10):
        shape = np.random.choice(['circle', 'square', 'triangle'])
        color = np.random.choice(['red', 'blue', 'green'])
        
        # Create zigzag manually
        frames = []
        pos = [20, 28]
        vel_x = 4
        
        for t in range(10):
            img = create_stimulus_on_canvas(
                shape=shape,
                color=color,
                size='small',
                canvas_size=56,
                center=(int(pos[0]), int(pos[1])),
                noise=0.02,
            )
            
            from predictive_atl import TemporalFrame
            frames.append(TemporalFrame(
                image=img,
                label=f"{color} {shape}",
                position=(int(pos[0]), int(pos[1])),
                velocity=(vel_x, 0),
                timestep=t,
            ))
            
            # Zigzag: reverse x direction every frame
            pos[0] += vel_x
            vel_x = -vel_x
        
        seq = TemporalSequence(
            frames=frames,
            object_shape=shape,
            object_color=color,
            motion_type='zigzag',
        )
        
        with torch.no_grad():
            for t in range(len(seq) - 1):
                current_img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                next_img = torch.from_numpy(seq.frames[t + 1].image).float().to(DEVICE)
                
                vel = torch.tensor([
                    seq.frames[t].velocity[0] / 10.0,
                    seq.frames[t].velocity[1] / 10.0,
                ], device=DEVICE)
                
                result = brain.evaluate_prediction(current_img, next_img, vel)
                zigzag_sims.append(result['cosine_similarity'])
    
    zigzag_mean = np.mean(zigzag_sims)
    results['patterns']['zigzag'] = zigzag_mean
    print(f"    Zigzag motion: {zigzag_mean:.4f}")
    
    # Test 3: Accelerating motion
    print("\n  Testing accelerating motion...")
    accel_sims = []
    for _ in range(10):
        shape = np.random.choice(['circle', 'square', 'triangle'])
        color = np.random.choice(['red', 'blue', 'green'])
        
        # Create accelerating motion
        frames = []
        pos = [15, 28]
        vel = 1.0
        
        for t in range(10):
            img = create_stimulus_on_canvas(
                shape=shape,
                color=color,
                size='small',
                canvas_size=56,
                center=(int(pos[0]), int(pos[1])),
                noise=0.02,
            )
            
            from predictive_atl import TemporalFrame
            frames.append(TemporalFrame(
                image=img,
                label=f"{color} {shape}",
                position=(int(pos[0]), int(pos[1])),
                velocity=(vel, 0),
                timestep=t,
            ))
            
            pos[0] += vel
            vel *= 1.3  # Accelerate
        
        seq = TemporalSequence(
            frames=frames,
            object_shape=shape,
            object_color=color,
            motion_type='accelerating',
        )
        
        with torch.no_grad():
            for t in range(len(seq) - 1):
                current_img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                next_img = torch.from_numpy(seq.frames[t + 1].image).float().to(DEVICE)
                
                vel = torch.tensor([
                    seq.frames[t].velocity[0] / 10.0,
                    seq.frames[t].velocity[1] / 10.0,
                ], device=DEVICE)
                
                result = brain.evaluate_prediction(current_img, next_img, vel)
                accel_sims.append(result['cosine_similarity'])
    
    accel_mean = np.mean(accel_sims)
    results['patterns']['accelerating'] = accel_mean
    print(f"    Accelerating motion: {accel_mean:.4f}")
    
    overall_mean = np.mean([circular_mean, zigzag_mean, accel_mean])
    print(f"\n  Overall novel motion accuracy: {overall_mean:.4f}")
    print(f"  Target: > 0.4 (harder test for novel patterns)")
    print(f"  Status: {'PASS ✓' if overall_mean > 0.4 else 'FAIL ✗'}")
    
    return {
        'test': 'novel_motion',
        'mean_cos_sim': overall_mean,
        'passed': overall_mean > 0.4,
        'details': results,
    }


def test_long_horizon_prediction(brain: PredictiveBrain, max_steps: int = 5) -> Dict:
    """
    Test 4: Multi-step prediction without ground truth correction.
    
    Predict t+1 from t, then t+2 from predicted t+1, etc.
    This tests accumulated error and planning potential.
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: Long-Horizon Prediction ({max_steps} steps ahead)")
    print("Predicting multiple steps without ground truth correction")
    print("=" * 60)
    
    results = {f'step_{i}': [] for i in range(1, max_steps + 1)}
    
    n_sequences = 20
    for i in range(n_sequences):
        shape = np.random.choice(['circle', 'square', 'triangle', 'star'])
        color = np.random.choice(['red', 'blue', 'green', 'yellow'])
        
        seq = create_moving_object_sequence(
            shape=shape,
            color=color,
            start_pos=(15, 28),
            velocity=(4, 0),
            n_frames=max_steps + 2,
            canvas_size=56,
        )
        
        with torch.no_grad():
            # Get initial activation
            current_img = torch.from_numpy(seq.frames[0].image).float().to(DEVICE)
            current_act = brain.get_scene_activation(current_img)
            
            for step in range(1, max_steps + 1):
                # Get velocity for this step
                vel = torch.tensor([
                    seq.frames[step - 1].velocity[0] / 10.0,
                    seq.frames[step - 1].velocity[1] / 10.0,
                ], device=DEVICE)
                
                # Predict next activation (from predicted, not ground truth)
                predicted_act = brain.atl.predict_with_velocity(current_act, vel)
                
                # Get ground truth activation for comparison
                gt_img = torch.from_numpy(seq.frames[step].image).float().to(DEVICE)
                gt_act = brain.get_scene_activation(gt_img)
                
                # Measure similarity
                cos_sim = F.cosine_similarity(
                    predicted_act.unsqueeze(0),
                    gt_act.unsqueeze(0)
                ).item()
                
                results[f'step_{step}'].append(cos_sim)
                
                # Use predicted activation for next step (no correction)
                current_act = predicted_act
    
    print("\n  Accuracy by prediction horizon:")
    step_means = {}
    for step in range(1, max_steps + 1):
        mean_sim = np.mean(results[f'step_{step}'])
        step_means[step] = mean_sim
        print(f"    Step {step}: cos_sim = {mean_sim:.4f}")
    
    # Decay rate: how fast does accuracy decrease?
    decay_rate = (step_means[1] - step_means[max_steps]) / max_steps
    print(f"\n  Accuracy decay per step: {decay_rate:.4f}")
    
    # Pass if step 3 still has > 0.5 similarity
    step3_passed = step_means.get(3, 0) > 0.5 if max_steps >= 3 else step_means[min(max_steps, 2)] > 0.5
    
    print(f"  Target: Step 3 similarity > 0.5")
    print(f"  Status: {'PASS ✓' if step3_passed else 'FAIL ✗'}")
    
    return {
        'test': 'long_horizon',
        'step_accuracies': step_means,
        'decay_rate': decay_rate,
        'passed': step3_passed,
    }


def test_stationary_vs_moving(brain: PredictiveBrain) -> Dict:
    """
    Test 5: Can the model distinguish stationary from moving objects?
    
    If velocity is zero, predicted state should be similar to current.
    If velocity is non-zero, predicted state should be different.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Stationary vs Moving Discrimination")
    print("Zero velocity → same state, non-zero → different state")
    print("=" * 60)
    
    stationary_diffs = []
    moving_diffs = []
    
    for _ in range(30):
        shape = np.random.choice(['circle', 'square', 'triangle'])
        color = np.random.choice(['red', 'blue', 'green'])
        
        pos_x = np.random.randint(20, 36)
        pos_y = np.random.randint(20, 36)
        
        img = create_stimulus_on_canvas(
            shape=shape,
            color=color,
            size='small',
            canvas_size=56,
            center=(pos_x, pos_y),
        )
        img_t = torch.from_numpy(img).float().to(DEVICE)
        
        with torch.no_grad():
            current_act = brain.get_scene_activation(img_t)
            
            # Stationary prediction (velocity = 0)
            zero_vel = torch.tensor([0.0, 0.0], device=DEVICE)
            stationary_pred = brain.atl.predict_with_velocity(current_act, zero_vel)
            
            stat_diff = 1 - F.cosine_similarity(
                current_act.unsqueeze(0),
                stationary_pred.unsqueeze(0)
            ).item()
            stationary_diffs.append(stat_diff)
            
            # Moving prediction (velocity ≠ 0)
            moving_vel = torch.tensor([0.4, 0.0], device=DEVICE)
            moving_pred = brain.atl.predict_with_velocity(current_act, moving_vel)
            
            mov_diff = 1 - F.cosine_similarity(
                current_act.unsqueeze(0),
                moving_pred.unsqueeze(0)
            ).item()
            moving_diffs.append(mov_diff)
    
    mean_stat_diff = np.mean(stationary_diffs)
    mean_mov_diff = np.mean(moving_diffs)
    
    print(f"\n  Mean difference from current state:")
    print(f"    Stationary (v=0): {mean_stat_diff:.4f} (should be LOW)")
    print(f"    Moving (v≠0): {mean_mov_diff:.4f} (should be HIGH)")
    
    # Discrimination = moving difference > stationary difference
    discrimination = mean_mov_diff - mean_stat_diff
    print(f"\n  Discrimination (moving - stationary): {discrimination:.4f}")
    
    # Pass if there's clear discrimination
    passed = discrimination > 0.02 and mean_stat_diff < 0.1
    print(f"  Target: Discrimination > 0.02 AND stationary diff < 0.1")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return {
        'test': 'stationary_vs_moving',
        'stationary_diff': mean_stat_diff,
        'moving_diff': mean_mov_diff,
        'discrimination': discrimination,
        'passed': passed,
    }


def test_velocity_sensitivity(brain: PredictiveBrain) -> Dict:
    """
    Test 6: Is prediction sensitive to velocity magnitude and direction?
    
    Different velocities should produce different predictions.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Velocity Sensitivity")
    print("Different velocities should produce different predictions")
    print("=" * 60)
    
    # Create a fixed scene
    img = create_stimulus_on_canvas(
        shape='circle',
        color='red',
        size='small',
        canvas_size=56,
        center=(28, 28),
    )
    img_t = torch.from_numpy(img).float().to(DEVICE)
    
    velocities = [
        (0.3, 0.0),   # Right slow
        (0.5, 0.0),   # Right fast
        (-0.3, 0.0),  # Left slow
        (0.0, 0.3),   # Down
        (0.0, -0.3),  # Up
        (0.3, 0.3),   # Diagonal
    ]
    
    with torch.no_grad():
        current_act = brain.get_scene_activation(img_t)
        
        predictions = []
        for vel in velocities:
            vel_t = torch.tensor(vel, device=DEVICE)
            pred = brain.atl.predict_with_velocity(current_act, vel_t)
            predictions.append(pred)
        
        # Calculate pairwise differences
        differences = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                diff = 1 - F.cosine_similarity(
                    predictions[i].unsqueeze(0),
                    predictions[j].unsqueeze(0)
                ).item()
                differences.append(diff)
    
    mean_diff = np.mean(differences)
    
    print(f"\n  Velocities tested: {velocities}")
    print(f"  Mean pairwise difference between predictions: {mean_diff:.4f}")
    
    # Check specific cases
    # Right slow vs Right fast
    right_slow_vs_fast = 1 - F.cosine_similarity(
        predictions[0].unsqueeze(0),
        predictions[1].unsqueeze(0)
    ).item()
    
    # Right vs Left
    right_vs_left = 1 - F.cosine_similarity(
        predictions[0].unsqueeze(0),
        predictions[2].unsqueeze(0)
    ).item()
    
    print(f"  Right slow vs Right fast: {right_slow_vs_fast:.4f}")
    print(f"  Right vs Left: {right_vs_left:.4f}")
    
    # Pass if predictions are sensitive to velocity
    passed = mean_diff > 0.01 and right_vs_left > 0.01
    print(f"\n  Target: Mean difference > 0.01, opposite directions different")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return {
        'test': 'velocity_sensitivity',
        'mean_pairwise_diff': mean_diff,
        'right_slow_vs_fast': right_slow_vs_fast,
        'right_vs_left': right_vs_left,
        'passed': passed,
    }


def run_all_hard_tests():
    """Run all rigorous tests and report results."""
    print("=" * 70)
    print("RIGOROUS PREDICTION TESTS - Hard & Honest Evaluation")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        brain = load_trained_model()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run train_predictive_atl.py first to train a model.")
        return None
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
    }
    
    # Run all tests
    tests = [
        test_novel_shapes,
        test_novel_colors,
        test_novel_motion_patterns,
        test_long_horizon_prediction,
        test_stationary_vs_moving,
        test_velocity_sensitivity,
    ]
    
    passed_count = 0
    for test_fn in tests:
        result = test_fn(brain)
        results['tests'][result['test']] = result
        if result['passed']:
            passed_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF HARD TESTS")
    print("=" * 70)
    
    for test_name, result in results['tests'].items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed_count}/{len(tests)} tests passed")
    
    overall_pass = passed_count >= 4  # At least 4/6 tests
    print(f"\n  Overall: {'PASS ✓' if overall_pass else 'NEEDS IMPROVEMENT'}")
    
    if overall_pass:
        print("\n  The model shows genuine temporal prediction capabilities!")
    else:
        print("\n  The model needs improvement for robust temporal prediction.")
    
    # Save results
    results_dir = '../prediction_results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'hard_tests_{timestamp}.json')
    
    # Convert for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\n  Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_all_hard_tests()
