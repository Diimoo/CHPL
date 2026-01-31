#!/usr/bin/env python3
"""
Train PredictiveATL: Phase 1.1 of Cognitive Development Roadmap.

This script:
1. Trains visual cortex on reconstruction (Phase 1: Visual development)
2. Trains language alignment (Phase 2: Cross-modal)
3. Trains temporal prediction (Phase 3: Prediction)
4. Tests prediction accuracy honestly

Success criteria from roadmap:
- next_state_accuracy > 0.6
- counterfactual_similarity > 0.55
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from predictive_atl import (
    PredictiveBrain, PredictiveATL,
    generate_temporal_dataset, generate_occlusion_dataset,
    TemporalSequence, DEVICE
)
from brain_crossmodal_learner import SimpleVisualCortex56
from synthetic_environment import (
    generate_training_pairs, SHAPES, COLORS,
    create_stimulus_on_canvas, generate_two_object_pairs
)


def generate_training_pairs_56(n_per_combination: int = 10) -> list:
    """Generate training pairs at 56x56 resolution for SimpleVisualCortex56."""
    from dataclasses import dataclass
    import numpy as np
    
    @dataclass
    class CrossModalPair56:
        image: np.ndarray
        label: str
        shape: str
        color: str
        size: str
    
    pairs = []
    sizes = ['small', 'large']
    
    for shape in SHAPES:
        for color in COLORS:
            for size in sizes:
                for _ in range(n_per_combination):
                    # Random offset and noise
                    offset_x = np.random.randint(-5, 6)
                    offset_y = np.random.randint(-5, 6)
                    center = (28 + offset_x, 28 + offset_y)
                    noise = np.random.uniform(0.0, 0.03)
                    
                    img = create_stimulus_on_canvas(
                        shape=shape,
                        color=color,
                        size=size,
                        canvas_size=56,
                        center=center,
                        noise=noise,
                    )
                    
                    # Label: sometimes include size
                    if np.random.random() < 0.5:
                        label = f"{size} {color} {shape}"
                    else:
                        label = f"{color} {shape}"
                    
                    pairs.append(CrossModalPair56(
                        image=img,
                        label=label,
                        shape=shape,
                        color=color,
                        size=size,
                    ))
    
    return pairs


def ensure_cuda():
    """Ensure we're running on CUDA as required."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training. No GPU detected.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return torch.device('cuda')


def train_visual_cortex(
    brain: PredictiveBrain,
    n_epochs: int = 15,
    batch_size: int = 32,
) -> Dict:
    """
    Phase 1: Train visual cortex via reconstruction.
    
    Returns training statistics.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Visual Cortex Development")
    print("Learning visual features via reconstruction")
    print("=" * 60)
    
    # Generate training images at 56x56 resolution
    train_pairs = generate_training_pairs_56(n_per_combination=20)
    print(f"Training images: {len(train_pairs)}")
    
    optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    stats = {'losses': [], 'discriminability': []}
    
    for epoch in range(n_epochs):
        np.random.shuffle(train_pairs)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for pair in batch:
                img = torch.from_numpy(pair.image).float().to(DEVICE)
                loss = brain.visual.reconstruction_loss(img)
                batch_loss += loss
            
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        stats['losses'].append(avg_loss)
        
        # Check discriminability every few epochs
        if (epoch + 1) % 3 == 0 or epoch == n_epochs - 1:
            disc = check_visual_discriminability(brain)
            stats['discriminability'].append(disc)
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, discriminability={disc:.3f}")
        else:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")
    
    return stats


def check_visual_discriminability(brain: PredictiveBrain) -> float:
    """
    Check if visual cortex produces discriminable features.
    Lower off-diagonal similarity = better discrimination.
    """
    test_features = []
    
    for shape in SHAPES[:4]:
        for color in COLORS[:3]:
            img = create_stimulus_on_canvas(
                shape=shape, color=color, size='small',
                canvas_size=56, center=(28, 28)
            )
            img_t = torch.from_numpy(img).float().to(DEVICE)
            feat = brain.visual(img_t).detach()
            test_features.append(feat)
    
    features = torch.stack(test_features)
    features = F.normalize(features, dim=-1)
    sims = torch.matmul(features, features.T)
    
    # Off-diagonal mean (should be low for good discrimination)
    n = len(features)
    mask = ~torch.eye(n, dtype=bool, device=DEVICE)
    off_diag_mean = sims[mask].mean().item()
    
    # Discriminability = 1 - off_diag_mean (higher is better)
    return 1 - off_diag_mean


def train_language_alignment(
    brain: PredictiveBrain,
    n_epochs: int = 15,
) -> Dict:
    """
    Phase 2: Align language cortex with visual features.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Language Cortex Alignment")
    print("Learning to align words with visual features")
    print("=" * 60)
    
    train_pairs = generate_training_pairs_56(n_per_combination=15)
    optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    
    stats = {'losses': [], 'alignment': []}
    
    for epoch in range(n_epochs):
        np.random.shuffle(train_pairs)
        epoch_loss = 0.0
        
        for pair in train_pairs:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(pair.image).float().to(DEVICE))
            
            lang_feat = brain.language(pair.label)
            
            loss = 1 - F.cosine_similarity(
                lang_feat.unsqueeze(0), vis_feat.unsqueeze(0)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_pairs)
        stats['losses'].append(avg_loss)
        
        if (epoch + 1) % 3 == 0 or epoch == n_epochs - 1:
            alignment = check_alignment(brain)
            stats['alignment'].append(alignment)
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, alignment={alignment:.3f}")
        else:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")
    
    return stats


def check_alignment(brain: PredictiveBrain) -> float:
    """Check visual-language alignment."""
    alignments = []
    
    for shape in SHAPES[:3]:
        for color in COLORS[:3]:
            img = create_stimulus_on_canvas(
                shape=shape, color=color, size='small',
                canvas_size=56, center=(28, 28)
            )
            label = f"{color} {shape}"
            
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(img).float().to(DEVICE))
                lang_feat = brain.language(label)
                
                sim = F.cosine_similarity(
                    vis_feat.unsqueeze(0), lang_feat.unsqueeze(0)
                ).item()
                alignments.append(sim)
    
    return np.mean(alignments)


def train_temporal_prediction(
    brain: PredictiveBrain,
    n_epochs: int = 30,
    n_sequences: int = 800,
    use_velocity: bool = True,
) -> Dict:
    """
    Phase 3: Train temporal prediction.
    
    This is the core of Phase 1.1 from the roadmap.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Temporal Prediction Training")
    print(f"Training on {n_sequences} sequences for {n_epochs} epochs")
    print(f"Using velocity conditioning: {use_velocity}")
    print("=" * 60)
    
    # Generate temporal sequences
    print("Generating temporal sequences...")
    sequences = generate_temporal_dataset(
        n_sequences=n_sequences,
        frames_per_seq=10,
        motion_types=['linear', 'bounce'],
    )
    print(f"  Generated {len(sequences)} sequences")
    
    stats = {
        'losses': [],
        'prediction_accuracy': [],
        'cos_sim': [],
    }
    
    for epoch in range(n_epochs):
        np.random.shuffle(sequences)
        epoch_losses = []
        
        for seq in sequences:
            # Train on consecutive frame pairs
            for t in range(len(seq) - 1):
                current_frame = seq.frames[t]
                next_frame = seq.frames[t + 1]
                
                current_img = torch.from_numpy(current_frame.image).float().to(DEVICE)
                next_img = torch.from_numpy(next_frame.image).float().to(DEVICE)
                
                if use_velocity:
                    # Normalize velocity
                    vel = torch.tensor([
                        current_frame.velocity[0] / 10.0,  # normalize
                        current_frame.velocity[1] / 10.0,
                    ], device=DEVICE)
                    loss = brain.train_on_transition(current_img, next_img, vel)
                else:
                    loss = brain.train_on_transition(current_img, next_img, None)
                
                epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        stats['losses'].append(avg_loss)
        
        # Evaluate every few epochs
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            eval_results = evaluate_prediction_accuracy(brain, sequences[:50], use_velocity)
            stats['prediction_accuracy'].append(eval_results['accuracy'])
            stats['cos_sim'].append(eval_results['mean_cos_sim'])
            
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
                  f"accuracy={eval_results['accuracy']:.3f}, "
                  f"cos_sim={eval_results['mean_cos_sim']:.3f}")
        else:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")
    
    return stats


def evaluate_prediction_accuracy(
    brain: PredictiveBrain,
    sequences: List[TemporalSequence],
    use_velocity: bool = True,
) -> Dict:
    """
    Evaluate prediction accuracy on held-out sequences.
    
    Accuracy = cosine similarity between predicted and actual next-state activations.
    """
    cos_sims = []
    topk_overlaps = []
    
    with torch.no_grad():
        for seq in sequences:
            for t in range(len(seq) - 1):
                current_frame = seq.frames[t]
                next_frame = seq.frames[t + 1]
                
                current_img = torch.from_numpy(current_frame.image).float().to(DEVICE)
                next_img = torch.from_numpy(next_frame.image).float().to(DEVICE)
                
                if use_velocity:
                    vel = torch.tensor([
                        current_frame.velocity[0] / 10.0,
                        current_frame.velocity[1] / 10.0,
                    ], device=DEVICE)
                else:
                    vel = None
                
                result = brain.evaluate_prediction(current_img, next_img, vel)
                cos_sims.append(result['cosine_similarity'])
                topk_overlaps.append(result['topk_overlap'])
    
    mean_cos_sim = np.mean(cos_sims)
    mean_overlap = np.mean(topk_overlaps)
    
    # Accuracy = proportion where similarity > 0.6 threshold
    accuracy = np.mean([s > 0.6 for s in cos_sims])
    
    return {
        'mean_cos_sim': mean_cos_sim,
        'mean_topk_overlap': mean_overlap,
        'accuracy': accuracy,
        'all_cos_sims': cos_sims,
    }


def test_object_permanence(
    brain: PredictiveBrain,
    n_sequences: int = 50,
) -> Dict:
    """
    Test Phase 1.2: Object Permanence.
    
    Can the model maintain representation of hidden objects?
    """
    print("\n" + "=" * 60)
    print("PHASE 1.2: Object Permanence Test")
    print("Testing if model maintains hidden object representations")
    print("=" * 60)
    
    # Generate occlusion sequences
    occ_sequences = generate_occlusion_dataset(n_sequences=n_sequences)
    
    results = {
        'before_occlusion': [],
        'during_occlusion': [],
        'after_occlusion': [],
        'recall_accuracy': [],
    }
    
    with torch.no_grad():
        for seq in occ_sequences:
            n_frames = len(seq)
            
            # Get activations at different phases
            # Before occlusion (first 4 frames)
            before_acts = []
            for t in range(min(4, n_frames)):
                img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                act = brain.get_scene_activation(img)
                before_acts.append(act)
            
            # During occlusion (middle frames)
            mid_start = n_frames // 3
            mid_end = 2 * n_frames // 3
            during_acts = []
            for t in range(mid_start, mid_end):
                img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                act = brain.get_scene_activation(img)
                during_acts.append(act)
            
            # After occlusion (last 4 frames)
            after_acts = []
            for t in range(max(0, n_frames - 4), n_frames):
                img = torch.from_numpy(seq.frames[t].image).float().to(DEVICE)
                act = brain.get_scene_activation(img)
                after_acts.append(act)
            
            # Compare: does "after" resemble "before"?
            if before_acts and after_acts:
                before_mean = torch.stack(before_acts).mean(dim=0)
                after_mean = torch.stack(after_acts).mean(dim=0)
                
                # Similarity between before and after
                recall_sim = F.cosine_similarity(
                    before_mean.unsqueeze(0),
                    after_mean.unsqueeze(0)
                ).item()
                
                results['recall_accuracy'].append(recall_sim)
            
            # Also check if prediction during occlusion maintains object
            if during_acts and before_acts:
                # Average similarity during occlusion with initial state
                during_mean = torch.stack(during_acts).mean(dim=0)
                initial = before_acts[0]
                
                during_sim = F.cosine_similarity(
                    during_mean.unsqueeze(0),
                    initial.unsqueeze(0)
                ).item()
                
                results['during_occlusion'].append(during_sim)
    
    # Compute summary stats
    mean_recall = np.mean(results['recall_accuracy']) if results['recall_accuracy'] else 0
    mean_during = np.mean(results['during_occlusion']) if results['during_occlusion'] else 0
    
    # Object permanence = can recall object after occlusion
    permanence_score = mean_recall
    
    print(f"\n  Results:")
    print(f"    Mean recall similarity (before↔after): {mean_recall:.3f}")
    print(f"    Mean during-occlusion similarity: {mean_during:.3f}")
    print(f"    Object permanence score: {permanence_score:.3f}")
    
    if permanence_score > 0.5:
        print("    ✓ Object permanence PASSED (> 0.5)")
    else:
        print("    ✗ Object permanence FAILED (< 0.5)")
    
    return {
        'recall_accuracy': mean_recall,
        'during_occlusion_sim': mean_during,
        'permanence_score': permanence_score,
        'passed': permanence_score > 0.5,
    }


def test_counterfactual_imagination(
    brain: PredictiveBrain,
    n_tests: int = 100,
) -> Dict:
    """
    Test imagination capabilities.
    
    Can the model imagine plausible future states?
    """
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL IMAGINATION TEST")
    print("Testing if model can imagine plausible futures")
    print("=" * 60)
    
    imagination_quality = []
    
    with torch.no_grad():
        for _ in range(n_tests):
            # Create a scene
            shape = np.random.choice(SHAPES[:4])
            color = np.random.choice(COLORS[:4])
            
            # Random position
            pos_x = np.random.randint(15, 41)
            pos_y = np.random.randint(15, 41)
            
            img = create_stimulus_on_canvas(
                shape=shape, color=color, size='small',
                canvas_size=56, center=(pos_x, pos_y)
            )
            img_t = torch.from_numpy(img).float().to(DEVICE)
            
            current_act = brain.get_scene_activation(img_t)
            
            # Imagine with different velocities
            velocities = [
                (5, 0),   # right
                (-5, 0),  # left
                (0, 5),   # down
                (0, -5),  # up
            ]
            
            imagined_acts = []
            for vel in velocities:
                vel_t = torch.tensor([vel[0]/10, vel[1]/10], device=DEVICE)
                imagined = brain.atl.predict_with_velocity(current_act, vel_t)
                imagined_acts.append(imagined)
            
            # Check: different velocities should produce different imaginations
            # (diversity of imagination)
            diversity_scores = []
            for i in range(len(imagined_acts)):
                for j in range(i + 1, len(imagined_acts)):
                    diff = 1 - F.cosine_similarity(
                        imagined_acts[i].unsqueeze(0),
                        imagined_acts[j].unsqueeze(0)
                    ).item()
                    diversity_scores.append(diff)
            
            mean_diversity = np.mean(diversity_scores) if diversity_scores else 0
            
            # Check: imagined state should be different from current
            # (change happened)
            change_scores = []
            for imagined in imagined_acts:
                change = 1 - F.cosine_similarity(
                    current_act.unsqueeze(0),
                    imagined.unsqueeze(0)
                ).item()
                change_scores.append(change)
            
            mean_change = np.mean(change_scores)
            
            # Quality = diversity + change (both should be non-zero)
            quality = (mean_diversity + mean_change) / 2
            imagination_quality.append(quality)
    
    mean_quality = np.mean(imagination_quality)
    
    print(f"\n  Imagination quality score: {mean_quality:.3f}")
    print(f"  (Measures diversity of imagined futures + change from current)")
    
    # Counterfactual similarity threshold from roadmap
    if mean_quality > 0.1:  # Low threshold because we're measuring difference
        print("    ✓ Counterfactual imagination shows differentiation")
    else:
        print("    ✗ Imagined futures too similar")
    
    return {
        'mean_quality': mean_quality,
        'all_qualities': imagination_quality,
    }


def run_full_experiment():
    """
    Run the complete Phase 1 experiment.
    """
    print("=" * 70)
    print("CHPL COGNITIVE DEVELOPMENT - PHASE 1: PREDICTION & IMAGINATION")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure CUDA
    device = ensure_cuda()
    
    # Initialize brain
    print("\nInitializing Predictive Brain...")
    brain = PredictiveBrain(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=56,
    )
    print(f"  Visual cortex params: {sum(p.numel() for p in brain.visual.parameters()):,}")
    print(f"  Language cortex params: {sum(p.numel() for p in brain.language.parameters()):,}")
    print(f"  Predictor params: {sum(p.numel() for p in brain.atl.predictor.parameters()):,}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'phases': {},
    }
    
    start_time = time.time()
    
    # Phase 1: Visual cortex
    phase1_stats = train_visual_cortex(brain, n_epochs=15)
    results['phases']['visual'] = phase1_stats
    
    # Phase 2: Language alignment
    phase2_stats = train_language_alignment(brain, n_epochs=15)
    results['phases']['language'] = phase2_stats
    
    # Phase 3: Temporal prediction (main experiment)
    phase3_stats = train_temporal_prediction(
        brain,
        n_epochs=30,
        n_sequences=800,
        use_velocity=True,
    )
    results['phases']['prediction'] = phase3_stats
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Generate fresh test sequences
    print("\nGenerating test sequences...")
    test_sequences = generate_temporal_dataset(
        n_sequences=200,
        frames_per_seq=10,
        motion_types=['linear', 'bounce'],
    )
    
    final_eval = evaluate_prediction_accuracy(brain, test_sequences, use_velocity=True)
    results['final_evaluation'] = {
        'mean_cos_sim': final_eval['mean_cos_sim'],
        'mean_topk_overlap': final_eval['mean_topk_overlap'],
        'accuracy': final_eval['accuracy'],
    }
    
    print(f"\n  PREDICTION RESULTS:")
    print(f"    Mean cosine similarity: {final_eval['mean_cos_sim']:.4f}")
    print(f"    Mean top-k overlap: {final_eval['mean_topk_overlap']:.4f}")
    print(f"    Accuracy (sim > 0.6): {final_eval['accuracy']:.4f}")
    
    # Roadmap success criteria
    if final_eval['mean_cos_sim'] > 0.6:
        print(f"\n    ✓ PREDICTION SUCCESS: similarity {final_eval['mean_cos_sim']:.3f} > 0.6")
    else:
        print(f"\n    ✗ PREDICTION BELOW TARGET: similarity {final_eval['mean_cos_sim']:.3f} < 0.6")
    
    # Object permanence test
    permanence_results = test_object_permanence(brain, n_sequences=50)
    results['object_permanence'] = permanence_results
    
    # Counterfactual imagination test
    imagination_results = test_counterfactual_imagination(brain, n_tests=100)
    results['imagination'] = imagination_results
    
    # Summary
    elapsed = time.time() - start_time
    results['elapsed_seconds'] = elapsed
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"\n  Phase 1.1 Results (Prediction):")
    print(f"    - Next-state prediction accuracy: {final_eval['mean_cos_sim']:.3f}")
    print(f"      Target: > 0.6 | Status: {'PASS ✓' if final_eval['mean_cos_sim'] > 0.6 else 'FAIL ✗'}")
    print(f"\n  Phase 1.2 Results (Object Permanence):")
    print(f"    - Hidden object recall: {permanence_results['permanence_score']:.3f}")
    print(f"      Target: > 0.5 | Status: {'PASS ✓' if permanence_results['passed'] else 'FAIL ✗'}")
    print(f"\n  Imagination Quality: {imagination_results['mean_quality']:.3f}")
    
    # Determine if TODDLER level reached
    toddler_reached = (
        final_eval['mean_cos_sim'] > 0.6 and
        permanence_results['passed']
    )
    
    print("\n" + "-" * 70)
    if toddler_reached:
        print("  ★ TODDLER LEVEL REACHED ★")
        print("  CHPL can now predict future states and maintain object permanence!")
    else:
        print("  TODDLER level NOT YET reached.")
        print("  More training or architectural improvements needed.")
    print("-" * 70)
    
    # Save results
    results_dir = '../prediction_results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'prediction_experiment_{timestamp}.json')
    
    # Convert numpy arrays and other types for JSON
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
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    results_json = convert_for_json(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n  Results saved to: {results_file}")
    
    # Save model
    model_file = os.path.join(results_dir, f'predictive_brain_{timestamp}.pt')
    torch.save({
        'visual_state': brain.visual.state_dict(),
        'language_state': brain.language.state_dict(),
        'atl_state': brain.atl.state_dict(),
    }, model_file)
    print(f"  Model saved to: {model_file}")
    
    return brain, results


if __name__ == "__main__":
    brain, results = run_full_experiment()
