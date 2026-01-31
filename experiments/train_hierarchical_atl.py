#!/usr/bin/env python3
"""
Phase 4: Abstraction & Creativity - Training Script

Trains AbstractBrain on:
1. Hierarchical concept abstraction
2. Analogical reasoning (A:B :: C:?)
3. Creative generation
4. Few-shot concept learning

This is the final phase of cognitive development.

Success Criteria (from roadmap):
- Analogy accuracy > 0.6
- Few-shot recognition > 0.7
- Creative diversity > 0.3
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

from hierarchical_atl import (
    AbstractBrain, HierarchicalATL,
    create_analogy_dataset, create_few_shot_dataset,
)
from language_atl import VOCAB, tokenize, DEVICE
from synthetic_environment import create_stimulus_on_canvas


def ensure_cuda():
    """Ensure we're running on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training. No GPU detected.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def find_latest_language_model(results_dir: str = '../language_results') -> str:
    """Find the most recent trained model from Phase 3."""
    model_files = [f for f in os.listdir(results_dir) 
                   if f.startswith('language_brain_') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No Phase 3 models found in {results_dir}")
    
    model_files.sort(reverse=True)
    return os.path.join(results_dir, model_files[0])


def train_hierarchical_abstraction(
    brain: AbstractBrain,
    n_epochs: int = 15,
) -> Dict:
    """
    Phase 4.1: Train hierarchical abstraction.
    
    Learn to activate concepts at multiple abstraction levels.
    """
    print("\n" + "=" * 60)
    print("PHASE 4.1: Hierarchical Abstraction Training")
    print("Learning multi-level concept representation")
    print("=" * 60)
    
    # Optimizer for abstractor networks
    optimizer = torch.optim.Adam(brain.atl.abstractors.parameters(), lr=1e-3)
    
    stats = {'losses': [], 'level_variances': []}
    
    shapes = ['circle', 'square', 'triangle', 'star']
    colors = ['red', 'blue', 'green', 'yellow']
    
    for epoch in range(n_epochs):
        epoch_losses = []
        level_vars = [[] for _ in range(brain.n_levels)]
        
        # Generate training samples
        for _ in range(200):
            shape = np.random.choice(shapes)
            color = np.random.choice(colors)
            
            img = create_stimulus_on_canvas(shape, color, 'small', 56, (28, 28))
            img_t = torch.from_numpy(img).float().permute(2, 0, 1).to(DEVICE)
            
            # Get features
            with torch.no_grad():
                features = brain.visual(img_t.unsqueeze(0)).squeeze(0)
            
            # Get hierarchical activations
            activations = brain.atl.activate_hierarchical(features)
            
            # Loss: encourage sparsity at lower levels, broader at higher
            loss = 0
            for level, act in enumerate(activations):
                # Entropy regularization: higher levels should have higher entropy
                entropy = -(act * torch.log(act + 1e-8)).sum()
                target_entropy = 0.5 + 0.3 * level  # Increase with level
                loss += (entropy - target_entropy) ** 2
                
                # Track variance
                level_vars[level].append(act.var().item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        epoch_loss = np.mean(epoch_losses)
        stats['losses'].append(epoch_loss)
        
        mean_vars = [np.mean(lv) for lv in level_vars]
        stats['level_variances'].append(mean_vars)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}")
            print(f"    Level variances: {[f'{v:.4f}' for v in mean_vars]}")
    
    return stats


def train_analogy_solving(
    brain: AbstractBrain,
    n_epochs: int = 20,
) -> Dict:
    """
    Phase 4.2: Train analogical reasoning.
    
    Learn to solve A:B :: C:? analogies.
    """
    print("\n" + "=" * 60)
    print("PHASE 4.2: Analogical Reasoning Training")
    print("Learning to solve A:B :: C:? analogies")
    print("=" * 60)
    
    # Generate training data
    train_data = create_analogy_dataset(n_samples=500)
    print(f"  Training samples: {len(train_data)}")
    
    optimizer = torch.optim.Adam(brain.atl.analogy_transform.parameters(), lr=1e-3)
    
    stats = {'losses': [], 'accuracies': []}
    
    for epoch in range(n_epochs):
        np.random.shuffle(train_data)
        epoch_losses = []
        correct = 0
        total = 0
        
        for item in train_data:
            # Get images
            A = torch.from_numpy(item['A']).float().permute(2, 0, 1).to(DEVICE)
            B = torch.from_numpy(item['B']).float().permute(2, 0, 1).to(DEVICE)
            C = torch.from_numpy(item['C']).float().permute(2, 0, 1).to(DEVICE)
            D = torch.from_numpy(item['D']).float().permute(2, 0, 1).to(DEVICE)
            
            # Get activations
            with torch.no_grad():
                act_A = brain.get_scene_activation(A)
                act_B = brain.get_scene_activation(B)
                act_C = brain.get_scene_activation(C)
                act_D_target = brain.get_scene_activation(D)
            
            # Solve analogy
            act_D_pred = brain.atl.solve_analogy(act_A, act_B, act_C)
            
            # Loss: predicted D should match target D
            loss = F.mse_loss(act_D_pred, act_D_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Accuracy: cosine similarity > 0.7 counts as correct
            sim = F.cosine_similarity(
                act_D_pred.unsqueeze(0),
                act_D_target.unsqueeze(0)
            ).item()
            
            if sim > 0.7:
                correct += 1
            total += 1
        
        epoch_loss = np.mean(epoch_losses)
        epoch_acc = correct / total if total > 0 else 0
        
        stats['losses'].append(epoch_loss)
        stats['accuracies'].append(epoch_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, accuracy={epoch_acc:.3f}")
    
    final_acc = stats['accuracies'][-1]
    print(f"\n  Final analogy accuracy: {final_acc:.3f}")
    print(f"  Target: > 0.6 | Status: {'PASS ✓' if final_acc > 0.6 else 'FAIL ✗'}")
    
    return stats


def test_analogy_solving(brain: AbstractBrain, n_test: int = 100) -> Dict:
    """Test analogical reasoning."""
    print("\n" + "=" * 60)
    print("ANALOGY SOLVING TEST")
    print("=" * 60)
    
    test_data = create_analogy_dataset(n_samples=n_test)
    
    correct = 0
    total = 0
    similarities = []
    per_type = {'color': [], 'shape': []}
    
    for item in test_data:
        A = torch.from_numpy(item['A']).float().permute(2, 0, 1).to(DEVICE)
        B = torch.from_numpy(item['B']).float().permute(2, 0, 1).to(DEVICE)
        C = torch.from_numpy(item['C']).float().permute(2, 0, 1).to(DEVICE)
        D = torch.from_numpy(item['D']).float().permute(2, 0, 1).to(DEVICE)
        
        with torch.no_grad():
            act_D_pred, _ = brain.solve_visual_analogy(A, B, C)
            act_D_target = brain.get_scene_activation(D)
        
        sim = F.cosine_similarity(
            act_D_pred.unsqueeze(0),
            act_D_target.unsqueeze(0)
        ).item()
        
        similarities.append(sim)
        per_type[item['transform_type']].append(sim)
        
        if sim > 0.7:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    mean_sim = np.mean(similarities)
    
    print(f"\n  Analogy accuracy: {accuracy:.3f}")
    print(f"  Mean similarity: {mean_sim:.3f}")
    print(f"  Target: > 0.6 | Status: {'PASS ✓' if accuracy > 0.6 else 'FAIL ✗'}")
    
    print("\n  Per-type similarity:")
    for t, sims in per_type.items():
        if sims:
            print(f"    {t}: {np.mean(sims):.3f}")
    
    return {
        'accuracy': accuracy,
        'mean_similarity': mean_sim,
        'per_type': {t: np.mean(s) if s else 0 for t, s in per_type.items()},
        'passed': accuracy > 0.6,
    }


def test_few_shot_learning(brain: AbstractBrain) -> Dict:
    """Test few-shot concept learning."""
    print("\n" + "=" * 60)
    print("FEW-SHOT LEARNING TEST")
    print("=" * 60)
    
    # Create test data
    test_data = create_few_shot_dataset(n_concepts=5, examples_per=3)
    
    results = []
    
    for concept_data in test_data:
        name = concept_data['name']
        examples = concept_data['examples']
        
        # Learn concept
        example_tensors = [
            torch.from_numpy(img).float().permute(2, 0, 1).to(DEVICE)
            for img in examples
        ]
        
        brain.learn_new_concept(example_tensors, name)
        
        # Test positive examples
        positive_correct = 0
        for img in concept_data['test_positive']:
            img_t = torch.from_numpy(img).float().permute(2, 0, 1).to(DEVICE)
            recognized = brain.recognize_in_image(img_t)
            if recognized == name:
                positive_correct += 1
        
        # Test negative examples
        negative_correct = 0
        for img in concept_data['test_negative']:
            img_t = torch.from_numpy(img).float().permute(2, 0, 1).to(DEVICE)
            recognized = brain.recognize_in_image(img_t)
            if recognized != name:
                negative_correct += 1
        
        results.append({
            'concept': name,
            'positive_acc': positive_correct / len(concept_data['test_positive']),
            'negative_acc': negative_correct / len(concept_data['test_negative']),
        })
        
        print(f"  {name}: pos={positive_correct}/{len(concept_data['test_positive'])}, neg={negative_correct}/{len(concept_data['test_negative'])}")
    
    # Overall accuracy
    mean_pos = np.mean([r['positive_acc'] for r in results])
    mean_neg = np.mean([r['negative_acc'] for r in results])
    overall = (mean_pos + mean_neg) / 2
    
    print(f"\n  Positive recognition: {mean_pos:.3f}")
    print(f"  Negative rejection: {mean_neg:.3f}")
    print(f"  Overall few-shot accuracy: {overall:.3f}")
    print(f"  Target: > 0.7 | Status: {'PASS ✓' if overall > 0.7 else 'FAIL ✗'}")
    
    return {
        'positive_accuracy': mean_pos,
        'negative_accuracy': mean_neg,
        'overall_accuracy': overall,
        'per_concept': results,
        'passed': overall > 0.7,
    }


def test_creative_generation(brain: AbstractBrain, n_samples: int = 30) -> Dict:
    """Test creative scene generation."""
    print("\n" + "=" * 60)
    print("CREATIVE GENERATION TEST")
    print("=" * 60)
    
    # Generate multiple creative outputs
    activations = []
    descriptions = []
    
    for _ in range(n_samples):
        act, desc = brain.generate_creative_scene(temperature=1.5)
        activations.append(act)
        descriptions.append(desc)
    
    # Measure diversity
    # 1. Activation diversity: pairwise distances
    act_stack = torch.stack(activations)
    pairwise_dists = []
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            dist = 1 - F.cosine_similarity(
                activations[i].unsqueeze(0),
                activations[j].unsqueeze(0)
            ).item()
            pairwise_dists.append(dist)
    
    mean_diversity = np.mean(pairwise_dists) if pairwise_dists else 0
    
    # 2. Description diversity: unique words
    all_words = set()
    for desc in descriptions:
        all_words.update(desc.lower().split())
    
    vocab_diversity = len(all_words) / (n_samples * 3)  # Normalize by expected words
    
    print(f"\n  Activation diversity: {mean_diversity:.3f}")
    print(f"  Vocabulary diversity: {vocab_diversity:.3f}")
    print(f"  Unique words used: {len(all_words)}")
    
    overall_diversity = (mean_diversity + vocab_diversity) / 2
    print(f"\n  Overall creative diversity: {overall_diversity:.3f}")
    print(f"  Target: > 0.3 | Status: {'PASS ✓' if overall_diversity > 0.3 else 'FAIL ✗'}")
    
    # Show examples
    print("\n  Sample creative descriptions:")
    for desc in descriptions[:5]:
        print(f"    '{desc}'")
    
    return {
        'activation_diversity': mean_diversity,
        'vocab_diversity': vocab_diversity,
        'overall_diversity': overall_diversity,
        'n_unique_words': len(all_words),
        'passed': overall_diversity > 0.3,
    }


def run_full_experiment():
    """Run complete Phase 4 experiment."""
    print("=" * 70)
    print("CHPL COGNITIVE DEVELOPMENT - PHASE 4: ABSTRACTION & CREATIVITY")
    print("=" * 70)
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ensure_cuda()
    
    # Initialize brain from Phase 3
    print("\nLoading Phase 3 brain...")
    model_path = find_latest_language_model()
    print(f"Found model: {model_path}")
    
    brain = AbstractBrain(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=56,
        n_levels=4,
    )
    
    # Load Phase 3 weights
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    brain.visual.load_state_dict(checkpoint['visual_state'])
    brain.generator.load_state_dict(checkpoint['generator_state'])
    brain.qa.load_state_dict(checkpoint['qa_state'])
    
    # Load ATL weights (compatible subset)
    atl_state = checkpoint['atl_state']
    current_state = brain.atl.state_dict()
    for key in atl_state:
        if key in current_state and atl_state[key].shape == current_state[key].shape:
            current_state[key] = atl_state[key]
    brain.atl.load_state_dict(current_state)
    
    print("Loaded Phase 3 weights")
    
    print(f"\nAbstract brain parameters:")
    print(f"  Hierarchical ATL: {sum(p.numel() for p in brain.atl.parameters()):,}")
    
    results = {}
    
    # Phase 4.1: Hierarchical Abstraction
    abstraction_stats = train_hierarchical_abstraction(brain, n_epochs=15)
    results['abstraction_training'] = {
        'final_loss': abstraction_stats['losses'][-1],
    }
    
    # Phase 4.2: Analogical Reasoning
    analogy_stats = train_analogy_solving(brain, n_epochs=20)
    results['analogy_training'] = {
        'final_accuracy': analogy_stats['accuracies'][-1],
    }
    
    # Testing
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    analogy_test = test_analogy_solving(brain, n_test=100)
    results['analogy_test'] = analogy_test
    
    few_shot_test = test_few_shot_learning(brain)
    results['few_shot_test'] = few_shot_test
    
    creative_test = test_creative_generation(brain, n_samples=30)
    results['creative_test'] = creative_test
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} minutes")
    
    print(f"\n  Phase 4.1 Results (Analogical Reasoning):")
    print(f"    - Accuracy: {analogy_test['accuracy']:.3f}")
    print(f"      Target: > 0.6 | Status: {'PASS ✓' if analogy_test['passed'] else 'FAIL ✗'}")
    
    print(f"\n  Phase 4.2 Results (Few-Shot Learning):")
    print(f"    - Accuracy: {few_shot_test['overall_accuracy']:.3f}")
    print(f"      Target: > 0.7 | Status: {'PASS ✓' if few_shot_test['passed'] else 'FAIL ✗'}")
    
    print(f"\n  Creative Generation:")
    print(f"    - Diversity: {creative_test['overall_diversity']:.3f}")
    print(f"      Target: > 0.3 | Status: {'PASS ✓' if creative_test['passed'] else 'FAIL ✗'}")
    
    # Overall assessment
    all_passed = analogy_test['passed'] and few_shot_test['passed'] and creative_test['passed']
    
    print("\n" + "-" * 70)
    if all_passed:
        print("  ★★★ ADVANCED CHILD LEVEL REACHED ★★★")
        print("  CHPL can now reason by analogy, learn new concepts, and create!")
    else:
        some_passed = analogy_test['passed'] or few_shot_test['passed'] or creative_test['passed']
        if some_passed:
            print("  ★ PARTIAL ADVANCED CHILD LEVEL ★")
            print("  Some abstraction capabilities achieved.")
        else:
            print("  Phase 4 needs more work.")
    print("-" * 70)
    
    # Save results
    results_dir = Path('../abstraction_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = results_dir / f'abstract_brain_{timestamp}.pt'
    torch.save({
        'visual_state': brain.visual.state_dict(),
        'language_state': brain.language.state_dict(),
        'atl_state': brain.atl.state_dict(),
        'generator_state': brain.generator.state_dict(),
        'qa_state': brain.qa.state_dict(),
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
    
    results_path = results_dir / f'abstraction_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    return brain, results


if __name__ == "__main__":
    brain, results = run_full_experiment()
