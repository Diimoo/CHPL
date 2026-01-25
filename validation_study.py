"""
Validation Study: Multi-Seed Statistical Validation

Scientific goal: Demonstrate that findings are robust across random initializations.

Run this overnight:
    python validation_study.py

Expected time: ~5-10 hours on RTX 4080 (10 seeds × 4 conditions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import scipy.stats as stats

from brain_crossmodal_learner import (
    BrainCrossModalLearner, SimpleVisualCortex, SimpleLanguageCortex,
    Hippocampus, ATL, DEVICE
)
from synthetic_environment import (
    create_stimulus, generate_training_pairs,
    SHAPES, COLORS, SIZES
)

# ============================================================
# CONFIGURATION
# ============================================================

N_SEEDS = 10
SAVE_DIR = Path("validation_results")
SAVE_DIR.mkdir(exist_ok=True)

# Training hyperparameters (same as ablation study)
N_EPOCHS_VISUAL = 10
N_EPOCHS_LANG = 15
N_EPOCHS_BINDING = 10


# ============================================================
# METRICS (copied from ablation_study.py for consistency)
# ============================================================

def measure_color_shape_bias(brain: BrainCrossModalLearner) -> dict:
    """Measure color vs shape bias in learned representations."""
    same_color_diff_shape = []
    diff_color_same_shape = []
    
    test_shapes = SHAPES[:4]
    test_colors = COLORS[:4]
    
    for shape1 in test_shapes:
        for shape2 in test_shapes:
            if shape1 == shape2:
                continue
            for color in test_colors:
                img1 = create_stimulus(shape1, color, 'small')
                img2 = create_stimulus(shape2, color, 'small')
                
                feat1 = brain.visual(torch.from_numpy(img1).float())
                feat2 = brain.visual(torch.from_numpy(img2).float())
                
                sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
                same_color_diff_shape.append(sim)
    
    for color1 in test_colors:
        for color2 in test_colors:
            if color1 == color2:
                continue
            for shape in test_shapes:
                img1 = create_stimulus(shape, color1, 'small')
                img2 = create_stimulus(shape, color2, 'small')
                
                feat1 = brain.visual(torch.from_numpy(img1).float())
                feat2 = brain.visual(torch.from_numpy(img2).float())
                
                sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
                diff_color_same_shape.append(sim)
    
    avg_same_color = np.mean(same_color_diff_shape)
    avg_diff_color = np.mean(diff_color_same_shape)
    bias_score = avg_same_color / (avg_diff_color + 1e-8)
    
    return {
        'bias_score': bias_score,
        'same_color_diff_shape': avg_same_color,
        'diff_color_same_shape': avg_diff_color,
    }


def measure_binding_rate(brain: BrainCrossModalLearner) -> float:
    """Measure cross-modal binding rate."""
    aligned = 0
    total = 0
    
    for shape in SHAPES[:4]:
        for color in COLORS[:4]:
            img = create_stimulus(shape, color, 'small')
            label = f'{color} {shape}'
            
            vis_feat = brain.visual(torch.from_numpy(img).float())
            lang_feat = brain.language(label)
            
            _, vis_concept = brain.atl.activate(vis_feat, 'visual')
            _, lang_concept = brain.atl.activate(lang_feat, 'language')
            
            total += 1
            if vis_concept == lang_concept:
                aligned += 1
    
    return aligned / total * 100


def count_active_concepts(brain: BrainCrossModalLearner) -> int:
    """Count number of distinct concepts used."""
    concepts_used = set()
    
    for shape in SHAPES:
        for color in COLORS:
            img = create_stimulus(shape, color, 'small')
            vis_feat = brain.visual(torch.from_numpy(img).float())
            _, concept = brain.atl.activate(vis_feat, 'visual')
            concepts_used.add(concept)
    
    return len(concepts_used)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_full_model(train_pairs):
    """Train full brain-like model."""
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    
    bias_trajectory = []
    
    # Phase 1: Visual cortex
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(N_EPOCHS_VISUAL):
        np.random.shuffle(train_pairs)
        for pair in train_pairs[:500]:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(pair.image).float())
            loss.backward()
            vis_optimizer.step()
        
        # Track trajectory
        bias = measure_color_shape_bias(brain)
        bias_trajectory.append(bias['bias_score'])
    
    # Phase 2: Language alignment
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(N_EPOCHS_LANG):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(pair.image).float())
            lang_feat = brain.language(pair.label)
            loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
            lang_optimizer.zero_grad()
            loss.backward()
            lang_optimizer.step()
    
    # Phase 3: Cross-modal binding
    for epoch in range(N_EPOCHS_BINDING):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            brain.experience(
                torch.from_numpy(pair.image).float(),
                pair.label,
                consolidate=True,
                train_cortices=False
            )
    
    return brain, bias_trajectory


def train_no_reconstruction(train_pairs):
    """Ablation: No reconstruction (random visual features)."""
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    
    # Skip Phase 1 - visual cortex stays random
    
    # Phase 2: Language alignment
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(N_EPOCHS_LANG):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(pair.image).float())
            lang_feat = brain.language(pair.label)
            loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
            lang_optimizer.zero_grad()
            loss.backward()
            lang_optimizer.step()
    
    # Phase 3: Cross-modal binding
    for epoch in range(N_EPOCHS_BINDING):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            brain.experience(
                torch.from_numpy(pair.image).float(),
                pair.label,
                consolidate=True,
                train_cortices=False
            )
    
    return brain


def train_no_consolidation(train_pairs):
    """Ablation: No ATL consolidation."""
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    
    # Phase 1: Visual cortex
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(N_EPOCHS_VISUAL):
        np.random.shuffle(train_pairs)
        for pair in train_pairs[:500]:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(pair.image).float())
            loss.backward()
            vis_optimizer.step()
    
    # Phase 2: Language alignment
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(N_EPOCHS_LANG):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(pair.image).float())
            lang_feat = brain.language(pair.label)
            loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
            lang_optimizer.zero_grad()
            loss.backward()
            lang_optimizer.step()
    
    # Phase 3: Cross-modal binding WITHOUT consolidation
    for epoch in range(N_EPOCHS_BINDING):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            brain.experience(
                torch.from_numpy(pair.image).float(),
                pair.label,
                consolidate=False,  # KEY DIFFERENCE
                train_cortices=False
            )
    
    return brain


def train_backprop(train_pairs):
    """Backprop baseline with supervised classification."""
    
    class BackpropClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 3 * 3, 64),
            )
            self.color_classifier = nn.Linear(64, len(COLORS))
            self.shape_classifier = nn.Linear(64, len(SHAPES))
            
        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            features = self.encoder(x)
            return features, self.color_classifier(features), self.shape_classifier(features)
        
        def get_features(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            return self.encoder(x)
    
    model = BackpropClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    color_to_idx = {c: i for i, c in enumerate(COLORS)}
    shape_to_idx = {s: i for i, s in enumerate(SHAPES)}
    
    for epoch in range(25):
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            parts = pair.label.split()
            if len(parts) == 2:
                color, shape = parts
            else:
                color, shape = parts[1], parts[2]
            
            if color not in color_to_idx or shape not in shape_to_idx:
                continue
                
            color_idx = torch.tensor([color_to_idx[color]], device=DEVICE)
            shape_idx = torch.tensor([shape_to_idx[shape]], device=DEVICE)
            
            img = torch.from_numpy(pair.image).float().to(DEVICE)
            
            optimizer.zero_grad()
            features, color_pred, shape_pred = model(img)
            
            loss = F.cross_entropy(color_pred, color_idx) + F.cross_entropy(shape_pred, shape_idx)
            loss.backward()
            optimizer.step()
    
    return model


def measure_backprop_bias(model) -> dict:
    """Measure bias in backprop model."""
    same_color_diff_shape = []
    diff_color_same_shape = []
    
    test_shapes = SHAPES[:4]
    test_colors = COLORS[:4]
    
    for shape1 in test_shapes:
        for shape2 in test_shapes:
            if shape1 == shape2:
                continue
            for color in test_colors:
                img1 = torch.from_numpy(create_stimulus(shape1, color, 'small')).float().to(DEVICE)
                img2 = torch.from_numpy(create_stimulus(shape2, color, 'small')).float().to(DEVICE)
                
                with torch.no_grad():
                    feat1 = model.get_features(img1)
                    feat2 = model.get_features(img2)
                
                sim = F.cosine_similarity(feat1, feat2).item()
                same_color_diff_shape.append(sim)
    
    for color1 in test_colors:
        for color2 in test_colors:
            if color1 == color2:
                continue
            for shape in test_shapes:
                img1 = torch.from_numpy(create_stimulus(shape, color1, 'small')).float().to(DEVICE)
                img2 = torch.from_numpy(create_stimulus(shape, color2, 'small')).float().to(DEVICE)
                
                with torch.no_grad():
                    feat1 = model.get_features(img1)
                    feat2 = model.get_features(img2)
                
                sim = F.cosine_similarity(feat1, feat2).item()
                diff_color_same_shape.append(sim)
    
    avg_same_color = np.mean(same_color_diff_shape)
    avg_diff_color = np.mean(diff_color_same_shape)
    bias_score = avg_same_color / (avg_diff_color + 1e-8)
    
    return {'bias_score': bias_score}


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_single_seed(seed: int) -> dict:
    """Run complete experiment for one seed."""
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Generate training data (same for all conditions within seed)
    train_pairs = generate_training_pairs(n_per_combination=20)
    
    results = {
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'conditions': {}
    }
    
    # 1. Full model
    print(f"  [Seed {seed}] Training FULL model...")
    start = time.time()
    brain_full, trajectory = train_full_model(train_pairs)
    
    bias = measure_color_shape_bias(brain_full)
    results['conditions']['full'] = {
        'bias_score': bias['bias_score'],
        'same_color_diff_shape': bias['same_color_diff_shape'],
        'diff_color_same_shape': bias['diff_color_same_shape'],
        'binding_rate': measure_binding_rate(brain_full),
        'n_concepts': count_active_concepts(brain_full),
        'trajectory': trajectory,
        'time_seconds': time.time() - start,
    }
    print(f"    Bias: {bias['bias_score']:.3f}, Binding: {results['conditions']['full']['binding_rate']:.1f}%")
    
    # 2. No reconstruction
    print(f"  [Seed {seed}] Training NO RECONSTRUCTION...")
    start = time.time()
    brain_no_recon = train_no_reconstruction(train_pairs)
    
    bias = measure_color_shape_bias(brain_no_recon)
    results['conditions']['no_reconstruction'] = {
        'bias_score': bias['bias_score'],
        'same_color_diff_shape': bias['same_color_diff_shape'],
        'diff_color_same_shape': bias['diff_color_same_shape'],
        'binding_rate': measure_binding_rate(brain_no_recon),
        'n_concepts': count_active_concepts(brain_no_recon),
        'time_seconds': time.time() - start,
    }
    print(f"    Bias: {bias['bias_score']:.3f}, Binding: {results['conditions']['no_reconstruction']['binding_rate']:.1f}%")
    
    # 3. No consolidation
    print(f"  [Seed {seed}] Training NO CONSOLIDATION...")
    start = time.time()
    brain_no_consol = train_no_consolidation(train_pairs)
    
    bias = measure_color_shape_bias(brain_no_consol)
    results['conditions']['no_consolidation'] = {
        'bias_score': bias['bias_score'],
        'same_color_diff_shape': bias['same_color_diff_shape'],
        'diff_color_same_shape': bias['diff_color_same_shape'],
        'binding_rate': measure_binding_rate(brain_no_consol),
        'n_concepts': count_active_concepts(brain_no_consol),
        'time_seconds': time.time() - start,
    }
    print(f"    Bias: {bias['bias_score']:.3f}, Binding: {results['conditions']['no_consolidation']['binding_rate']:.1f}%")
    
    # 4. Backprop baseline
    print(f"  [Seed {seed}] Training BACKPROP baseline...")
    start = time.time()
    backprop_model = train_backprop(train_pairs)
    
    bias = measure_backprop_bias(backprop_model)
    results['conditions']['backprop'] = {
        'bias_score': bias['bias_score'],
        'time_seconds': time.time() - start,
    }
    print(f"    Bias: {bias['bias_score']:.3f}")
    
    return results


def aggregate_and_analyze(all_results: list) -> dict:
    """Compute statistics and run significance tests."""
    
    conditions = ['full', 'no_reconstruction', 'no_consolidation', 'backprop']
    
    summary = {'conditions': {}, 'tests': {}}
    
    for cond in conditions:
        biases = [r['conditions'][cond]['bias_score'] for r in all_results]
        
        summary['conditions'][cond] = {
            'bias': {
                'mean': float(np.mean(biases)),
                'std': float(np.std(biases)),
                'values': biases,
            }
        }
        
        if cond != 'backprop':
            bindings = [r['conditions'][cond]['binding_rate'] for r in all_results]
            n_concepts = [r['conditions'][cond]['n_concepts'] for r in all_results]
            
            summary['conditions'][cond]['binding'] = {
                'mean': float(np.mean(bindings)),
                'std': float(np.std(bindings)),
                'values': bindings,
            }
            summary['conditions'][cond]['n_concepts'] = {
                'mean': float(np.mean(n_concepts)),
                'std': float(np.std(n_concepts)),
                'values': n_concepts,
            }
    
    # Statistical tests
    full_biases = summary['conditions']['full']['bias']['values']
    backprop_biases = summary['conditions']['backprop']['bias']['values']
    
    # Test 1: Brain-like vs Backprop bias
    t_stat, p_value = stats.ttest_ind(full_biases, backprop_biases)
    summary['tests']['full_vs_backprop_bias'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }
    
    # Test 2: Consolidation effect on binding
    full_binding = summary['conditions']['full']['binding']['values']
    no_consol_binding = summary['conditions']['no_consolidation']['binding']['values']
    
    t_stat, p_value = stats.ttest_rel(full_binding, no_consol_binding)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(full_binding)**2 + np.std(no_consol_binding)**2) / 2)
    cohens_d = (np.mean(full_binding) - np.mean(no_consol_binding)) / pooled_std
    
    summary['tests']['consolidation_effect'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'effect_size': 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small',
    }
    
    return summary


def print_results(summary: dict):
    """Print publication-ready results."""
    
    print("\n" + "="*70)
    print("PUBLICATION-READY RESULTS")
    print("="*70)
    
    print("\n{:<20} {:>15} {:>15} {:>12}".format(
        "Condition", "Bias Score", "Binding Rate", "Concepts"
    ))
    print("-"*62)
    
    for cond in ['full', 'no_reconstruction', 'no_consolidation', 'backprop']:
        bias_mean = summary['conditions'][cond]['bias']['mean']
        bias_std = summary['conditions'][cond]['bias']['std']
        
        if cond != 'backprop':
            bind_mean = summary['conditions'][cond]['binding']['mean']
            bind_std = summary['conditions'][cond]['binding']['std']
            concepts = summary['conditions'][cond]['n_concepts']['mean']
            
            print("{:<20} {:>6.3f} ± {:<6.3f} {:>5.1f}% ± {:<5.1f}% {:>8.1f}".format(
                cond, bias_mean, bias_std, bind_mean, bind_std, concepts
            ))
        else:
            print("{:<20} {:>6.3f} ± {:<6.3f} {:>15} {:>12}".format(
                cond, bias_mean, bias_std, "N/A", "N/A"
            ))
    
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)
    
    test1 = summary['tests']['full_vs_backprop_bias']
    print(f"\nTest 1: Brain-like vs Backprop color bias")
    print(f"  t-statistic: {test1['t_statistic']:.3f}")
    print(f"  p-value: {test1['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {'YES ✓' if test1['significant'] else 'NO'}")
    
    test2 = summary['tests']['consolidation_effect']
    print(f"\nTest 2: Consolidation effect on binding")
    print(f"  t-statistic: {test2['t_statistic']:.3f}")
    print(f"  p-value: {test2['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {'YES ✓' if test2['significant'] else 'NO'}")
    print(f"  Cohen's d: {test2['cohens_d']:.3f} ({test2['effect_size']})")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("VALIDATION STUDY: Multi-Seed Statistical Validation")
    print(f"Seeds: {N_SEEDS}")
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    all_results = []
    
    for seed in range(N_SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{N_SEEDS}")
        print(f"{'='*70}")
        
        result = run_single_seed(seed)
        all_results.append(result)
        
        # Save checkpoint after each seed
        checkpoint = {
            'results': all_results,
            'n_completed': len(all_results),
            'n_total': N_SEEDS,
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(checkpoint, SAVE_DIR / "checkpoint.pt")
        
        # Also save as JSON for easy inspection
        with open(SAVE_DIR / "checkpoint.json", 'w') as f:
            # Convert for JSON (remove non-serializable items)
            json_safe = []
            for r in all_results:
                r_copy = {
                    'seed': r['seed'],
                    'timestamp': r['timestamp'],
                    'conditions': {}
                }
                for c, v in r['conditions'].items():
                    r_copy['conditions'][c] = {
                        k: float(val) if isinstance(val, (int, float)) else val
                        for k, val in v.items()
                        if k != 'trajectory'
                    }
                json_safe.append(r_copy)
            json.dump(json_safe, f, indent=2)
        
        print(f"\n  Checkpoint saved ({len(all_results)}/{N_SEEDS} complete)")
    
    # Final aggregation and analysis
    print("\n" + "="*70)
    print("AGGREGATING RESULTS...")
    print("="*70)
    
    summary = aggregate_and_analyze(all_results)
    
    # Save final results
    final_results = {
        'all_results': all_results,
        'summary': summary,
        'config': {
            'n_seeds': N_SEEDS,
            'n_epochs_visual': N_EPOCHS_VISUAL,
            'n_epochs_lang': N_EPOCHS_LANG,
            'n_epochs_binding': N_EPOCHS_BINDING,
        },
        'completed': datetime.now().isoformat(),
    }
    torch.save(final_results, SAVE_DIR / "complete.pt")
    
    # Print results
    print_results(summary)
    
    print("\n" + "="*70)
    print("VALIDATION STUDY COMPLETE")
    print(f"Results saved to: {SAVE_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*70)
