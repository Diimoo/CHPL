#!/usr/bin/env python3
"""
Language Ablation Study: Does label content drive bias trajectory?

Hypothesis: Language supervision drives shape learning.
- Color-only labels ("red", "blue") → No shape bias emerges
- Shape-only labels ("circle", "square") → Shape bias emerges earlier/stronger
- Full labels ("red circle") → Baseline (shape bias at ~epoch 600)

Run: python experiments/language_ablation.py
Expected time: ~2 hours for 3 conditions × 3 seeds × 1000 epochs
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import create_stimulus, SHAPES, COLORS
from validation_study import measure_color_shape_bias
from extended_training import measure_discrimination, measure_spatial_frequency_learning

# ============================================================
# CONFIGURATION
# ============================================================

N_SEEDS = 3
N_EPOCHS = 1000
MEASURE_EVERY = 50
SAVE_DIR = Path("language_ablation_results")
SAVE_DIR.mkdir(exist_ok=True)

# Label conditions
CONDITIONS = {
    'color_only': lambda color, shape: color,           # "red", "blue", etc.
    'shape_only': lambda color, shape: shape,           # "circle", "square", etc.
    'full': lambda color, shape: f"{color} {shape}",    # "red circle", etc.
}


def generate_training_pairs_with_labels(label_fn, n_per_combo=20):
    """Generate training pairs with custom label function."""
    pairs = []
    for shape in SHAPES[:3]:
        for color in COLORS[:3]:
            for _ in range(n_per_combo):
                noise = np.random.uniform(0, 0.05)
                offset = (np.random.randint(-2, 3), np.random.randint(-2, 3))
                img = create_stimulus(shape, color, 'small', noise=noise, offset=offset)
                label = label_fn(color, shape)
                pairs.append({'image': img, 'label': label, 'color': color, 'shape': shape})
    return pairs


def train_with_label_condition(seed: int, condition_name: str, label_fn) -> dict:
    """Train with specific label condition and track trajectory."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    train_pairs = generate_training_pairs_with_labels(label_fn, n_per_combo=20)
    
    trajectory = {
        'epochs': [],
        'bias_scores': [],
        'discrimination_ratios': [],
        'freq_ratios': [],
        'condition': condition_name,
        'seed': seed,
    }
    
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    
    for epoch in range(N_EPOCHS):
        np.random.shuffle(train_pairs)
        
        # Phase 1: Visual reconstruction
        for pair in train_pairs[:200]:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(pair['image']).float())
            loss.backward()
            vis_optimizer.step()
        
        # Phase 2: Language alignment (after epoch 5)
        if epoch >= 5:
            for pair in train_pairs[:100]:
                with torch.no_grad():
                    vis_feat = brain.visual(torch.from_numpy(pair['image']).float())
                lang_feat = brain.language(pair['label'])
                loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
                lang_optimizer.zero_grad()
                loss.backward()
                lang_optimizer.step()
        
        # Phase 3: Cross-modal binding (after epoch 10)
        if epoch >= 10:
            for pair in train_pairs[:50]:
                brain.experience(
                    torch.from_numpy(pair['image']).float(),
                    pair['label'],
                    consolidate=True,
                    train_cortices=False
                )
        
        # Measure metrics
        if epoch % MEASURE_EVERY == 0 or epoch == N_EPOCHS - 1:
            bias = measure_color_shape_bias(brain)
            discrim = measure_discrimination(brain)
            freq = measure_spatial_frequency_learning(brain)
            
            trajectory['epochs'].append(epoch)
            trajectory['bias_scores'].append(bias['bias_score'])
            trajectory['discrimination_ratios'].append(discrim['discrimination_ratio'])
            trajectory['freq_ratios'].append(freq['freq_ratio'])
            
            print(f"      Epoch {epoch:4d}: Bias={bias['bias_score']:.3f}, "
                  f"Discrim={discrim['discrimination_ratio']:.2f}")
    
    return trajectory


def main():
    print("=" * 70)
    print("LANGUAGE ABLATION STUDY")
    print("=" * 70)
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds: {N_SEEDS}, Epochs: {N_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    all_results = {cond: [] for cond in CONDITIONS}
    
    for condition_name, label_fn in CONDITIONS.items():
        print(f"\n{'=' * 70}")
        print(f"CONDITION: {condition_name.upper()}")
        print(f"{'=' * 70}")
        
        for seed in range(N_SEEDS):
            print(f"\n  Seed {seed + 1}/{N_SEEDS}...")
            start = time.time()
            
            trajectory = train_with_label_condition(seed, condition_name, label_fn)
            trajectory['time_seconds'] = time.time() - start
            all_results[condition_name].append(trajectory)
            
            print(f"    Completed in {trajectory['time_seconds']:.1f}s")
            
            # Checkpoint
            torch.save(all_results, SAVE_DIR / "checkpoint.pt")
    
    # Final save
    torch.save({
        'results': all_results,
        'config': {
            'n_seeds': N_SEEDS,
            'n_epochs': N_EPOCHS,
            'measure_every': MEASURE_EVERY,
            'conditions': list(CONDITIONS.keys()),
        },
        'completed': datetime.now().isoformat(),
    }, SAVE_DIR / "complete.pt")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for cond in CONDITIONS:
        trajectories = all_results[cond]
        final_biases = [t['bias_scores'][-1] for t in trajectories]
        early_biases = [t['bias_scores'][1] for t in trajectories]  # epoch 50
        
        print(f"\n{cond.upper()}:")
        print(f"  Early (epoch 50): {np.mean(early_biases):.3f} ± {np.std(early_biases):.3f}")
        print(f"  Final (epoch {N_EPOCHS}): {np.mean(final_biases):.3f} ± {np.std(final_biases):.3f}")
        
        if np.mean(final_biases) < 0.9:
            print(f"  → SHAPE BIAS EMERGED")
        elif np.mean(final_biases) > 1.1:
            print(f"  → COLOR BIAS PERSISTED")
        else:
            print(f"  → NEUTRAL")
    
    # Statistical comparison
    print("\n" + "-" * 70)
    print("HYPOTHESIS TEST")
    print("-" * 70)
    
    color_only_final = [t['bias_scores'][-1] for t in all_results['color_only']]
    shape_only_final = [t['bias_scores'][-1] for t in all_results['shape_only']]
    full_final = [t['bias_scores'][-1] for t in all_results['full']]
    
    from scipy import stats
    
    # Shape-only should have lower bias than color-only
    t_stat, p_val = stats.ttest_ind(shape_only_final, color_only_final)
    print(f"\nShape-only vs Color-only:")
    print(f"  Shape-only: {np.mean(shape_only_final):.3f} ± {np.std(shape_only_final):.3f}")
    print(f"  Color-only: {np.mean(color_only_final):.3f} ± {np.std(color_only_final):.3f}")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}")
    
    if p_val < 0.05 and np.mean(shape_only_final) < np.mean(color_only_final):
        print(f"  → CONFIRMED: Shape labels drive shape bias")
    else:
        print(f"  → NOT SIGNIFICANT (need more seeds or epochs)")
    
    print("\n" + "=" * 70)
    print("LANGUAGE ABLATION COMPLETE")
    print(f"Results saved to: {SAVE_DIR}/")
    print("=" * 70)
    
    # Generate comparison figure
    generate_comparison_figure(all_results)


def generate_comparison_figure(all_results):
    """Generate comparison figure for ablation study."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = {'color_only': '#e74c3c', 'shape_only': '#3498db', 'full': '#2ecc71'}
        
        # Panel A: Bias trajectory
        for cond, trajectories in all_results.items():
            epochs = trajectories[0]['epochs']
            biases = np.array([t['bias_scores'] for t in trajectories])
            mean_bias = np.mean(biases, axis=0)
            std_bias = np.std(biases, axis=0)
            
            axes[0].plot(epochs, mean_bias, color=colors[cond], linewidth=2, label=cond)
            axes[0].fill_between(epochs, mean_bias - std_bias, mean_bias + std_bias,
                                color=colors[cond], alpha=0.2)
        
        axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Neutral')
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Color Bias Score', fontsize=12, fontweight='bold')
        axes[0].set_title('(A) Bias Trajectory by Label Condition', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Panel B: Final bias comparison
        conditions = list(all_results.keys())
        final_means = [np.mean([t['bias_scores'][-1] for t in all_results[c]]) for c in conditions]
        final_stds = [np.std([t['bias_scores'][-1] for t in all_results[c]]) for c in conditions]
        
        bars = axes[1].bar(conditions, final_means, yerr=final_stds, 
                          color=[colors[c] for c in conditions], capsize=5, alpha=0.8)
        axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Final Color Bias Score', fontsize=12, fontweight='bold')
        axes[1].set_title('(B) Final Bias by Condition', fontsize=13, fontweight='bold')
        axes[1].set_ylim(0, max(final_means) + max(final_stds) + 0.3)
        
        plt.tight_layout()
        plt.savefig(SAVE_DIR / "language_ablation.png", dpi=150, bbox_inches='tight')
        plt.savefig(SAVE_DIR / "language_ablation.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved language_ablation.png/pdf")
        
    except Exception as e:
        print(f"Warning: Could not generate figure: {e}")


if __name__ == "__main__":
    main()
