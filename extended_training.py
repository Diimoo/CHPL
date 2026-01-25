"""
Extended Training: Track developmental trajectory over 100 epochs.

Scientific question: Does color bias persist, reduce, or oscillate?

Run:
    python extended_training.py

Expected time: ~60 min for 3 seeds × 100 epochs on RTX 4080
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import create_stimulus, generate_training_pairs, SHAPES, COLORS
from validation_study import measure_color_shape_bias, measure_binding_rate, count_active_concepts

# ============================================================
# CONFIGURATION
# ============================================================

N_SEEDS = 3
N_EPOCHS = 100
MEASURE_EVERY = 5  # Measure metrics every N epochs
SAVE_DIR = Path("extended_training_results")
SAVE_DIR.mkdir(exist_ok=True)


def train_with_trajectory(seed: int) -> dict:
    """Train for extended period and track developmental trajectory."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Initialize
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    train_pairs = generate_training_pairs(n_per_combination=20)
    
    trajectory = {
        'epochs': [],
        'bias_scores': [],
        'binding_rates': [],
        'concept_counts': [],
        'reconstruction_losses': [],
    }
    
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    
    print(f"\n  Training {N_EPOCHS} epochs...")
    
    for epoch in range(N_EPOCHS):
        epoch_recon_loss = 0
        n_samples = 0
        
        np.random.shuffle(train_pairs)
        
        # Phase 1: Visual reconstruction (always active)
        for pair in train_pairs[:200]:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(pair.image).float())
            loss.backward()
            vis_optimizer.step()
            epoch_recon_loss += loss.item()
            n_samples += 1
        
        # Phase 2: Language alignment (after epoch 5)
        if epoch >= 5:
            for pair in train_pairs[:100]:
                with torch.no_grad():
                    vis_feat = brain.visual(torch.from_numpy(pair.image).float())
                lang_feat = brain.language(pair.label)
                loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
                lang_optimizer.zero_grad()
                loss.backward()
                lang_optimizer.step()
        
        # Phase 3: Cross-modal binding (after epoch 10)
        if epoch >= 10:
            for pair in train_pairs[:50]:
                brain.experience(
                    torch.from_numpy(pair.image).float(),
                    pair.label,
                    consolidate=True,
                    train_cortices=False
                )
        
        # Measure metrics periodically
        if epoch % MEASURE_EVERY == 0 or epoch == N_EPOCHS - 1:
            bias = measure_color_shape_bias(brain)
            binding = measure_binding_rate(brain)
            concepts = count_active_concepts(brain)
            
            trajectory['epochs'].append(epoch)
            trajectory['bias_scores'].append(bias['bias_score'])
            trajectory['binding_rates'].append(binding)
            trajectory['concept_counts'].append(concepts)
            trajectory['reconstruction_losses'].append(epoch_recon_loss / n_samples)
            
            print(f"    Epoch {epoch:3d}: Bias={bias['bias_score']:.3f}, "
                  f"Binding={binding:.1f}%, Concepts={concepts}")
    
    return trajectory


def main():
    print("="*70)
    print("EXTENDED TRAINING: Developmental Trajectory")
    print(f"Seeds: {N_SEEDS}, Epochs: {N_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    all_trajectories = []
    
    for seed in range(N_SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{N_SEEDS}")
        print(f"{'='*70}")
        
        start = time.time()
        trajectory = train_with_trajectory(seed)
        trajectory['seed'] = seed
        trajectory['time_seconds'] = time.time() - start
        
        all_trajectories.append(trajectory)
        
        # Save checkpoint
        torch.save({
            'trajectories': all_trajectories,
            'n_completed': len(all_trajectories),
            'n_total': N_SEEDS,
        }, SAVE_DIR / "checkpoint.pt")
        
        print(f"\n  Completed in {trajectory['time_seconds']:.1f}s")
    
    # Save final results
    torch.save({
        'trajectories': all_trajectories,
        'config': {
            'n_seeds': N_SEEDS,
            'n_epochs': N_EPOCHS,
            'measure_every': MEASURE_EVERY,
        },
        'completed': datetime.now().isoformat(),
    }, SAVE_DIR / "complete.pt")
    
    # Summary statistics
    print("\n" + "="*70)
    print("TRAJECTORY SUMMARY")
    print("="*70)
    
    final_biases = [t['bias_scores'][-1] for t in all_trajectories]
    final_bindings = [t['binding_rates'][-1] for t in all_trajectories]
    
    print(f"\nFinal (epoch {N_EPOCHS}):")
    print(f"  Bias: {np.mean(final_biases):.3f} ± {np.std(final_biases):.3f}")
    print(f"  Binding: {np.mean(final_bindings):.1f}% ± {np.std(final_bindings):.1f}%")
    
    # Early vs late comparison
    early_biases = [t['bias_scores'][2] for t in all_trajectories]  # Epoch 10
    print(f"\nEarly (epoch 10) → Final (epoch {N_EPOCHS}):")
    print(f"  Bias: {np.mean(early_biases):.3f} → {np.mean(final_biases):.3f}")
    
    if np.mean(final_biases) < np.mean(early_biases) - 0.1:
        print("  → Color bias DECREASED (shape emerging)")
    elif np.mean(final_biases) > np.mean(early_biases) + 0.1:
        print("  → Color bias INCREASED")
    else:
        print("  → Color bias STABLE")
    
    print("\n" + "="*70)
    print("EXTENDED TRAINING COMPLETE")
    print(f"Results saved to: {SAVE_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*70)
    
    # Generate Figure 3
    print("\nGenerating Figure 3...")
    from generate_figures import generate_figure3
    generate_figure3(all_trajectories)


if __name__ == "__main__":
    main()
