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
from synthetic_environment import create_stimulus, generate_training_pairs, SHAPES, COLORS, SIZES
from validation_study import measure_color_shape_bias, measure_binding_rate, count_active_concepts


# ============================================================
# NEW MECHANISTIC METRICS
# ============================================================

def measure_discrimination(brain) -> dict:
    """
    Measure whether model organizes features by color vs shape.
    
    Tests feature distance for:
    - Same color, different shape pairs
    - Different color, same shape pairs
    
    Returns:
        discrimination_ratio > 1: model discriminates by color
        discrimination_ratio < 1: model discriminates by shape
    """
    same_color_distances = []
    diff_color_distances = []
    
    brain.visual.eval()
    with torch.no_grad():
        for color in COLORS[:3]:  # red, blue, green
            for shape1 in SHAPES[:3]:  # circle, square, triangle
                for shape2 in SHAPES[:3]:
                    if shape1 >= shape2:
                        continue
                    # Same color, different shapes
                    img1 = create_stimulus(shape1, color, 'small')
                    img2 = create_stimulus(shape2, color, 'small')
                    f1 = brain.visual(torch.from_numpy(img1).float())
                    f2 = brain.visual(torch.from_numpy(img2).float())
                    dist = 1 - F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                    same_color_distances.append(dist)
        
        for shape in SHAPES[:3]:
            for color1 in COLORS[:3]:
                for color2 in COLORS[:3]:
                    if color1 >= color2:
                        continue
                    # Different colors, same shape
                    img1 = create_stimulus(shape, color1, 'small')
                    img2 = create_stimulus(shape, color2, 'small')
                    f1 = brain.visual(torch.from_numpy(img1).float())
                    f2 = brain.visual(torch.from_numpy(img2).float())
                    dist = 1 - F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                    diff_color_distances.append(dist)
    
    avg_same_color = np.mean(same_color_distances)
    avg_diff_color = np.mean(diff_color_distances)
    
    return {
        'same_color_diff_shape_dist': avg_same_color,
        'diff_color_same_shape_dist': avg_diff_color,
        'discrimination_ratio': avg_diff_color / (avg_same_color + 1e-8)
    }


def measure_spatial_frequency_learning(brain) -> dict:
    """
    Measure how well model reconstructs low-freq (color) vs high-freq (edge) components.
    
    Uses Gaussian blur to separate frequency components.
    Lower ratio = low-freq learned better relative to high-freq.
    """
    def gaussian_kernel(kernel_size=5, sigma=2.0):
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel / kernel.sum()
        return kernel.expand(3, 1, kernel_size, kernel_size).to(DEVICE)
    
    kernel = gaussian_kernel()
    low_freq_errors = []
    high_freq_errors = []
    
    brain.visual.eval()
    with torch.no_grad():
        for shape in SHAPES[:3]:
            for color in COLORS[:3]:
                img = create_stimulus(shape, color, 'small')
                img_tensor = torch.from_numpy(img).float()  # [H, W, C]
                
                # Get features (brain.visual handles format internally)
                features = brain.visual(img_tensor)
                
                # Get reconstruction - returns [C, H, W] or [B, C, H, W]
                recon = brain.visual.reconstruct(features)
                
                # Ensure img_tensor is [C, H, W] for conv2d
                if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
                
                # Ensure recon is [C, H, W]
                if recon.dim() == 4:
                    recon = recon.squeeze(0)  # [1,C,H,W] -> [C,H,W]
                
                # Add batch dimension for conv2d: [1, C, H, W]
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                recon = recon.unsqueeze(0).to(DEVICE)
                
                # Decompose by frequency
                img_low = F.conv2d(img_tensor, kernel, padding=2, groups=3)
                recon_low = F.conv2d(recon, kernel, padding=2, groups=3)
                low_freq_errors.append(F.mse_loss(recon_low, img_low).item())
                
                img_high = img_tensor - img_low
                recon_high = recon - recon_low
                high_freq_errors.append(F.mse_loss(recon_high, img_high).item())
    
    return {
        'low_freq_error': np.mean(low_freq_errors),
        'high_freq_error': np.mean(high_freq_errors),
        'freq_ratio': np.mean(low_freq_errors) / (np.mean(high_freq_errors) + 1e-8)
    }

# ============================================================
# CONFIGURATION
# ============================================================

N_SEEDS = 1
N_EPOCHS = 100000
MEASURE_EVERY = 1000  # Measure metrics every N epochs
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
        # New mechanistic metrics
        'discrimination_ratios': [],
        'freq_ratios': [],
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
            discrimination = measure_discrimination(brain)
            freq = measure_spatial_frequency_learning(brain)
            
            trajectory['epochs'].append(epoch)
            trajectory['bias_scores'].append(bias['bias_score'])
            trajectory['binding_rates'].append(binding)
            trajectory['concept_counts'].append(concepts)
            trajectory['reconstruction_losses'].append(epoch_recon_loss / n_samples)
            trajectory['discrimination_ratios'].append(discrimination['discrimination_ratio'])
            trajectory['freq_ratios'].append(freq['freq_ratio'])
            
            print(f"    Epoch {epoch:3d}: Bias={bias['bias_score']:.3f}, "
                  f"Discrim={discrimination['discrimination_ratio']:.2f}, "
                  f"FreqRatio={freq['freq_ratio']:.2f}")
    
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
    final_discrim = [t['discrimination_ratios'][-1] for t in all_trajectories]
    final_freq = [t['freq_ratios'][-1] for t in all_trajectories]
    
    print(f"\nFinal (epoch {N_EPOCHS}):")
    print(f"  Bias: {np.mean(final_biases):.3f} ± {np.std(final_biases):.3f}")
    print(f"  Binding: {np.mean(final_bindings):.1f}% ± {np.std(final_bindings):.1f}%")
    print(f"  Discrimination: {np.mean(final_discrim):.3f} ± {np.std(final_discrim):.3f}")
    print(f"  Freq Ratio: {np.mean(final_freq):.3f} ± {np.std(final_freq):.3f}")
    
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
