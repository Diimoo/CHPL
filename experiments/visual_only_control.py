"""
Visual-Only Control Experiment

Critical control for the language ablation study.
Tests whether shape bias emerges from visual reconstruction alone,
without ANY language training (no Phase 2 or Phase 3).

If shape bias emerges here → reconstruction drives it
If shape bias does NOT emerge → language scaffolding is necessary
"""

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import create_stimulus, COLORS, SHAPES, SIZE_PARAMS

# Configuration
N_SEEDS = 3
N_EPOCHS = 1000
MEASURE_EVERY = 50
OUTPUT_DIR = Path('visual_only_results')
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_visual_only_data(n_per_combo: int = 3):
    """Generate images only, no labels."""
    data = []
    for color in COLORS:
        for shape in SHAPES:
            for size in SIZE_PARAMS:
                for _ in range(n_per_combo):
                    img = create_stimulus(shape, color, size)
                    data.append({
                        'image': torch.tensor(img, dtype=torch.float32),
                        'color': color,
                        'shape': shape
                    })
    return data


def measure_bias(model):
    """Measure color vs shape bias using cosine distances."""
    same_color_diff_shape = []
    diff_color_same_shape = []
    
    # Generate test stimuli
    test_imgs = {}
    for color in COLORS:
        for shape in SHAPES:
            img = create_stimulus(shape, color, 'small')
            img_t = torch.tensor(img, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                feat = model.visual(img_t)
            test_imgs[(color, shape)] = feat
    
    # Compare pairs
    for c1 in COLORS:
        for c2 in COLORS:
            for s1 in SHAPES:
                for s2 in SHAPES:
                    if c1 >= c2 and s1 >= s2:
                        continue
                    
                    f1 = test_imgs[(c1, s1)]
                    f2 = test_imgs[(c2, s2)]
                    
                    cos_dist = 1 - torch.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                    
                    if c1 == c2 and s1 != s2:
                        same_color_diff_shape.append(cos_dist)
                    elif c1 != c2 and s1 == s2:
                        diff_color_same_shape.append(cos_dist)
    
    scds = np.mean(same_color_diff_shape)
    dcss = np.mean(diff_color_same_shape)
    bias = scds / dcss if dcss > 0 else 1.0
    
    return bias, scds, dcss


def train_visual_only(seed: int):
    """Train with visual reconstruction ONLY - no language phases."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    data = generate_visual_only_data(n_per_combo=3)
    
    trajectory = {
        'epochs': [],
        'bias': [],
        'scds': [],
        'dcss': []
    }
    
    for epoch in range(N_EPOCHS):
        np.random.shuffle(data)
        
        # ONLY visual reconstruction - no language at all
        for item in data:
            img = item['image'].to(DEVICE)
            
            # Pure visual reconstruction learning
            features = model.visual(img)
            recon = model.visual.reconstruct(features)
            
            # Reconstruction loss
            if img.dim() == 3:
                img_chw = img.permute(2, 0, 1).unsqueeze(0)
            else:
                img_chw = img
            
            loss = torch.nn.functional.mse_loss(recon, img_chw)
            
            # Manual gradient update (Hebbian-style)
            model.visual.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.visual.parameters():
                    if param.grad is not None:
                        param.data -= 0.01 * param.grad
        
        # Measure
        if epoch % MEASURE_EVERY == 0 or epoch == N_EPOCHS - 1:
            bias, scds, dcss = measure_bias(model)
            trajectory['epochs'].append(epoch)
            trajectory['bias'].append(bias)
            trajectory['scds'].append(scds)
            trajectory['dcss'].append(dcss)
            print(f"      Epoch {epoch:4d}: Bias={bias:.3f}")
    
    return trajectory


def main():
    print("=" * 70)
    print("VISUAL-ONLY CONTROL EXPERIMENT")
    print("Testing: Does shape bias emerge without ANY language training?")
    print("=" * 70)
    
    all_trajectories = []
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\n  Seed {seed}/{N_SEEDS}...")
        traj = train_visual_only(seed)
        all_trajectories.append(traj)
        
        # Save checkpoint
        torch.save({
            'seed': seed,
            'trajectory': traj,
            'all_trajectories': all_trajectories
        }, OUTPUT_DIR / 'checkpoint.pt', _use_new_zipfile_serialization=True)
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS: VISUAL-ONLY CONTROL")
    print("=" * 70)
    
    early_biases = [t['bias'][1] for t in all_trajectories]  # epoch 50
    final_biases = [t['bias'][-1] for t in all_trajectories]  # epoch 1000
    
    early_mean, early_std = np.mean(early_biases), np.std(early_biases)
    final_mean, final_std = np.mean(final_biases), np.std(final_biases)
    
    print(f"\n  Early (epoch 50):  {early_mean:.3f} ± {early_std:.3f}")
    print(f"  Final (epoch 1000): {final_mean:.3f} ± {final_std:.3f}")
    
    if final_mean < 0.9:
        print(f"\n  → SHAPE BIAS EMERGED (bias < 0.9)")
        print(f"  → This suggests reconstruction alone drives shape learning")
    elif final_mean > 1.1:
        print(f"\n  → COLOR BIAS PERSISTS (bias > 1.1)")
        print(f"  → This suggests language is necessary for shape bias")
    else:
        print(f"\n  → NEUTRAL (0.9 < bias < 1.1)")
        print(f"  → Inconclusive - neither strong bias emerged")
    
    # Compare to language conditions (load previous results)
    lang_results_path = Path('language_ablation_results/complete.pt')
    if lang_results_path.exists():
        lang_data = torch.load(lang_results_path, weights_only=False)
        
        print("\n" + "-" * 70)
        print("COMPARISON WITH LANGUAGE CONDITIONS")
        print("-" * 70)
        
        for cond_name, cond_trajs in lang_data['trajectories'].items():
            cond_finals = [t['bias'][-1] for t in cond_trajs]
            cond_mean = np.mean(cond_finals)
            cond_std = np.std(cond_finals)
            print(f"  {cond_name:12s}: {cond_mean:.3f} ± {cond_std:.3f}")
        
        print(f"  {'visual_only':12s}: {final_mean:.3f} ± {final_std:.3f}")
        
        # Statistical test: visual-only vs full
        full_finals = [t['bias'][-1] for t in lang_data['trajectories']['full']]
        t_stat, p_val = stats.ttest_ind(final_biases, full_finals)
        print(f"\n  Visual-only vs Full labels:")
        print(f"    t={t_stat:.3f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            if final_mean < np.mean(full_finals):
                print(f"    → Visual-only shows STRONGER shape bias (significant)")
            else:
                print(f"    → Full labels show stronger shape bias (significant)")
        else:
            print(f"    → No significant difference")
    
    # Save final results
    results = {
        'condition': 'visual_only',
        'n_seeds': N_SEEDS,
        'n_epochs': N_EPOCHS,
        'early_bias': {'mean': early_mean, 'std': early_std},
        'final_bias': {'mean': final_mean, 'std': final_std},
        'trajectories': all_trajectories,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(results, OUTPUT_DIR / 'complete.pt')
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump({
            'early_bias': {'mean': early_mean, 'std': early_std},
            'final_bias': {'mean': final_mean, 'std': final_std},
            'interpretation': 'shape_bias' if final_mean < 0.9 else ('color_bias' if final_mean > 1.1 else 'neutral')
        }, f, indent=2)
    
    # Generate figure
    generate_figure(all_trajectories, lang_results_path)
    
    print("\n" + "=" * 70)
    print("VISUAL-ONLY CONTROL COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)


def generate_figure(visual_trajectories, lang_results_path):
    """Generate comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Trajectories
    ax = axes[0]
    
    # Plot visual-only
    epochs = visual_trajectories[0]['epochs']
    biases = np.array([t['bias'] for t in visual_trajectories])
    mean_bias = biases.mean(axis=0)
    std_bias = biases.std(axis=0)
    
    ax.plot(epochs, mean_bias, 'k-', linewidth=2, label='visual_only')
    ax.fill_between(epochs, mean_bias - std_bias, mean_bias + std_bias, alpha=0.2, color='black')
    
    # Plot language conditions if available
    colors = {'color_only': 'red', 'shape_only': 'blue', 'full': 'green'}
    if lang_results_path.exists():
        lang_data = torch.load(lang_results_path, weights_only=False)
        for cond_name, cond_trajs in lang_data['trajectories'].items():
            epochs_c = cond_trajs[0]['epochs']
            biases_c = np.array([t['bias'] for t in cond_trajs])
            mean_c = biases_c.mean(axis=0)
            std_c = biases_c.std(axis=0)
            ax.plot(epochs_c, mean_c, color=colors.get(cond_name, 'gray'), linewidth=1.5, alpha=0.7, label=cond_name)
            ax.fill_between(epochs_c, mean_c - std_c, mean_c + std_c, alpha=0.1, color=colors.get(cond_name, 'gray'))
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Color Bias Score')
    ax.set_title('(A) Bias Trajectory: Visual-Only vs Language Conditions')
    ax.legend(loc='upper right')
    ax.set_ylim(0.3, 1.2)
    
    # Panel B: Final comparison
    ax = axes[1]
    
    conditions = ['visual_only']
    means = [np.mean([t['bias'][-1] for t in visual_trajectories])]
    stds = [np.std([t['bias'][-1] for t in visual_trajectories])]
    bar_colors = ['black']
    
    if lang_results_path.exists():
        for cond_name in ['color_only', 'shape_only', 'full']:
            cond_trajs = lang_data['trajectories'][cond_name]
            conditions.append(cond_name)
            means.append(np.mean([t['bias'][-1] for t in cond_trajs]))
            stds.append(np.std([t['bias'][-1] for t in cond_trajs]))
            bar_colors.append(colors[cond_name])
    
    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel('Final Color Bias Score')
    ax.set_title('(B) Final Bias by Condition')
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'visual_only_control.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'visual_only_control.pdf', bbox_inches='tight')
    print(f"✓ Saved visual_only_control.png/pdf")


if __name__ == '__main__':
    main()
