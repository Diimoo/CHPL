"""
Generate publication-quality figures for brain-like learning paper.

Figure 2: Ablation study results with error bars
Figure 3: Developmental trajectory (requires extended training data)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2


def generate_figure2():
    """Generate Figure 2: Ablation results with error bars."""
    
    # Data from validation study (10 seeds)
    conditions = ['Full\nModel', 'No\nReconstruction', 'No\nConsolidation', 'Backprop']
    bias_means = [1.227, 1.020, 1.177, 2.245]
    bias_stds = [0.090, 0.010, 0.151, 0.475]
    
    binding_means = [62.5, 100.0, 44.4]  # No backprop binding
    binding_stds = [8.4, 0, 15.7]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color palette
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    # Panel A: Color Bias Score
    bars = axes[0].bar(range(4), bias_means, yerr=bias_stds, 
                        capsize=8, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=1.0, color='red', linestyle='--', 
                    linewidth=2, alpha=0.5, label='Neutral (no bias)')
    axes[0].set_ylabel('Color Bias Score', fontsize=14, fontweight='bold')
    axes[0].set_title('(A) Color vs Shape Categorization Bias', 
                       fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(conditions, fontsize=11)
    axes[0].set_ylim(0, 3.2)
    axes[0].legend(fontsize=11, loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add significance bracket between Full Model and Backprop
    axes[0].plot([0, 0, 3, 3], [2.85, 2.95, 2.95, 2.85], 'k-', linewidth=1.5)
    axes[0].text(1.5, 3.0, '***  p < 0.00001', ha='center', fontsize=11, fontweight='bold')
    
    # Panel B: Binding Rate
    bars = axes[1].bar(range(3), binding_means, yerr=binding_stds,
                        capsize=8, color=colors[:3], alpha=0.8,
                        edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Cross-Modal Binding Rate (%)', fontsize=14, fontweight='bold')
    axes[1].set_title('(B) Vision-Language Alignment', 
                       fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(conditions[:3], fontsize=11)
    axes[1].set_ylim(0, 125)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add significance bracket
    axes[1].plot([0, 0, 2, 2], [95, 100, 100, 95], 'k-', linewidth=1.5)
    axes[1].text(1, 103, '*  p = 0.010', ha='center', fontsize=11, fontweight='bold')
    
    # Add note about no-recon collapse
    axes[1].annotate('Collapsed\n(1 concept)', xy=(1, 100), xytext=(1, 85),
                     ha='center', fontsize=10, style='italic', color='#c0392b',
                     arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('figure2_ablation_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_ablation_results.pdf', bbox_inches='tight')
    print("✓ Saved figure2_ablation_results.png/pdf")
    plt.close()


def generate_figure3(trajectories: list):
    """
    Generate Figure 3: Developmental trajectory.
    
    Args:
        trajectories: List of dicts with 'epochs', 'bias_scores', 
                      'binding_rates', 'concept_counts'
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Generate colors dynamically for any number of seeds
    n_seeds = len(trajectories)
    cmap = plt.cm.tab10 if n_seeds <= 10 else plt.cm.tab20
    colors_seeds = [cmap(i / max(n_seeds, 1)) for i in range(n_seeds)]
    
    # Panel A: Bias over time
    for i, traj in enumerate(trajectories):
        axes[0].plot(traj['epochs'], traj['bias_scores'], 
                     color=colors_seeds[i], alpha=0.4, linewidth=1.5)
    
    # Mean trajectory
    epochs = trajectories[0]['epochs']
    mean_bias = np.mean([t['bias_scores'] for t in trajectories], axis=0)
    std_bias = np.std([t['bias_scores'] for t in trajectories], axis=0)
    
    axes[0].plot(epochs, mean_bias, 'k-', linewidth=3, label='Mean')
    axes[0].fill_between(epochs, mean_bias - std_bias, mean_bias + std_bias, 
                         color='gray', alpha=0.2)
    axes[0].axhline(y=1.0, color='red', linestyle='--', 
                    linewidth=2, alpha=0.5, label='Neutral')
    axes[0].set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Color Bias Score', fontsize=12, fontweight='bold')
    axes[0].set_title('(A) Developmental Trajectory of Color Bias', 
                       fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, max(epochs))
    
    # Panel B: Binding rate over time
    for i, traj in enumerate(trajectories):
        axes[1].plot(traj['epochs'], traj['binding_rates'], 
                     color=colors_seeds[i], alpha=0.5, linewidth=2)
    
    mean_binding = np.mean([t['binding_rates'] for t in trajectories], axis=0)
    std_binding = np.std([t['binding_rates'] for t in trajectories], axis=0)
    
    axes[1].plot(epochs, mean_binding, 'k-', linewidth=3, label='Mean')
    axes[1].fill_between(epochs, mean_binding - std_binding, mean_binding + std_binding,
                         color='gray', alpha=0.2)
    axes[1].set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Binding Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('(B) Cross-Modal Alignment Over Time', 
                       fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, max(epochs))
    axes[1].set_ylim(0, 100)
    
    # Panel C: Concept count over time
    for i, traj in enumerate(trajectories):
        axes[2].plot(traj['epochs'], traj['concept_counts'], 
                     color=colors_seeds[i], alpha=0.5, linewidth=2)
    
    mean_concepts = np.mean([t['concept_counts'] for t in trajectories], axis=0)
    std_concepts = np.std([t['concept_counts'] for t in trajectories], axis=0)
    
    axes[2].plot(epochs, mean_concepts, 'k-', linewidth=3, label='Mean')
    axes[2].fill_between(epochs, mean_concepts - std_concepts, mean_concepts + std_concepts,
                         color='gray', alpha=0.2)
    axes[2].set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Active Concepts', fontsize=12, fontweight='bold')
    axes[2].set_title('(C) Concept Differentiation', 
                       fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(0, max(epochs))
    
    plt.tight_layout()
    plt.savefig('figure3_developmental_trajectory.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_developmental_trajectory.pdf', bbox_inches='tight')
    print("✓ Saved figure3_developmental_trajectory.png/pdf")
    plt.close()


if __name__ == "__main__":
    print("Generating publication figures...")
    
    # Figure 2: Ablation results
    generate_figure2()
    
    # Figure 3: Check if extended training data exists
    extended_path = Path("extended_training_results")
    if extended_path.exists():
        import torch
        data = torch.load(extended_path / "complete.pt", weights_only=False)
        generate_figure3(data['trajectories'])
    else:
        print("⚠ Extended training data not found. Run extended_training.py first for Figure 3.")
