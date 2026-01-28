#!/usr/bin/env python3
"""
Generate paper figures for Distributed ATL paper.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Ensure output directory exists
out_dir = Path(__file__).parent.parent / 'paper' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)


def figure1_main_comparison():
    """
    Bar plot: Held-out performance comparison between baseline and distributed ATL.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    systems = ['Winner-Takes-All\n(Baseline)', 'Distributed ATL\n(Ours)']
    held_out = [0.512, 0.663]
    train = [0.862, 0.686]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train, width, label='Train', 
                   color=['#1f77b4', '#1f77b4'], alpha=0.7)
    bars2 = ax.bar(x + width/2, held_out, width, label='Held-out', 
                   color=['#d62728', '#2ca02c'])
    
    # Add gap annotations
    gaps = [train[0] - held_out[0], train[1] - held_out[1]]
    for i, g in enumerate(gaps):
        y_pos = max(train[i], held_out[i]) + 0.03
        ax.text(i, y_pos, f'gap={g:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add improvement annotation
    improvement = (held_out[1] - held_out[0]) / held_out[0] * 100
    ax.annotate('', xy=(1.175, held_out[1]), xytext=(1.175, held_out[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(1.35, (held_out[0] + held_out[1]) / 2, f'+{improvement:.1f}%', 
            fontsize=11, color='green', fontweight='bold', va='center')
    
    ax.set_ylabel('Pattern Similarity', fontsize=12)
    ax.set_title('Compositional Generalization:\nDistributed vs Winner-Takes-All', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'figure1_main_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'figure1_main_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1: {out_dir / 'figure1_main_comparison.png'}")


def figure2_generalization():
    """
    Multiple subplots showing different generalization tests.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Harder splits
    splits = ['Color\nHoldout', 'Relation\nHoldout', 'Swap\nGen.', 'Novel\nCombo']
    held_out = [0.663, 0.676, 0.649, 0.648]
    gaps = [0.023, 0.020, 0.017, 0.042]
    
    bars = axes[0].bar(splits, held_out, color='#2ca02c', alpha=0.8, edgecolor='black')
    axes[0].axhline(0.512, color='#d62728', linestyle='--', 
                    label='Baseline (0.512)', linewidth=2)
    axes[0].axhline(0.6, color='gray', linestyle=':', alpha=0.5, label='Success threshold')
    
    # Add gap labels on bars
    for i, (h, g) in enumerate(zip(held_out, gaps)):
        axes[0].text(i, h + 0.015, f'gap={g:.3f}', ha='center', fontsize=8)
    
    axes[0].set_ylabel('Held-out Pattern Similarity', fontsize=11)
    axes[0].set_title('(A) Generalization Across Splits', fontweight='bold', fontsize=12)
    axes[0].set_ylim(0.4, 0.75)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Panel B: Variable object counts
    counts = ['1 obj', '2 obj', '3 obj', '4 obj']
    similarities = [0.639, 0.667, 0.661, 0.637]
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e']
    edgecolors = ['black', 'black', 'black', 'red']
    linewidths = [1, 1, 1, 3]
    
    bars = axes[1].bar(counts, similarities, color=colors, alpha=0.8, 
                       edgecolor=edgecolors, linewidth=linewidths)
    axes[1].axhline(0.6, color='gray', linestyle=':', alpha=0.5)
    
    # Add train/test labels
    for i, c in enumerate(counts):
        label = 'TRAIN' if i < 3 else 'TEST'
        color = 'black' if i < 3 else 'red'
        axes[1].text(i, similarities[i] + 0.015, label, ha='center', 
                     fontsize=9, fontweight='bold', color=color)
    
    axes[1].set_ylabel('Pattern Similarity', fontsize=11)
    axes[1].set_title('(B) Variable Object Counts', fontweight='bold', fontsize=12)
    axes[1].set_ylim(0.5, 0.75)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Panel C: Temperature sweep (ablation)
    temps = ['0.1\n(sparse)', '0.2\n(optimal)', '0.5\n(diffuse)']
    train_sim = [0.324, 0.704, 0.939]
    held_out_sim = [0.201, 0.732, 0.937]
    
    x = np.arange(len(temps))
    width = 0.35
    
    axes[2].bar(x - width/2, train_sim, width, label='Train', 
                color='#1f77b4', alpha=0.7)
    axes[2].bar(x + width/2, held_out_sim, width, label='Held-out', 
                color='#2ca02c', alpha=0.7)
    
    # Highlight optimal
    axes[2].axvspan(0.5, 1.5, alpha=0.15, color='green')
    axes[2].text(1, 0.95, '✓ Optimal', ha='center', fontsize=10, color='green', fontweight='bold')
    
    axes[2].set_xlabel('Temperature (τ)', fontsize=11)
    axes[2].set_ylabel('Pattern Similarity', fontsize=11)
    axes[2].set_title('(C) Temperature Ablation', fontweight='bold', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(temps, fontsize=10)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'figure2_generalization.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'figure2_generalization.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2: {out_dir / 'figure2_generalization.png'}")


def figure3_architecture():
    """
    Architecture diagram of Distributed ATL.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.97, 'Distributed ATL Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    
    # === INPUT LAYER ===
    # Visual input box
    rect_vis_in = plt.Rectangle((0.05, 0.75), 0.18, 0.15, 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect_vis_in)
    ax.text(0.14, 0.825, 'Visual Input\n(56×56 RGB)', ha='center', va='center', fontsize=10)
    
    # Language input box
    rect_lang_in = plt.Rectangle((0.77, 0.75), 0.18, 0.15, 
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rect_lang_in)
    ax.text(0.86, 0.825, 'Language Input\n"red circle above\nblue square"', 
            ha='center', va='center', fontsize=9)
    
    # === CORTEX LAYER ===
    # Visual cortex
    rect_vis_ctx = plt.Rectangle((0.05, 0.5), 0.18, 0.15, 
                                   facecolor='steelblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect_vis_ctx)
    ax.text(0.14, 0.575, 'Visual Cortex\n(Conv Encoder)', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # Language cortex
    rect_lang_ctx = plt.Rectangle((0.77, 0.5), 0.18, 0.15, 
                                    facecolor='forestgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rect_lang_ctx)
    ax.text(0.86, 0.575, 'Language Cortex\n(Embedding)', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # Arrows from input to cortex
    ax.annotate('', xy=(0.14, 0.65), xytext=(0.14, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(0.86, 0.65), xytext=(0.86, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Feature labels
    ax.text(0.14, 0.44, 'f_vis (64-dim)', ha='center', fontsize=9, style='italic')
    ax.text(0.86, 0.44, 'f_lang (64-dim)', ha='center', fontsize=9, style='italic')
    
    # === DISTRIBUTED ATL (Center) ===
    rect_atl = plt.Rectangle((0.3, 0.18), 0.4, 0.22, 
                              facecolor='orange', edgecolor='darkred', linewidth=3)
    ax.add_patch(rect_atl)
    ax.text(0.5, 0.34, 'Distributed ATL', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.28, '200 Prototypes', ha='center', fontsize=11)
    ax.text(0.5, 0.22, r'$\alpha_i = \mathrm{softmax}(\mathbf{p}_i \cdot \mathbf{f} / \tau)$', 
            ha='center', fontsize=11)
    
    # Arrows from cortex to ATL
    ax.annotate('', xy=(0.3, 0.29), xytext=(0.23, 0.5),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5))
    ax.annotate('', xy=(0.7, 0.29), xytext=(0.77, 0.5),
                arrowprops=dict(arrowstyle='->', color='forestgreen', lw=2.5))
    
    # Projection labels
    ax.text(0.24, 0.38, 'Project', ha='center', fontsize=9, color='steelblue', rotation=45)
    ax.text(0.76, 0.38, 'Project', ha='center', fontsize=9, color='forestgreen', rotation=-45)
    
    # === ACTIVATION PATTERNS (visual representation) ===
    # Visual activation pattern
    ax.text(0.15, 0.15, 'α_vis', ha='center', fontsize=10, fontweight='bold', color='steelblue')
    pattern_x = np.linspace(0.05, 0.25, 20)
    pattern_y_vis = 0.08 + 0.04 * np.random.rand(20)
    pattern_y_vis[5:8] = 0.12  # Some high activations
    ax.bar(pattern_x, pattern_y_vis - 0.04, width=0.008, color='steelblue', alpha=0.7)
    ax.axhline(0.04, xmin=0.05, xmax=0.25, color='black', linewidth=0.5)
    
    # Language activation pattern
    ax.text(0.85, 0.15, 'α_lang', ha='center', fontsize=10, fontweight='bold', color='forestgreen')
    pattern_x2 = np.linspace(0.75, 0.95, 20)
    pattern_y_lang = 0.08 + 0.04 * np.random.rand(20)
    pattern_y_lang[5:8] = 0.12  # Similar high activations
    ax.bar(pattern_x2, pattern_y_lang - 0.04, width=0.008, color='forestgreen', alpha=0.7)
    ax.axhline(0.04, xmin=0.75, xmax=0.95, color='black', linewidth=0.5)
    
    # === OUTPUT ===
    ax.text(0.5, 0.08, 'Binding Quality = cos(α_vis, α_lang)', 
            ha='center', fontsize=12, style='italic', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Arrows from ATL to patterns
    ax.annotate('', xy=(0.15, 0.16), xytext=(0.35, 0.18),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))
    ax.annotate('', xy=(0.85, 0.16), xytext=(0.65, 0.18),
                arrowprops=dict(arrowstyle='->', color='forestgreen', lw=1.5))
    
    # Key insight box
    ax.text(0.5, 0.01, 'Key: Composition emerges from PATTERN similarity, not single-winner matching',
            ha='center', fontsize=10, style='italic', color='darkred')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'figure3_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'figure3_architecture.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3: {out_dir / 'figure3_architecture.png'}")


def figure4_multiseed():
    """
    Multi-seed validation results showing robustness.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Panel A: Per-seed breakdown
    seeds = [0, 42, 123, 456, 999]
    train_sims = [0.679, 0.677, 0.680, 0.700, 0.696]
    held_out_sims = [0.667, 0.679, 0.626, 0.698, 0.647]
    
    x = np.arange(len(seeds))
    width = 0.35
    
    axes[0].bar(x - width/2, train_sims, width, label='Train', color='#1f77b4', alpha=0.7)
    axes[0].bar(x + width/2, held_out_sims, width, label='Held-out', color='#2ca02c', alpha=0.7)
    
    axes[0].axhline(0.512, color='#d62728', linestyle='--', label='Baseline', linewidth=2)
    axes[0].axhline(0.65, color='gray', linestyle=':', alpha=0.5)
    
    axes[0].set_xlabel('Random Seed', fontsize=11)
    axes[0].set_ylabel('Pattern Similarity', fontsize=11)
    axes[0].set_title('(A) Per-Seed Results (n=5)', fontweight='bold', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds)
    axes[0].set_ylim(0.4, 0.8)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Panel B: Aggregated with error bars
    methods = ['Winner-Takes-All\n(Baseline)', 'Distributed ATL\n(Ours, n=5)']
    means = [0.512, 0.663]
    stds = [0.0, 0.025]  # Baseline is single run
    
    bars = axes[1].bar(methods, means, yerr=stds, capsize=8, 
                       color=['#d62728', '#2ca02c'], alpha=0.8, edgecolor='black')
    
    # Add improvement arrow
    axes[1].annotate('', xy=(1, means[1]), xytext=(0, means[0]),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                                     connectionstyle='arc3,rad=0.3'))
    axes[1].text(0.5, 0.58, '+29.6%', ha='center', fontsize=12, fontweight='bold', color='green')
    
    axes[1].set_ylabel('Held-out Pattern Similarity', fontsize=11)
    axes[1].set_title('(B) Aggregated Comparison', fontweight='bold', fontsize=12)
    axes[1].set_ylim(0.3, 0.8)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add mean±std annotation
    axes[1].text(1, means[1] + stds[1] + 0.03, f'{means[1]:.3f}±{stds[1]:.3f}', 
                 ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'figure4_multiseed.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'figure4_multiseed.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4: {out_dir / 'figure4_multiseed.png'}")


if __name__ == '__main__':
    print("Generating paper figures...")
    print("=" * 50)
    
    figure1_main_comparison()
    figure2_generalization()
    figure3_architecture()
    figure4_multiseed()
    
    print("=" * 50)
    print("All figures saved to paper/figures/")
