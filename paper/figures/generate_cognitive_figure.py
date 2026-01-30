#!/usr/bin/env python3
"""
Generate Figure: Cognitive Capabilities Overview

Shows the progression from perception to abstraction across 4 phases.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['font.family'] = 'sans-serif'

def create_cognitive_capabilities_figure():
    """Create 4-panel figure showing cognitive capabilities."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color scheme
    colors = {
        'pass': '#2ca02c',  # Green
        'fail': '#d62728',  # Red
        'partial': '#ff7f0e',  # Orange
        'primary': '#1f77b4',  # Blue
    }
    
    # ========== Panel A: Temporal Prediction ==========
    ax = axes[0, 0]
    
    steps = ['1-step', '2-step', '3-step', '4-step', '5-step']
    accuracies = [0.946, 0.834, 0.618, 0.412, 0.287]
    
    bars = ax.bar(steps, accuracies, color=colors['primary'], alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if accuracies[i] >= 0.6:
            bar.set_color(colors['pass'])
        elif accuracies[i] >= 0.4:
            bar.set_color(colors['partial'])
        else:
            bar.set_color(colors['fail'])
        bar.set_alpha(0.7)
    
    ax.axhline(0.6, color='red', linestyle='--', linewidth=2, label='Target (0.6)')
    ax.set_title('(A) Temporal Prediction', fontweight='bold', fontsize=14)
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # ========== Panel B: Causal Reasoning ==========
    ax = axes[0, 1]
    
    categories = ['Interaction\nClassification', 'Goal\nPlanning', 'Counterfactual']
    scores = [1.000, 1.000, 0.031]
    targets = [0.7, 0.6, 0.05]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bar_colors = [colors['pass'], colors['pass'], colors['fail']]
    bars = ax.bar(x, scores, width, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add target lines
    for i, target in enumerate(targets):
        ax.plot([x[i] - width/2, x[i] + width/2], [target, target], 
                color='red', linestyle='--', linewidth=2)
    
    ax.set_title('(B) Causal Reasoning', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.2)
    
    # Add value labels
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # ========== Panel C: Language Understanding ==========
    ax = axes[1, 0]
    
    tasks = ['Description', 'Color QA', 'Shape QA', 'Relation QA', 'Location QA']
    scores = [0.811, 1.000, 0.818, 0.895, 0.636]
    
    bar_colors = [colors['pass'] if s >= 0.7 else colors['partial'] if s >= 0.5 else colors['fail'] 
                  for s in scores]
    
    bars = ax.bar(tasks, scores, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='Target (0.7)')
    ax.set_title('(C) Language Understanding', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy / Word Overlap')
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(loc='upper right')
    
    # Add value labels
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # ========== Panel D: Abstract Reasoning ==========
    ax = axes[1, 1]
    
    tasks = ['Analogy\nSolving', 'Few-shot\n(3 examples)', 'Creative\nDiversity']
    scores = [1.000, 0.500, 0.546]
    targets = [0.6, 0.7, 0.3]
    
    bar_colors = [colors['pass'], colors['fail'], colors['pass']]
    bars = ax.bar(tasks, scores, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add target lines
    for i, target in enumerate(targets):
        ax.plot([i - 0.3, i + 0.3], [target, target], 
                color='red', linestyle='--', linewidth=2)
    
    ax.set_title('(D) Abstract Reasoning', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy / Diversity')
    ax.set_ylim(0, 1.2)
    
    # Add value labels
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('figure_cognitive_capabilities.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_cognitive_capabilities.pdf', bbox_inches='tight')
    print("Saved: figure_cognitive_capabilities.png")
    print("Saved: figure_cognitive_capabilities.pdf")
    
    plt.close()


def create_developmental_progression_figure():
    """Create figure showing developmental progression."""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Data
    phases = ['Phase 1\nPrediction', 'Phase 2\nCausality', 'Phase 3\nLanguage', 'Phase 4\nAbstraction']
    times = [6.5, 0.6, 0.9, 0.3]  # minutes
    levels = ['TODDLER', 'EARLY CHILD', 'CHILD', 'ADV. CHILD']
    key_metrics = ['Prediction: 0.946', 'Causal: 1.000', 'QA: 0.860', 'Analogy: 1.000']
    
    x = np.arange(len(phases))
    
    # Create bars with gradient colors
    colors = ['#a6d96a', '#66bd63', '#1a9850', '#006837']
    bars = ax.bar(x, times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_xlabel('Developmental Phase', fontsize=12)
    ax.set_title('Cognitive Development Timeline: 8.3 Minutes Total', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    
    # Add level labels above bars
    for i, (bar, level, metric) in enumerate(zip(bars, levels, key_metrics)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
                f'★ {level}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                metric, ha='center', va='bottom', fontsize=9, style='italic')
    
    # Add cumulative time line
    cumulative = np.cumsum(times)
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, 'ro-', markersize=10, linewidth=2, label='Cumulative Time')
    ax2.set_ylabel('Cumulative Time (minutes)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 10)
    
    # Add annotations for cumulative points
    for i, cum in enumerate(cumulative):
        ax2.annotate(f'{cum:.1f} min', (x[i], cum), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9, color='red')
    
    ax.set_ylim(0, 8)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figure_developmental_progression.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_developmental_progression.pdf', bbox_inches='tight')
    print("Saved: figure_developmental_progression.png")
    print("Saved: figure_developmental_progression.pdf")
    
    plt.close()


def create_unified_architecture_figure():
    """Create figure showing the unified architecture."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.axis('off')
    
    # Define box positions and sizes
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2)
    
    # Core components
    components = [
        (0.5, 0.9, 'Visual Cortex\n(Encoder-Decoder)', 'lightcoral'),
        (0.2, 0.7, 'Language Cortex\n(Embedding + LSTM)', 'lightgreen'),
        (0.5, 0.5, 'Distributed ATL\n(200 prototypes, τ=0.2)', 'lightyellow'),
        (0.8, 0.7, 'Temporal Predictor\n(State + Velocity)', 'lightblue'),
        (0.2, 0.3, 'Causal Network\n(Inference + Planning)', 'plum'),
        (0.5, 0.1, 'Hierarchical Abstraction\n(4 levels)', 'wheat'),
        (0.8, 0.3, 'Language Generator\n(LSTM Decoder)', 'paleturquoise'),
    ]
    
    for x, y, text, color in components:
        ax.annotate(text, xy=(x, y), fontsize=11, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor='black', linewidth=1.5))
    
    # Draw arrows
    arrows = [
        (0.5, 0.85, 0.5, 0.55),  # Visual → ATL
        (0.25, 0.65, 0.45, 0.55),  # Language → ATL
        (0.55, 0.55, 0.75, 0.65),  # ATL → Temporal
        (0.45, 0.45, 0.25, 0.35),  # ATL → Causal
        (0.5, 0.45, 0.5, 0.15),  # ATL → Hierarchical
        (0.55, 0.45, 0.75, 0.35),  # ATL → Generator
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_title('Unified Cognitive Architecture: Distributed ATL as Central Hub', 
                 fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figure_unified_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_unified_architecture.pdf', bbox_inches='tight')
    print("Saved: figure_unified_architecture.png")
    print("Saved: figure_unified_architecture.pdf")
    
    plt.close()


if __name__ == "__main__":
    print("Generating cognitive capabilities figures...")
    
    create_cognitive_capabilities_figure()
    create_developmental_progression_figure()
    create_unified_architecture_figure()
    
    print("\n✓ All figures generated!")
