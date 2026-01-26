"""
Fair Baseline Comparison: Addressing Methodological Critique

This experiment addresses the critique that comparing brain-like learning to 
classification-based backprop is unfair because they have different objectives.

We implement:
1. **Backprop-Autoencoder Baseline**: Same reconstruction objective as brain-like,
   but using standard backpropagation (no Hebbian, no competitive learning)
2. **Simultaneous Training Mode**: All modules trained together from epoch 0
   to test whether developmental trajectory is an artifact of phased training

Scientific question: 
- Is brain-like learning actually different from backprop with same objective?
- Is the color→shape trajectory emergent or an artifact of training phases?

Run:
    python experiments/fair_baseline_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import scipy.stats as stats

from brain_crossmodal_learner import (
    BrainCrossModalLearner, SimpleVisualCortex, SimpleLanguageCortex,
    ATL, DEVICE
)
from synthetic_environment import (
    create_stimulus, generate_training_pairs, generate_test_pairs,
    SHAPES, COLORS, SIZES
)
from validation_study import measure_color_shape_bias, measure_binding_rate, count_active_concepts


# ============================================================
# BACKPROP AUTOENCODER BASELINE
# ============================================================

class BackpropAutoencoder(nn.Module):
    """
    Standard Backprop Autoencoder - Fair baseline.
    
    Same architecture as SimpleVisualCortex but:
    - No Hebbian learning
    - No competitive learning in binding
    - Standard end-to-end backpropagation
    
    This is the FAIR comparison: same objective (reconstruction),
    different learning rule (backprop vs Hebbian).
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Encoder (same as SimpleVisualCortex)
        self.v1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.v2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.v4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.it = nn.Linear(64 * 3 * 3, feature_dim)
        
        # Decoder (same as SimpleVisualCortex)
        self.decoder_fc = nn.Linear(feature_dim, 64 * 3 * 3)
        self.decoder_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_v4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_v2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_v1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.feature_dim = feature_dim
        self.to(DEVICE)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to features."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.to(DEVICE)
        
        x = F.relu(self.v1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.v2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.v4(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = self.it(x)
        x = F.normalize(x, dim=-1)
        return x.squeeze(0) if x.size(0) == 1 else x
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to image."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        x = F.relu(self.decoder_fc(features))
        x = x.view(-1, 64, 3, 3)
        x = self.decoder_upsample(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.decoder_v4(x))
        x = self.decoder_upsample(x)
        x = F.relu(self.decoder_v2(x))
        x = self.decoder_upsample(x)
        x = torch.sigmoid(self.decoder_v1(x))
        return x.squeeze(0) if x.size(0) == 1 else x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns features (for compatibility)."""
        return self.encode(x)
    
    def reconstruction_loss(self, image: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        image = image.to(DEVICE)
        
        features = self.encode(image.permute(0, 2, 3, 1))
        if features.dim() == 1:
            features = features.unsqueeze(0)
        reconstructed = self.decode(features)
        
        return F.mse_loss(reconstructed, image)


class BackpropCrossModalLearner:
    """
    Cross-modal learner using ONLY backpropagation.
    
    No Hebbian learning, no competitive prototypes.
    Uses learned projection heads for cross-modal alignment.
    """
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        
        # Visual encoder (backprop autoencoder)
        self.visual = BackpropAutoencoder(feature_dim)
        
        # Language encoder (same architecture, backprop training)
        self.language = SimpleLanguageCortex(feature_dim)
        
        # Cross-modal projection heads (backprop-learned)
        self.vis_proj = nn.Linear(feature_dim, feature_dim).to(DEVICE)
        self.lang_proj = nn.Linear(feature_dim, feature_dim).to(DEVICE)
        
        # Learnable concept prototypes (backprop, not competitive)
        self.n_concepts = 100
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(self.n_concepts, feature_dim, device=DEVICE), dim=-1)
        )
        
        # Single optimizer for end-to-end training
        self.optimizer = torch.optim.Adam([
            {'params': self.visual.parameters(), 'lr': 1e-3},
            {'params': self.language.parameters(), 'lr': 1e-3},
            {'params': self.vis_proj.parameters(), 'lr': 1e-3},
            {'params': self.lang_proj.parameters(), 'lr': 1e-3},
            {'params': [self.prototypes], 'lr': 1e-2},
        ])
        
        self.stats = {'episodes': 0}
    
    def train_step(self, image: torch.Tensor, label: str) -> Dict:
        """
        Single training step with backprop.
        
        Loss = reconstruction + cross-modal alignment + concept binding
        """
        self.optimizer.zero_grad()
        
        # Visual reconstruction loss
        recon_loss = self.visual.reconstruction_loss(image)
        
        # Get features
        vis_feat = self.visual(image)
        lang_feat = self.language(label)
        
        # Project to shared space
        vis_proj = F.normalize(self.vis_proj(vis_feat), dim=-1)
        lang_proj = F.normalize(self.lang_proj(lang_feat), dim=-1)
        
        # Cross-modal alignment loss (contrastive-like)
        align_loss = 1 - F.cosine_similarity(vis_proj.unsqueeze(0), lang_proj.unsqueeze(0))
        
        # Concept binding: both should map to same prototype
        vis_sims = torch.matmul(self.prototypes, vis_proj)
        lang_sims = torch.matmul(self.prototypes, lang_proj)
        
        # Softmax over prototypes
        vis_probs = F.softmax(vis_sims / 0.1, dim=0)
        lang_probs = F.softmax(lang_sims / 0.1, dim=0)
        
        # KL divergence: visual and language should have same concept distribution
        binding_loss = F.kl_div(vis_probs.log(), lang_probs, reduction='sum')
        
        # Total loss
        total_loss = recon_loss + 0.5 * align_loss + 0.1 * binding_loss
        total_loss.backward()
        self.optimizer.step()
        
        # Renormalize prototypes
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        
        self.stats['episodes'] += 1
        
        return {
            'recon_loss': recon_loss.item(),
            'align_loss': align_loss.item(),
            'binding_loss': binding_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    def get_concept(self, features: torch.Tensor, modality: str) -> int:
        """Get winning concept for features."""
        if modality == 'visual':
            proj = F.normalize(self.vis_proj(features), dim=-1)
        else:
            proj = F.normalize(self.lang_proj(features), dim=-1)
        
        sims = torch.matmul(self.prototypes, proj)
        return sims.argmax().item()
    
    # Compatibility interface with BrainCrossModalLearner
    class ATLProxy:
        def __init__(self, parent):
            self.parent = parent
            self.usage = torch.zeros(parent.n_concepts, device=DEVICE)
        
        def activate(self, features, modality):
            concept = self.parent.get_concept(features, modality)
            self.usage[concept] += 1
            return None, concept
        
        def get_active_concepts(self):
            return int((self.usage > 0).sum().item())
    
    @property
    def atl(self):
        if not hasattr(self, '_atl_proxy'):
            self._atl_proxy = self.ATLProxy(self)
        return self._atl_proxy


# ============================================================
# SIMULTANEOUS TRAINING MODE
# ============================================================

def train_brain_simultaneous(seed: int, n_epochs: int = 50) -> Dict:
    """
    Train brain-like model with ALL modules simultaneously from epoch 0.
    
    This tests whether the developmental trajectory is:
    - EMERGENT: Appears even with simultaneous training
    - ARTIFACT: Only appears because of phased training
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    train_pairs = generate_training_pairs(n_per_combination=20)
    
    # Single optimizer for all components
    optimizer = torch.optim.Adam([
        {'params': brain.visual.parameters(), 'lr': 1e-3},
        {'params': brain.language.parameters(), 'lr': 1e-3},
        {'params': brain.atl.vis_proj.parameters(), 'lr': 1e-3},
        {'params': brain.atl.lang_proj.parameters(), 'lr': 1e-3},
    ])
    
    trajectory = {
        'epochs': [],
        'bias_scores': [],
        'binding_rates': [],
        'concept_counts': [],
    }
    
    for epoch in range(n_epochs):
        np.random.shuffle(train_pairs)
        
        for pair in train_pairs[:200]:
            optimizer.zero_grad()
            
            # Visual reconstruction (always)
            recon_loss = brain.visual.reconstruction_loss(
                torch.from_numpy(pair.image).float()
            )
            
            # Get features
            vis_feat = brain.visual(torch.from_numpy(pair.image).float())
            lang_feat = brain.language(pair.label)
            
            # Language alignment (always, from epoch 0)
            align_loss = 1 - F.cosine_similarity(
                lang_feat.unsqueeze(0), vis_feat.detach().unsqueeze(0)
            )
            
            # Combined loss
            total_loss = recon_loss + 0.3 * align_loss
            total_loss.backward()
            optimizer.step()
            
            # Hebbian consolidation (always, from epoch 0)
            with torch.no_grad():
                brain.atl.consolidate(vis_feat.detach(), lang_feat.detach())
        
        # Measure every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            bias = measure_color_shape_bias(brain)
            binding = measure_binding_rate(brain)
            concepts = count_active_concepts(brain)
            
            trajectory['epochs'].append(epoch)
            trajectory['bias_scores'].append(bias['bias_score'])
            trajectory['binding_rates'].append(binding)
            trajectory['concept_counts'].append(concepts)
    
    return trajectory


def train_brain_phased(seed: int, n_epochs: int = 50) -> Dict:
    """
    Train brain-like model with PHASED training (original method).
    
    Phase 1 (epochs 0-9): Visual cortex only
    Phase 2 (epochs 10-24): + Language alignment
    Phase 3 (epochs 25+): + Cross-modal binding
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    train_pairs = generate_training_pairs(n_per_combination=20)
    
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    
    trajectory = {
        'epochs': [],
        'bias_scores': [],
        'binding_rates': [],
        'concept_counts': [],
    }
    
    for epoch in range(n_epochs):
        np.random.shuffle(train_pairs)
        
        # Phase 1: Visual only (epochs 0-9)
        for pair in train_pairs[:200]:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(
                torch.from_numpy(pair.image).float()
            )
            loss.backward()
            vis_optimizer.step()
        
        # Phase 2: + Language (epochs 10+)
        if epoch >= 10:
            for pair in train_pairs[:100]:
                with torch.no_grad():
                    vis_feat = brain.visual(torch.from_numpy(pair.image).float())
                lang_feat = brain.language(pair.label)
                loss = 1 - F.cosine_similarity(
                    lang_feat.unsqueeze(0), vis_feat.unsqueeze(0)
                )
                lang_optimizer.zero_grad()
                loss.backward()
                lang_optimizer.step()
        
        # Phase 3: + Binding (epochs 20+)
        if epoch >= 20:
            for pair in train_pairs[:50]:
                vis_feat = brain.visual(torch.from_numpy(pair.image).float())
                lang_feat = brain.language(pair.label)
                brain.atl.consolidate(vis_feat.detach(), lang_feat.detach())
        
        # Measure every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            bias = measure_color_shape_bias(brain)
            binding = measure_binding_rate(brain)
            concepts = count_active_concepts(brain)
            
            trajectory['epochs'].append(epoch)
            trajectory['bias_scores'].append(bias['bias_score'])
            trajectory['binding_rates'].append(binding)
            trajectory['concept_counts'].append(concepts)
    
    return trajectory


def train_backprop_baseline(seed: int, n_epochs: int = 50) -> Dict:
    """
    Train backprop autoencoder baseline.
    
    Same objective (reconstruction + alignment), different learning rule.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = BackpropCrossModalLearner(feature_dim=64)
    train_pairs = generate_training_pairs(n_per_combination=20)
    
    trajectory = {
        'epochs': [],
        'bias_scores': [],
        'binding_rates': [],
        'concept_counts': [],
    }
    
    for epoch in range(n_epochs):
        np.random.shuffle(train_pairs)
        
        for pair in train_pairs[:200]:
            model.train_step(
                torch.from_numpy(pair.image).float(),
                pair.label
            )
        
        # Measure every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            bias = measure_color_shape_bias_backprop(model)
            binding = measure_binding_rate_backprop(model)
            concepts = model.atl.get_active_concepts()
            
            trajectory['epochs'].append(epoch)
            trajectory['bias_scores'].append(bias['bias_score'])
            trajectory['binding_rates'].append(binding)
            trajectory['concept_counts'].append(concepts)
    
    return trajectory


# ============================================================
# METRICS FOR BACKPROP BASELINE
# ============================================================

def measure_color_shape_bias_backprop(model: BackpropCrossModalLearner) -> dict:
    """Measure color vs shape bias for backprop model."""
    same_color_diff_shape = []
    diff_color_same_shape = []
    
    test_shapes = SHAPES[:4]
    test_colors = COLORS[:4]
    
    model.visual.eval()
    with torch.no_grad():
        for shape1 in test_shapes:
            for shape2 in test_shapes:
                if shape1 == shape2:
                    continue
                for color in test_colors:
                    img1 = create_stimulus(shape1, color, 'small')
                    img2 = create_stimulus(shape2, color, 'small')
                    
                    feat1 = model.visual(torch.from_numpy(img1).float())
                    feat2 = model.visual(torch.from_numpy(img2).float())
                    
                    sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
                    same_color_diff_shape.append(sim)
        
        for color1 in test_colors:
            for color2 in test_colors:
                if color1 == color2:
                    continue
                for shape in test_shapes:
                    img1 = create_stimulus(shape, color1, 'small')
                    img2 = create_stimulus(shape, color2, 'small')
                    
                    feat1 = model.visual(torch.from_numpy(img1).float())
                    feat2 = model.visual(torch.from_numpy(img2).float())
                    
                    sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
                    diff_color_same_shape.append(sim)
    
    model.visual.train()
    
    avg_same_color = np.mean(same_color_diff_shape)
    avg_diff_color = np.mean(diff_color_same_shape)
    
    return {
        'bias_score': avg_same_color / (avg_diff_color + 1e-8),
        'same_color_diff_shape': avg_same_color,
        'diff_color_same_shape': avg_diff_color,
    }


def measure_binding_rate_backprop(model: BackpropCrossModalLearner) -> float:
    """Measure cross-modal binding rate for backprop model."""
    aligned = 0
    total = 0
    
    model.visual.eval()
    with torch.no_grad():
        for shape in SHAPES[:4]:
            for color in COLORS[:4]:
                img = create_stimulus(shape, color, 'small')
                label = f'{color} {shape}'
                
                vis_feat = model.visual(torch.from_numpy(img).float())
                lang_feat = model.language(label)
                
                vis_concept = model.get_concept(vis_feat, 'visual')
                lang_concept = model.get_concept(lang_feat, 'language')
                
                total += 1
                if vis_concept == lang_concept:
                    aligned += 1
    
    model.visual.train()
    return aligned / total * 100


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_fair_comparison(n_seeds: int = 5, n_epochs: int = 50):
    """
    Run fair baseline comparison experiment.
    
    Compares:
    1. Brain-like (phased training) - original method
    2. Brain-like (simultaneous training) - artifact test
    3. Backprop autoencoder - fair baseline
    """
    print("=" * 70)
    print("FAIR BASELINE COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Epochs per run: {n_epochs}")
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {
        'brain_phased': [],
        'brain_simultaneous': [],
        'backprop_baseline': [],
    }
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print("="*70)
        
        # Brain-like with phased training
        print("\n  [1/3] Brain-like (phased)...")
        t0 = time.time()
        traj_phased = train_brain_phased(seed, n_epochs)
        print(f"        Done in {time.time()-t0:.1f}s, final bias={traj_phased['bias_scores'][-1]:.3f}")
        results['brain_phased'].append(traj_phased)
        
        # Brain-like with simultaneous training
        print("\n  [2/3] Brain-like (simultaneous)...")
        t0 = time.time()
        traj_simul = train_brain_simultaneous(seed, n_epochs)
        print(f"        Done in {time.time()-t0:.1f}s, final bias={traj_simul['bias_scores'][-1]:.3f}")
        results['brain_simultaneous'].append(traj_simul)
        
        # Backprop baseline
        print("\n  [3/3] Backprop autoencoder baseline...")
        t0 = time.time()
        traj_backprop = train_backprop_baseline(seed, n_epochs)
        print(f"        Done in {time.time()-t0:.1f}s, final bias={traj_backprop['bias_scores'][-1]:.3f}")
        results['backprop_baseline'].append(traj_backprop)
    
    # Statistical analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for condition, trajectories in results.items():
        final_biases = [t['bias_scores'][-1] for t in trajectories]
        final_bindings = [t['binding_rates'][-1] for t in trajectories]
        
        print(f"\n{condition.upper()}:")
        print(f"  Final Bias: {np.mean(final_biases):.3f} ± {np.std(final_biases):.3f}")
        print(f"  Final Binding: {np.mean(final_bindings):.1f}% ± {np.std(final_bindings):.1f}%")
        
        # Trajectory analysis
        early_biases = [t['bias_scores'][2] for t in trajectories]  # ~epoch 10
        print(f"  Early→Final Bias: {np.mean(early_biases):.3f} → {np.mean(final_biases):.3f}")
    
    # Statistical tests
    print("\n" + "-"*70)
    print("STATISTICAL COMPARISONS (t-tests)")
    print("-"*70)
    
    # Phased vs Simultaneous (artifact test)
    phased_final = [t['bias_scores'][-1] for t in results['brain_phased']]
    simul_final = [t['bias_scores'][-1] for t in results['brain_simultaneous']]
    t_stat, p_val = stats.ttest_ind(phased_final, simul_final)
    print(f"\nPhased vs Simultaneous (trajectory artifact test):")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  → SIGNIFICANT: Training order matters!")
    else:
        print(f"  → Not significant: Trajectory may be emergent")
    
    # Brain-like vs Backprop (fair comparison)
    brain_final = [t['bias_scores'][-1] for t in results['brain_phased']]
    backprop_final = [t['bias_scores'][-1] for t in results['backprop_baseline']]
    t_stat, p_val = stats.ttest_ind(brain_final, backprop_final)
    print(f"\nBrain-like vs Backprop (fair baseline test):")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  → SIGNIFICANT: Learning rule matters even with same objective!")
    else:
        print(f"  → Not significant: Same objective → similar bias")
    
    # Effect sizes (Cohen's d)
    print("\n" + "-"*70)
    print("EFFECT SIZES (Cohen's d)")
    print("-"*70)
    
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.std(x)**2 + (ny-1)*np.std(y)**2) / (nx+ny-2))
        return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)
    
    d_artifact = cohens_d(phased_final, simul_final)
    d_baseline = cohens_d(brain_final, backprop_final)
    
    print(f"  Phased vs Simultaneous: d={d_artifact:.3f}")
    print(f"  Brain-like vs Backprop: d={d_baseline:.3f}")
    
    # Save results
    save_dir = Path("fair_comparison_results")
    save_dir.mkdir(exist_ok=True)
    
    torch.save({
        'results': results,
        'config': {'n_seeds': n_seeds, 'n_epochs': n_epochs},
        'completed': datetime.now().isoformat(),
    }, save_dir / "results.pt")
    
    print(f"\n\nResults saved to: {save_dir}/results.pt")
    
    # Generate comparison figure
    generate_comparison_figure(results, save_dir)
    
    return results


def generate_comparison_figure(results: Dict, save_dir: Path):
    """Generate comparison figure."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = {
            'brain_phased': '#2ecc71',
            'brain_simultaneous': '#3498db', 
            'backprop_baseline': '#e74c3c',
        }
        labels = {
            'brain_phased': 'Brain-like (Phased)',
            'brain_simultaneous': 'Brain-like (Simultaneous)',
            'backprop_baseline': 'Backprop Autoencoder',
        }
        
        # Plot 1: Bias trajectory
        ax = axes[0]
        for condition, trajectories in results.items():
            epochs = trajectories[0]['epochs']
            biases = np.array([t['bias_scores'] for t in trajectories])
            mean = biases.mean(axis=0)
            std = biases.std(axis=0)
            
            ax.plot(epochs, mean, color=colors[condition], label=labels[condition], linewidth=2)
            ax.fill_between(epochs, mean-std, mean+std, color=colors[condition], alpha=0.2)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No bias')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Color Bias Score')
        ax.set_title('A) Developmental Trajectory')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Binding rate trajectory
        ax = axes[1]
        for condition, trajectories in results.items():
            epochs = trajectories[0]['epochs']
            bindings = np.array([t['binding_rates'] for t in trajectories])
            mean = bindings.mean(axis=0)
            std = bindings.std(axis=0)
            
            ax.plot(epochs, mean, color=colors[condition], label=labels[condition], linewidth=2)
            ax.fill_between(epochs, mean-std, mean+std, color=colors[condition], alpha=0.2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Binding Rate (%)')
        ax.set_title('B) Cross-Modal Binding')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Final comparison bar chart
        ax = axes[2]
        conditions = list(results.keys())
        x = np.arange(len(conditions))
        width = 0.35
        
        final_biases = [np.mean([t['bias_scores'][-1] for t in results[c]]) for c in conditions]
        final_biases_std = [np.std([t['bias_scores'][-1] for t in results[c]]) for c in conditions]
        
        bars = ax.bar(x, final_biases, width, yerr=final_biases_std, 
                     color=[colors[c] for c in conditions], capsize=5)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Final Color Bias')
        ax.set_title('C) Final Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['Phased', 'Simultaneous', 'Backprop'], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'fair_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir / 'fair_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Figure saved to: {save_dir}/fair_comparison.png")
        
    except ImportError:
        print("matplotlib not available for visualization")


if __name__ == "__main__":
    results = run_fair_comparison(n_seeds=5, n_epochs=50)
