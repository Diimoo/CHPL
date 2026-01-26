#!/usr/bin/env python3
"""
Mechanistic Analysis of Developmental Bias Emergence

Tests multiple competing hypotheses for why color bias emerges before shape bias:
H1: Gradient magnitude - color changes produce larger gradients
H2: Spatial frequency - MSE loss favors low-frequency (color) over high-frequency (edges)
H3: Architecture bias - conv layers favor certain feature types
H4: Initialization effects - random weights favor color over shape

Success metrics:
- Quantify gradient flow for color vs shape reconstruction
- Measure loss decomposition (color error vs edge error)
- Track feature activation patterns over training
- Compare to infant development data (if available)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

sys.path.append(str(Path(__file__).parent.parent))
from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import create_stimulus, SHAPES, COLORS, SIZES


@dataclass
class MechanisticMetrics:
    """
    Metrics for understanding bias emergence.
    
    NOTE: We use reconstruction loss as a PROXY for gradient signal strength.
    Lower loss = easier to reconstruct = stronger learning signal during training.
    This is an approximation - true per-example gradient tracking would require
    restructuring the training loop (see Option B in documentation).
    """
    epoch: int
    
    # H1: Reconstruction difficulty as proxy for gradient signal
    # Lower = easier to reconstruct = stronger gradient signal during training
    color_change_loss: float  # How hard to reconstruct color variations
    shape_change_loss: float  # How hard to reconstruct shape variations
    loss_ratio: float  # color_loss / shape_loss (< 1 means color is easier)
    
    # H2: Spatial frequency hypothesis
    low_freq_error: float  # Color-like (large spatial scale)
    high_freq_error: float  # Edge-like (small spatial scale)
    freq_ratio: float  # low_freq / high_freq
    
    # H3: Feature clustering - which dimension organizes representations?
    feature_similarity_within_color: float  # Do same-color items cluster?
    feature_similarity_within_shape: float  # Do same-shape items cluster?
    clustering_bias: float  # color_sim - shape_sim (> 0 means color-organized)
    
    # H4: Overall training gradient magnitude (from actual backward pass)
    total_gradient_magnitude: float


class MechanisticAnalyzer:
    """
    Instruments CHPL training to test mechanistic hypotheses.
    Hooks into actual training loop to capture real gradients.
    """
    
    def __init__(self, model: BrainCrossModalLearner):
        self.model = model
        self.metrics_history = []
        self.captured_gradients = {}  # Store gradients from actual backward pass
        self.test_pairs = None  # Cache test pairs
        
    def create_test_pairs(self, n_samples: int = 50) -> List[Tuple]:
        """
        Create controlled test pairs:
        - Same color, different shape
        - Same shape, different color
        - Same size, different color/shape
        """
        pairs = []
        
        # Sample diverse combinations
        for _ in range(n_samples):
            color = np.random.choice(COLORS)
            shape = np.random.choice(SHAPES)
            size = np.random.choice(SIZES)
            
            # Create base stimulus
            base_img = create_stimulus(shape, color, size)
            
            # Create color-change pair (same shape/size)
            other_color = np.random.choice([c for c in COLORS if c != color])
            color_change_img = create_stimulus(shape, other_color, size)
            
            # Create shape-change pair (same color/size)
            other_shape = np.random.choice([s for s in SHAPES if s != shape])
            shape_change_img = create_stimulus(other_shape, color, size)
            
            pairs.append({
                'base': base_img,
                'color_change': color_change_img,
                'shape_change': shape_change_img,
                'metadata': {'color': color, 'shape': shape, 'size': size}
            })
            
        return pairs
    
    def compute_spatial_frequency_error(self, img: torch.Tensor, recon: torch.Tensor) -> Tuple[float, float]:
        """
        Decompose reconstruction error by spatial frequency.
        Low freq = color-like, High freq = edge-like
        
        Both inputs should be [B, C, H, W] format.
        """
        # Ensure [B, C, H, W] format
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.shape[-1] == 3 and img.shape[1] != 3:
            img = img.permute(0, 3, 1, 2)
        
        if recon.dim() == 3:
            recon = recon.unsqueeze(0)
        if recon.shape[-1] == 3 and recon.shape[1] != 3:
            recon = recon.permute(0, 3, 1, 2)
        
        # Simple approach: Gaussian blur for low-freq, subtract for high-freq
        kernel_size = 5
        sigma = 2.0
        
        # Create Gaussian kernel
        kernel = self._gaussian_kernel(kernel_size, sigma).to(img.device)
        
        # Low-frequency components (blurred)
        img_low = F.conv2d(img, kernel, padding=kernel_size//2, groups=3)
        recon_low = F.conv2d(recon, kernel, padding=kernel_size//2, groups=3)
        low_freq_error = F.mse_loss(recon_low, img_low).item()
        
        # High-frequency components (residual)
        img_high = img - img_low
        recon_high = recon - recon_low
        high_freq_error = F.mse_loss(recon_high, img_high).item()
        
        return low_freq_error, high_freq_error
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian blur kernel"""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        
        # 2D kernel
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel / kernel.sum()
        
        # Expand to 3 channels (RGB)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        return kernel
    
    def capture_training_gradients(self) -> Dict[str, float]:
        """
        Capture gradients after backward pass during actual training.
        Call this AFTER loss.backward() but BEFORE optimizer.step()
        """
        grad_dict = {}
        # BrainCrossModalLearner is not an nn.Module, so iterate over its components
        for component_name in ['visual', 'language']:
            component = getattr(self.model, component_name, None)
            if component is not None and hasattr(component, 'named_parameters'):
                for name, param in component.named_parameters():
                    if param.grad is not None:
                        full_name = f"{component_name}.{name}"
                        grad_dict[full_name] = param.grad.norm().item()
        
        self.captured_gradients = grad_dict
        return grad_dict
    
    def get_visual_cortex_gradient_magnitude(self) -> float:
        """Extract average gradient magnitude for visual cortex layers"""
        visual_grads = [v for k, v in self.captured_gradients.items() 
                        if k.startswith('visual.')]
        return np.mean(visual_grads) if visual_grads else 0.0
    
    def compute_feature_similarity(self, features: torch.Tensor, 
                                   grouping: List[str]) -> float:
        """
        Compute average cosine similarity within a group.
        Features: [N, D] tensor
        Grouping: list of group labels for each feature
        """
        similarities = []
        unique_groups = list(set(grouping))
        
        for group in unique_groups:
            indices = [i for i, g in enumerate(grouping) if g == group]
            if len(indices) < 2:
                continue
                
            group_features = features[indices]
            # Pairwise cosine similarity
            norm_features = F.normalize(group_features, dim=1)
            sim_matrix = torch.mm(norm_features, norm_features.t())
            
            # Average off-diagonal (exclude self-similarity)
            mask = ~torch.eye(len(indices), dtype=torch.bool, device=features.device)
            similarities.append(sim_matrix[mask].mean().item())
        
        return np.mean(similarities) if similarities else 0.0
    
    def analyze_batch(self, batch_imgs: torch.Tensor, 
                      color_mask: torch.Tensor, 
                      shape_mask: torch.Tensor) -> Dict[str, float]:
        """
        Analyze a batch during training.
        Call this AFTER capturing gradients from actual backward pass.
        
        Args:
            batch_imgs: [B, C, H, W] input images
            color_mask: [B] boolean mask indicating color-change examples
            shape_mask: [B] boolean mask indicating shape-change examples
        """
        self.model.visual.eval()
        with torch.no_grad():
            features = self.model.visual(batch_imgs)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            recons = self.model.visual.reconstruct(features)
            
            # Per-example losses
            per_example_loss = F.mse_loss(recons, batch_imgs, reduction='none')
            per_example_loss = per_example_loss.mean(dim=[1, 2, 3])  # [B]
            
            # Separate by color vs shape changes
            color_losses = per_example_loss[color_mask]
            shape_losses = per_example_loss[shape_mask]
            
            # Spatial frequency analysis on a sample
            sample_img = batch_imgs[0:1]
            sample_recon = recons[0:1]
            low_freq, high_freq = self.compute_spatial_frequency_error(sample_img, sample_recon)
            
            return {
                'color_loss': color_losses.mean().item() if len(color_losses) > 0 else 0.0,
                'shape_loss': shape_losses.mean().item() if len(shape_losses) > 0 else 0.0,
                'low_freq_error': low_freq,
                'high_freq_error': high_freq,
                'visual_grad_mag': self.get_visual_cortex_gradient_magnitude()
            }
    
    def analyze_epoch(self, epoch: int, test_pairs: Optional[List[Dict]] = None) -> MechanisticMetrics:
        """
        Perform comprehensive analysis at a given epoch.
        Uses cached test pairs and accumulated batch statistics.
        """
        if test_pairs is None:
            test_pairs = self.test_pairs
        if test_pairs is None:
            raise ValueError("No test pairs available for analysis")
            
        self.model.visual.eval()
        
        # Accumulators
        low_freq_errors = []
        high_freq_errors = []
        color_losses = []
        shape_losses = []
        
        features_by_color = defaultdict(list)
        features_by_shape = defaultdict(list)
        
        with torch.no_grad():
            for pair_data in test_pairs:
                base = torch.tensor(pair_data['base'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                color_change = torch.tensor(pair_data['color_change'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                shape_change = torch.tensor(pair_data['shape_change'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                metadata = pair_data['metadata']
                
                # Reconstruction losses (no gradient computation)
                base_features = self.model.visual(base)
                if base_features.dim() == 1:
                    base_features = base_features.unsqueeze(0)
                base_recon = self.model.visual.reconstruct(base_features)
                # Convert base to same format for comparison
                base_cmp = base.permute(0, 3, 1, 2) if base.shape[-1] == 3 else base
                
                color_change_features = self.model.visual(color_change)
                if color_change_features.dim() == 1:
                    color_change_features = color_change_features.unsqueeze(0)
                color_change_recon = self.model.visual.reconstruct(color_change_features)
                if color_change_recon.dim() == 3:
                    color_change_recon = color_change_recon.unsqueeze(0)
                color_cmp = color_change.permute(0, 3, 1, 2) if color_change.shape[-1] == 3 else color_change
                color_losses.append(F.mse_loss(color_change_recon, color_cmp).item())
                
                shape_change_features = self.model.visual(shape_change)
                if shape_change_features.dim() == 1:
                    shape_change_features = shape_change_features.unsqueeze(0)
                shape_change_recon = self.model.visual.reconstruct(shape_change_features)
                if shape_change_recon.dim() == 3:
                    shape_change_recon = shape_change_recon.unsqueeze(0)
                shape_cmp = shape_change.permute(0, 3, 1, 2) if shape_change.shape[-1] == 3 else shape_change
                shape_losses.append(F.mse_loss(shape_change_recon, shape_cmp).item())
                
                # H2: Spatial frequency decomposition
                low_freq, high_freq = self.compute_spatial_frequency_error(base, base_recon)
                low_freq_errors.append(low_freq)
                high_freq_errors.append(high_freq)
                
                # H4: Feature grouping
                features_by_color[metadata['color']].append(base_features)
                features_by_shape[metadata['shape']].append(base_features)
        
        # Aggregate features for similarity analysis
        all_features_color = []
        color_labels = []
        for color, feat_list in features_by_color.items():
            all_features_color.extend(feat_list)
            color_labels.extend([color] * len(feat_list))
        
        all_features_shape = []
        shape_labels = []
        for shape, feat_list in features_by_shape.items():
            all_features_shape.extend(feat_list)
            shape_labels.extend([shape] * len(feat_list))
        
        if all_features_color:
            features_color_tensor = torch.cat(all_features_color, dim=0)
            color_similarity = self.compute_feature_similarity(features_color_tensor, color_labels)
        else:
            color_similarity = 0.0
        
        if all_features_shape:
            features_shape_tensor = torch.cat(all_features_shape, dim=0)
            shape_similarity = self.compute_feature_similarity(features_shape_tensor, shape_labels)
        else:
            shape_similarity = 0.0
        
        # Compute derived metrics
        avg_color_loss = np.mean(color_losses)
        avg_shape_loss = np.mean(shape_losses)
        avg_low_freq = np.mean(low_freq_errors)
        avg_high_freq = np.mean(high_freq_errors)
        
        return MechanisticMetrics(
            epoch=epoch,
            # H1: Reconstruction difficulty proxy
            color_change_loss=avg_color_loss,
            shape_change_loss=avg_shape_loss,
            loss_ratio=avg_color_loss / (avg_shape_loss + 1e-8),
            # H2: Spatial frequency
            low_freq_error=avg_low_freq,
            high_freq_error=avg_high_freq,
            freq_ratio=avg_low_freq / (avg_high_freq + 1e-8),
            # H3: Feature clustering
            feature_similarity_within_color=color_similarity,
            feature_similarity_within_shape=shape_similarity,
            clustering_bias=color_similarity - shape_similarity,
            # H4: Actual gradient from training (if captured)
            total_gradient_magnitude=self.get_visual_cortex_gradient_magnitude()
        )
    
    def save_results(self, output_dir: Path):
        """Save analysis results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw metrics
        metrics_data = [vars(m) for m in self.metrics_history]
        with open(output_dir / "mechanistic_metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Generate plots
        self.plot_results(output_dir)
    
    def plot_results(self, output_dir: Path):
        """Generate visualization of mechanistic analysis"""
        if not self.metrics_history:
            return
        
        epochs = [m.epoch for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Mechanistic Analysis of Developmental Bias', fontsize=16)
        
        # H1: Reconstruction difficulty (proxy for gradient signal)
        ax = axes[0, 0]
        ax.plot(epochs, [m.color_change_loss for m in self.metrics_history], 
                label='Color-change loss', linewidth=2, color='red')
        ax.plot(epochs, [m.shape_change_loss for m in self.metrics_history], 
                label='Shape-change loss', linewidth=2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss (MSE)')
        ax.set_title('H1: Reconstruction Difficulty\n(Lower = Easier = Stronger Gradient Signal)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # H2: Spatial frequency errors
        ax = axes[0, 1]
        ax.plot(epochs, [m.low_freq_error for m in self.metrics_history], 
                label='Low-freq (color-like)', linewidth=2)
        ax.plot(epochs, [m.high_freq_error for m in self.metrics_history], 
                label='High-freq (edge-like)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('H2: Spatial Frequency Decomposition')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # H3: Feature clustering bias
        ax = axes[0, 2]
        ax.plot(epochs, [m.feature_similarity_within_color for m in self.metrics_history], 
                label='Within-color similarity', linewidth=2, color='red')
        ax.plot(epochs, [m.feature_similarity_within_shape for m in self.metrics_history], 
                label='Within-shape similarity', linewidth=2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('H3: Feature Clustering\n(Higher = Stronger Organization)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # H4: Clustering bias over time
        ax = axes[1, 0]
        ax.plot(epochs, [m.clustering_bias for m in self.metrics_history], 
                linewidth=2, color='green')
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='No bias')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Color Sim - Shape Sim')
        ax.set_title('H4: Clustering Bias\n(> 0 = Color-organized)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Loss ratio (color/shape) - key hypothesis test
        ax = axes[1, 1]
        ax.plot(epochs, [m.loss_ratio for m in self.metrics_history], 
                linewidth=2, color='purple')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal difficulty')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Color Loss / Shape Loss')
        ax.set_title('Loss Ratio Over Training\n(< 1 = Color Easier)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Frequency ratio (low/high)
        ax = axes[1, 2]
        freq_ratios = [m.low_freq_error / (m.high_freq_error + 1e-8) 
                       for m in self.metrics_history]
        ax.plot(epochs, freq_ratios, linewidth=2, color='orange')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Low-freq Error / High-freq Error')
        ax.set_title('Spatial Frequency Ratio')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mechanistic_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'mechanistic_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Mechanistic analysis plots saved to {output_dir}")


if __name__ == "__main__":
    print("="*60)
    print("MECHANISTIC ANALYSIS OF DEVELOPMENTAL BIAS")
    print("="*60)
    
    # Create model
    model = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    analyzer = MechanisticAnalyzer(model)
    
    # Create test pairs once
    print("\nGenerating test stimulus pairs...")
    test_pairs = analyzer.create_test_pairs(n_samples=50)
    analyzer.test_pairs = test_pairs  # Cache for later use
    print(f"✓ Created {len(test_pairs)} test pairs")
    
    # Analyze at epoch 0 (random initialization)
    print("\nEpoch 0: Analyzing random initialization...")
    metrics = analyzer.analyze_epoch(epoch=0)
    analyzer.metrics_history.append(metrics)
    
    print(f"  Color-change loss: {metrics.color_change_loss:.6f}")
    print(f"  Shape-change loss: {metrics.shape_change_loss:.6f}")
    print(f"  Loss ratio: {metrics.loss_ratio:.3f} (< 1 means color easier)")
    print(f"  Low-freq error: {metrics.low_freq_error:.6f}")
    print(f"  High-freq error: {metrics.high_freq_error:.6f}")
    print(f"  Freq ratio: {metrics.freq_ratio:.3f}")
    print(f"  Clustering bias: {metrics.clustering_bias:.3f} (> 0 means color-organized)")
    
    # Save results
    output_dir = Path("results/mechanistic_analysis")
    analyzer.save_results(output_dir)
    
    print("\n" + "="*60)
    print("SUCCESS CRITERIA (Option A - Reconstruction Proxy):")
    print("="*60)
    print("H1: loss_ratio < 1 early → color easier to reconstruct")
    print("H2: freq_ratio decreases over time → low-freq learned first")
    print("H3: clustering_bias > 0 early → features organized by color first")
    print("")
    print("LIMITATION: These are proxies for gradient signal, not true")
    print("per-example gradient tracking. See Option B for that approach.")
    print("\nNext step: Integrate into training loop to track over epochs")
