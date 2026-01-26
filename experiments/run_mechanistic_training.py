#!/usr/bin/env python3
"""
Minimal training run with mechanistic analysis.
Tracks bias emergence over 10 epochs to test hypotheses.

Run time: ~2-5 minutes depending on hardware.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))
from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import generate_training_pairs
from mechanistic_analysis import MechanisticAnalyzer

def run_phase1_training(model: BrainCrossModalLearner, 
                        analyzer: MechanisticAnalyzer,
                        n_epochs: int = 10,
                        n_samples: int = 200):
    """
    Run Phase 1 (visual reconstruction) with mechanistic tracking.
    """
    print(f"\nPhase 1: Visual Reconstruction Training")
    print(f"  Epochs: {n_epochs}")
    print(f"  Samples/epoch: {n_samples}")
    print("-" * 50)
    
    # Generate training data
    pairs = generate_training_pairs(n_samples)
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.visual.train()
        
        epoch_losses = []
        for pair in pairs:
            img_tensor = torch.tensor(pair.image, dtype=torch.float32).to(DEVICE)
            
            # Phase 1: Reconstruction only
            model.cortex_optimizer.zero_grad()
            recon_loss = model.visual.reconstruction_loss(img_tensor)
            recon_loss.backward()
            
            # Capture gradients from actual training
            analyzer.capture_training_gradients()
            
            model.cortex_optimizer.step()
            epoch_losses.append(recon_loss.item())
        
        # Analyze at epoch end
        metrics = analyzer.analyze_epoch(epoch=epoch)
        analyzer.metrics_history.append(metrics)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d} | Loss: {np.mean(epoch_losses):.4f} | "
              f"Color/Shape: {metrics.loss_ratio:.3f} | "
              f"Cluster bias: {metrics.clustering_bias:+.3f} | "
              f"Time: {epoch_time:.1f}s")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("MECHANISTIC TRAINING ANALYSIS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Create model and analyzer
    model = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    analyzer = MechanisticAnalyzer(model)
    
    # Create and cache test pairs
    print("\nGenerating test pairs for analysis...")
    analyzer.test_pairs = analyzer.create_test_pairs(n_samples=30)
    print(f"✓ Created {len(analyzer.test_pairs)} test pairs")
    
    # Run training with tracking
    start_time = time.time()
    model = run_phase1_training(model, analyzer, n_epochs=10, n_samples=200)
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path("results/mechanistic_training")
    analyzer.save_results(output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if len(analyzer.metrics_history) >= 2:
        first = analyzer.metrics_history[0]
        last = analyzer.metrics_history[-1]
        
        print(f"\nH1 (Reconstruction difficulty):")
        print(f"  Start: color/shape ratio = {first.loss_ratio:.3f}")
        print(f"  End:   color/shape ratio = {last.loss_ratio:.3f}")
        print(f"  → {'Color easier' if last.loss_ratio < 1 else 'Shape easier or equal'}")
        
        print(f"\nH2 (Spatial frequency):")
        print(f"  Start: freq ratio = {first.freq_ratio:.3f}")
        print(f"  End:   freq ratio = {last.freq_ratio:.3f}")
        print(f"  → {'Low-freq learned faster' if last.freq_ratio < first.freq_ratio else 'High-freq caught up'}")
        
        print(f"\nH3 (Feature clustering):")
        print(f"  Start: clustering bias = {first.clustering_bias:+.3f}")
        print(f"  End:   clustering bias = {last.clustering_bias:+.3f}")
        print(f"  → {'Color-organized' if last.clustering_bias > 0.05 else 'Shape-organized' if last.clustering_bias < -0.05 else 'Mixed/neutral'}")
    
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")
