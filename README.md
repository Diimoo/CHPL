# Brain-Like Visual Learning

Official code for the paper:
> **Brain-Like Visual Learning Shows Emergent Developmental Bias and Implicit Regularization**  
> Ahmed Trabelsi  
> *arXiv preprint arXiv:XXXX.XXXXX (2025)*

## Overview

We investigate whether brain-inspired plasticity rules can achieve cross-modal learning from scratch in a controlled synthetic environment. Using reconstruction-based predictive coding and Hebbian consolidation, our model (CHPL) learns to bind visual shapes with linguistic labels without supervised classification.

### Key Findings

1. **Reduced Color Bias**: Brain-inspired learning (1.23 ± 0.09) shows significantly less categorization bias than supervised backpropagation (2.25 ± 0.48), $p < 0.00001$.
2. **Emergent Development**: Naturally recapitulates the infant color-to-shape bias trajectory over 100 epochs (neutral → peak color bias → reduction).
3. **Mechanistic Insights**: ATL consolidation is identified as critical for cross-modal alignment (+41% improvement, $d=1.44$).

## Installation

```bash
pip install torch numpy matplotlib scipy
```

Requires Python 3.8+ and PyTorch 2.0+ (CUDA recommended for faster validation).

## Quick Start

```python
from brain_crossmodal_learner import BrainCrossModalLearner
from synthetic_environment import generate_training_pairs

# Create model
model = BrainCrossModalLearner(
    visual_dim=64,
    language_dim=64,
    n_concepts=100
)

# Train (Simplified)
pairs = generate_training_pairs(n_samples=1000)
for img, label in pairs:
    # Phase-dependent training steps handled internally or via scripts
    pass

# Evaluate
# See validation_study.py for full evaluation protocols
```

## Reproducing Paper Results

### 1. Main Results (Ablation Study)
```bash
python validation_study.py
```
Runs 10 random seeds across 4 conditions (Full, No-Recon, No-Consol, Backprop).  
Results saved to `results/validation_results/`.

### 2. Developmental Trajectory
```bash
python extended_training.py
```
Tracks bias emergence over 100 epochs (3 seeds).  
Results saved to `results/extended_training_results/`.

### 3. Generate Figures
```bash
python generate_figures.py
```
Generates publication-quality figures:
- `figure2_ablation_results.png`
- `figure3_developmental_trajectory.png`

## Project Structure

- `brain_crossmodal_learner.py`: Core model implementation (CHPL).
- `synthetic_environment.py`: Synthetic visual-linguistic stimulus generator.
- `validation_study.py`: Statistical validation script (10 seeds).
- `extended_training.py`: Long-term trajectory tracking.
- `generate_figures.py`: Figure generation script.
- `paper/`: LaTeX and Markdown drafts of the paper.
- `results/`: Pre-computed validation and trajectory data.

## Citation

```bibtex
@article{trabelsi2025brain,
  title={Brain-Like Visual Learning Shows Emergent Developmental Bias and Implicit Regularization},
  author={Trabelsi, Ahmed},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Related Framework

For a more comprehensive, modular brain simulation framework supporting multiple regions and disease modes, see the [Digital Brain](https://github.com/Diimoo/CHPL) (Extended Version).
