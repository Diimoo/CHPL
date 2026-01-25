# Brain-Like Cross-Modal Learning: Scientific Findings

## Overview

This study implements brain-like learning rules (reconstruction-based predictive coding, Hebbian consolidation, hippocampal episodic binding) in a simplified cross-modal environment and discovers emergent properties matching infant developmental psychology.

## Key Finding: Color > Shape Bias

Brain-like learning with pixel-level reconstruction shows **color bias** (bias score 1.24) - the system categorizes by color before shape, matching infant development literature:

- **Bornstein et al. (1976)**: Infants show color categorization before shape
- **Smith & Heise (1992)**: Shape bias emerges around 24 months, color dominates earlier
- **Landau et al. (1988)**: Color is perceptually more salient for young learners

## Ablation Study Results

| Condition | Bias Score | Binding Rate | Concepts | Key Finding |
|-----------|------------|--------------|----------|-------------|
| Full Model | 1.24 | 68.8% | 2 | Baseline |
| No Reconstruction | 1.02 | 100%* | 1 | *Collapse - no learning |
| No Consolidation | 1.22 | 43.8% | 3 | Binding -36% |
| Backprop Baseline | 1.58 | N/A | N/A | +28% MORE biased |

## Scientific Insights

### 1. Reconstruction is Necessary for Learning
Without reconstruction loss, visual features remain random and collapse to a single concept. The 100% binding rate is meaningless - everything maps to the same place.

### 2. Color Bias Emerges During Training
```
Epoch 1:  bias = 0.88 (SHAPE bias initially!)
Epoch 2:  bias = 1.08 (Color bias emerges)
Epoch 5+: bias = 1.24 (Stabilizes)
```
This developmental trajectory mirrors infant visual development.

### 3. Brain-Like Learning is MORE Balanced than Backprop
- Brain-like: 1.24 color bias
- Backprop: 1.58 color bias (+28% stronger)

Supervised backprop with classification loss produces STRONGER color bias than brain-like reconstruction learning.

### 4. ATL Consolidation Improves Cross-Modal Binding
- With consolidation: 68.8% binding
- Without consolidation: 43.8% binding
- Improvement: +57%

The Hebbian consolidation in ATL is critical for aligning visual and language representations.

## Mechanism: Why Color Dominates

```python
# Pixel-level reconstruction error:
red_circle vs blue_circle:  MSE ≈ 0.85  (many pixels differ)
red_circle vs red_square:   MSE ≈ 0.12  (only edge pixels differ)

# Gradient magnitude:
∇L_color >> ∇L_shape
```

Color changes affect MORE pixels than shape changes, so reconstruction loss naturally learns color features first.

## Files

- `brain_crossmodal_learner.py` - Main brain-like learning system
- `synthetic_environment.py` - Controlled stimulus generation
- `ablation_study.py` - Ablation experiments
- `ablation_results.png` - Visualization

## Future Directions

1. **Compositional Learning**: Learn color AND shape as separate features
2. **Longer Training**: Does shape bias emerge later (like 24-month infants)?
3. **Architecture Variations**: Different reconstruction targets
4. **Comparison to Human Data**: Match developmental timelines

## Citation

If using this work, please cite the developmental psychology literature that inspired it:
- Bornstein, M. H. (1976). Infants are trichromats. Journal of Experimental Child Psychology.
- Smith, L. B., & Heise, D. (1992). Perceptual similarity and conceptual structure.
- Landau, B., Smith, L. B., & Jones, S. S. (1988). The importance of shape in early lexical learning.
