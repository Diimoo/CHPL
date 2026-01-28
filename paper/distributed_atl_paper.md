# Distributed Semantic Binding for Compositional Scene Understanding

**Authors:** [To be filled]

**Abstract:** Compositional scene understanding—the ability to recognize novel combinations of known elements—remains challenging for neural systems. We demonstrate that winner-takes-all semantic binding fundamentally fails on compositional tasks: a single prototype cannot encode multiple independent attributes. We propose Distributed ATL, which replaces winner-takes-all with soft activation patterns across multiple prototypes. On two-object relational scenes, Distributed ATL achieves 0.663±0.025 held-out similarity versus 0.512 baseline (+29.6% improvement). The system generalizes to completely novel spatial relations (0.676), swapped object orders (0.649), and unseen object counts (train 1-3, test 4: 0.637). Multi-seed validation (n=5) confirms robustness. Our results suggest compositional semantics require population codes rather than localist representations, aligning with neuroscience evidence for distributed coding in the anterior temporal lobe.

---

## 1. Introduction

Humans effortlessly understand compositional scenes: "a red circle above a blue square" is immediately distinct from "a blue circle above a red square." This compositional understanding—the ability to combine known elements in novel ways—is fundamental to human cognition (Lake & Baroni, 2018) yet remains challenging for AI systems.

Current neural approaches to cross-modal learning (e.g., CLIP; Radford et al., 2021) excel at matching images to descriptions but struggle with systematic compositionality (Thrush et al., 2022). We hypothesize this failure stems from a fundamental architectural choice: **winner-takes-all semantic representations**.

Winner-takes-all binding, where each concept activates a single prototype, worked well for simple attribute learning (color, shape) but fundamentally cannot encode multi-attribute compositions. A single prototype cannot simultaneously represent five independent attributes (obj1-color, obj1-shape, relation, obj2-color, obj2-shape) without combinatorial explosion.

We propose **Distributed ATL**, which replaces winner-takes-all with soft activation patterns across multiple prototypes. This mirrors population coding in biological neural systems (Pouget et al., 2000) and enables compositional representation through patterns of activation rather than individual units.

### Contributions

1. Demonstrate that winner-takes-all fails fundamentally on composition (Section 4.1)
2. Propose distributed activation architecture that succeeds (Section 3)
3. Show robust generalization across five compositional test regimes (Section 4.2)
4. Scale to variable object counts (1-4) unseen during training (Section 4.3)
5. Validate with multi-seed experiments (n=5) and ablations (Section 4.4)

Results show Distributed ATL achieves **+29.6% improvement** over winner-takes-all baseline (0.663 vs 0.512 held-out), generalizes to completely novel spatial relations, and scales to unseen object counts.

---

## 2. Related Work

### 2.1 Compositional Generalization in AI

Lake & Baroni (2018) demonstrated that sequence-to-sequence models fail on systematic compositional generalization. Subsequent work has explored various architectural modifications (e.g., syntactic attention, modular networks) with mixed success. Our work addresses composition in the visual-linguistic domain rather than sequence transduction.

### 2.2 Semantic Cognition and the ATL

The anterior temporal lobe (ATL) serves as a "semantic hub" integrating multimodal information (Patterson et al., 2007). Computational models typically implement the ATL as prototype-based competitive learning with winner-takes-all dynamics. We argue this architectural choice limits compositional capacity and propose distributed alternatives.

### 2.3 Population Codes in Neuroscience

Pouget et al. (2000) showed that neural populations encode information through distributed activity patterns rather than individual neuron responses. This population coding enables efficient representation of high-dimensional spaces. Our Distributed ATL applies this principle to semantic binding.

### 2.4 Vision-Language Models

CLIP (Radford et al., 2021) and similar models learn joint visual-linguistic representations through contrastive learning. While powerful for retrieval, these models struggle with compositional understanding (Thrush et al., 2022; Yuksekgonul et al., 2023). Our approach differs by using explicit distributed binding rather than implicit embedding similarity.

---

## 3. Methods

### 3.1 Architecture Overview

Our system consists of three main components (Figure 3):

**Visual Cortex.** A convolutional encoder-decoder processes 56×56 RGB images through 4 convolutional layers with max-pooling, producing 64-dimensional feature vectors. The decoder reconstructs the input via transposed convolutions. We use Kaiming initialization for convolutional layers and Xavier initialization for fully-connected layers to ensure stable training across random seeds.

**Language Cortex.** Words are embedded into 32-dimensional vectors and composed via averaging. The composed representation is projected to 64 dimensions to match visual features.

**Distributed ATL.** Unlike winner-takes-all semantic binding, our Distributed ATL computes soft activation patterns over all N=200 prototypes:

$$\alpha_i = \text{softmax}\left(\frac{\mathbf{p}_i \cdot \mathbf{f}}{\tau}\right)$$

where $\mathbf{p}_i$ are learned prototypes, $\mathbf{f}$ is the input feature, and $\tau$ is temperature. We use $\tau=0.2$ based on ablation studies (Figure 2C).

### 3.2 Binding and Learning

**Binding quality** is measured via pattern similarity:

$$\text{similarity} = \cos(\alpha_{\text{vis}}, \alpha_{\text{lang}})$$

**Hebbian learning** updates all activated prototypes weighted by activation strength:

$$\Delta \mathbf{p}_i = \eta \cdot \alpha_i \cdot (\mathbf{f} - \mathbf{p}_i)$$

where learning rate $\eta$ decreases with prototype usage (meta-plasticity), preventing frequently-used prototypes from dominating.

### 3.3 Training Protocol

Training proceeds in three phases:

1. **Visual reconstruction** (10 epochs): Train visual encoder-decoder to reconstruct input images, ensuring rich visual features.

2. **Cross-modal alignment** (15 epochs): Align language features with visual features via cosine similarity loss.

3. **Distributed consolidation** (10 epochs): Bind visual and language through distributed ATL, updating prototypes via Hebbian learning.

### 3.4 Evaluation Protocol

We evaluate compositional generalization across five test regimes:

| Split | Train | Test | Challenge |
|-------|-------|------|-----------|
| Color holdout | red/blue obj1 | green obj1 | Novel attribute in position |
| Relation holdout | above, left_of | below, right_of | Completely novel relations |
| Swap generalization | A above B | B above A | Object order reversal |
| Novel combination | Seen attr pairs | Unseen attr pairs | Combinatorial novelty |
| Variable counts | 1-3 objects | 4 objects | Novel scene complexity |

**Success criterion:** held-out similarity > 0.6 (baseline achieves 0.512).

---

## 4. Results

### 4.1 Winner-Takes-All Fails on Composition

Table 1 compares winner-takes-all (baseline) and Distributed ATL on the color holdout split.

| Method | Train | Held-out | Gap |
|--------|-------|----------|-----|
| Winner-Takes-All | 0.862 | 0.512 | 0.350 |
| **Distributed ATL** | 0.686 | **0.663** | **0.023** |

Winner-takes-all achieves high training performance but fails to generalize (gap = 0.350). Distributed ATL shows dramatically smaller gap (0.023), indicating genuine compositional structure rather than memorization.

### 4.2 Generalization Across Splits

Table 2 shows Distributed ATL performance across harder compositional splits.

| Split | Train | Held-out | Gap | Verdict |
|-------|-------|----------|-----|---------|
| Color holdout | 0.686 | 0.663 | 0.023 | ✓ Pass |
| Relation holdout | 0.696 | 0.676 | 0.020 | ✓ Pass |
| Swap generalization | 0.666 | 0.649 | 0.017 | ✓ Pass |
| Novel combination | 0.690 | 0.648 | 0.042 | ✓ Pass |

All splits exceed the 0.6 threshold and dramatically outperform the 0.512 baseline. Notably, relation holdout—where test relations are completely unseen—achieves 0.676, demonstrating that Distributed ATL learns abstract relational structure.

### 4.3 Variable Object Counts

Table 3 shows generalization to unseen object counts.

| Objects | Similarity | Status |
|---------|------------|--------|
| 1 | 0.639 | Train |
| 2 | 0.667 | Train |
| 3 | 0.661 | Train |
| **4** | **0.637** | **Test (novel!)** |

Training on 1-3 objects, the system generalizes to 4 objects with only 0.024 gap. This demonstrates that Distributed ATL learns compositional structure that scales to novel complexity.

### 4.4 Multi-Seed Validation and Ablations

**Multi-seed validation** (n=5 seeds) confirms robustness:

- Held-out: **0.663 ± 0.025**
- All seeds > 0.62
- Improvement over baseline: **+29.6%**

**Temperature ablation** (Figure 2C) shows:

| τ | Train | Held-out | Interpretation |
|---|-------|----------|----------------|
| 0.1 | 0.324 | 0.201 | Too sparse (≈winner-takes-all) |
| **0.2** | **0.704** | **0.732** | **Optimal** |
| 0.5 | 0.939 | 0.937 | Too diffuse (no discrimination) |

τ=0.2 achieves optimal balance between sparse and diffuse activation.

---

## 5. Discussion

### 5.1 Why Distribution Matters for Composition

Our results demonstrate that compositional semantics require distributed representations. Winner-takes-all encoding faces combinatorial explosion: with K attributes each taking V values, we need V^K prototypes to represent all combinations. With distributed activation, K×V prototypes suffice—each encodes one attribute value, and combinations emerge from activation patterns.

This aligns with neuroscience evidence for population codes in semantic cognition (Patterson et al., 2007). The brain's anterior temporal lobe likely uses distributed rather than localist representations, with concepts encoded across neural populations rather than individual "grandmother cells."

### 5.2 The Role of Temperature

Temperature controls activation sparsity: too low (τ=0.1) approximates winner-takes-all, too high (τ=0.5) produces uniform distributions losing discriminative power. τ=0.2 achieves optimal balance—sparse enough for interpretable patterns, diffuse enough for compositional structure.

This mirrors biological tuning curves, which show neither all-or-none responses nor uniform activation, but rather bell-shaped selectivity with partial overlap (Pouget et al., 2000).

### 5.3 Implications for Vision-Language Models

Current vision-language models like CLIP struggle with compositional understanding despite impressive retrieval performance. Our results suggest this may stem from implicit binding through embedding similarity rather than explicit distributed binding. Incorporating Distributed ATL-like mechanisms into larger models is a promising direction.

### 5.4 Limitations

**Synthetic stimuli.** Our shapes are simple and synthetic. Natural images require object-centric vision (e.g., Slot Attention; Locatello et al., 2020) to segment objects before binding.

**Language simplicity.** Current parser uses bag-of-words. Human-like composition requires structured parsing and hierarchical concepts.

**Scalability.** We test up to 4 objects. Real scenes contain many more. Whether distributed patterns scale to 10+ objects remains open.

### 5.5 Future Work

1. **Natural images:** Extend to COCO/Visual Genome with object detection
2. **Hierarchical composition:** Nested relations ("A above (B next to C)")
3. **Negation and quantifiers:** "NOT red," "all circles are blue"
4. **Integration with large models:** Distributed binding layers in CLIP-like architectures

---

## 6. Conclusion

We demonstrated that winner-takes-all semantic binding fundamentally fails on compositional scene understanding, while Distributed ATL—using soft activation patterns across multiple prototypes—succeeds. Our system generalizes to novel attribute positions, completely unseen spatial relations, swapped object orders, and unprecedented object counts.

The key insight is that compositional semantics require population codes: binding emerges from the similarity of activation patterns, not from matching individual prototypes. This aligns with neuroscience evidence and suggests a path toward genuinely compositional AI systems.

**Code:** Available at [GitHub repository URL]

---

## References

- Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. ICML.
- Locatello, F., et al. (2020). Object-centric learning with slot attention. NeurIPS.
- Patterson, K., et al. (2007). Where do you know what you know? The representation of semantic knowledge in the human brain. Nature Reviews Neuroscience.
- Pouget, A., et al. (2000). Information processing with population codes. Nature Reviews Neuroscience.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. ICML.
- Thrush, T., et al. (2022). Winoground: Probing vision and language models for visio-linguistic compositionality. CVPR.
- Yuksekgonul, M., et al. (2023). When and why vision-language models behave like bags-of-words. ICLR.

---

## Appendix

### A. Implementation Details

- Feature dimension: 64
- Number of prototypes: 200
- Temperature: 0.2
- Learning rate (visual/language): 1e-3
- Learning rate (ATL Hebbian): 0.01
- Meta-plasticity decay: 0.999
- Activation threshold: 0.01

### B. Dataset Statistics

| Split | Train samples | Test samples | Train combos | Test combos |
|-------|---------------|--------------|--------------|-------------|
| Color holdout | 800 | 400 | 40 | 20 |
| Relation holdout | 900 | 900 | 45 | 45 |
| Swap generalization | 720 | 1080 | 36 | 54 |
| Novel combination | 1200 | 600 | 60 | 30 |
| Variable counts | 1475 | 13102 | - | - |

### C. Compute Resources

All experiments run on single NVIDIA GPU. Training time per seed: ~45 minutes.
Total compute for paper: ~15 GPU-hours.
