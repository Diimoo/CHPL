# Brain-Like Visual Learning Shows Emergent Developmental Bias and Implicit Regularization

**Ahmed Trabelsi**  
hi@locentia.com


---

## Abstract

Biological learning mechanisms remain poorly understood compared to supervised deep learning. We investigate whether brain-inspired plasticity rules can achieve cross-modal learning in a controlled synthetic environment. Using reconstruction-based predictive coding and Hebbian consolidation, our model learns to bind visual shapes with linguistic labels. Statistical validation (n=10) reveals reduced color bias (1.23 ± 0.09) compared to backpropagation (2.25 ± 0.48; t=-6.32, p<0.00001). Extended training shows an emergent developmental trajectory from neutral (0.99) through peak color bias (1.32) to reduction (1.07), recapitulating infant development. Ablation studies identify reconstruction as necessary and ATL consolidation as critical (+41% binding, p=0.01, d=1.44). Our work demonstrates brain-inspired learning achieves meaningful representations while recapitulating developmental phenomena.

**Keywords:** brain-like learning, cross-modal binding, developmental trajectory, Hebbian plasticity, predictive coding, color bias

---

## 1. Introduction

Understanding how the brain learns to integrate information across sensory modalities remains a fundamental challenge in cognitive neuroscience. Humans effortlessly learn to associate visual objects with their linguistic labels, enabling rich conceptual representations that support reasoning and communication. While deep learning has achieved remarkable success in multimodal learning through supervised training on large datasets (Radford et al., 2021), the biological mechanisms underlying cross-modal binding in the brain operate under fundamentally different constraints.

Biological neural networks do not have access to externally-provided error gradients. Instead, learning proceeds through local plasticity rules such as Hebbian learning (Hebb, 1949), modulated by global neuromodulatory signals and constrained by the brain's hierarchical architecture. The anterior temporal lobe (ATL) has been identified as a critical hub for semantic memory, integrating information from visual, auditory, and linguistic cortices into unified conceptual representations (Patterson et al., 2007; Lambon Ralph et al., 2017). Meanwhile, predictive coding theories suggest that cortical hierarchies learn by minimizing prediction errors through recurrent message passing (Rao & Ballard, 1999; Friston, 2005).

A particularly intriguing aspect of human visual development is the systematic progression of perceptual biases. Infants initially show strong color categorization before shape (Bornstein, 1976), with shape bias only emerging around 24 months of age (Smith & Heise, 1992; Landau et al., 1988). This developmental trajectory has been extensively studied in psychology, yet computational models that recapitulate this phenomenon through biologically-plausible learning remain scarce.

In this work, we ask: **Can brain-inspired plasticity rules achieve cross-modal learning from scratch, and if so, what emergent properties arise?** We address this question by developing a simplified neural architecture inspired by visual cortex, language cortex, and ATL, trained using reconstruction-based predictive coding and Hebbian consolidation. Our key contributions are:

1. We demonstrate that brain-like learning achieves meaningful cross-modal binding (62.5% alignment) using only local plasticity rules, without supervised classification.

2. We show that brain-like learning exhibits **significantly less color bias** than supervised backpropagation (1.23 vs 2.25, p<0.00001), suggesting implicit regularization from biological constraints.

3. We discover an **emergent developmental trajectory** where color bias rises and then falls (neutral → peak → reduction), recapitulating infant visual development without explicit programming.

4. Through systematic ablations, we identify **mechanistic requirements**: reconstruction loss prevents collapse; ATL consolidation is critical for binding.

---

## 2. Related Work

### 2.1 Multimodal Learning in Deep Learning

Recent advances in multimodal learning have been driven by large-scale supervised approaches. CLIP (Radford et al., 2021) learns visual-linguistic alignments from 400 million image-text pairs using contrastive learning. Similar approaches include ALIGN (Jia et al., 2021) and FLAVA (Singh et al., 2022). While highly effective, these models require massive datasets and compute, and their learning mechanisms differ fundamentally from biological systems.

### 2.2 Biologically-Inspired Learning

Self-organizing maps (SOMs; Kohonen, 1982) implement competitive learning similar to our ATL module, forming topographic representations through winner-take-all dynamics. Adaptive Resonance Theory (ART; Grossberg, 1987) addresses the stability-plasticity dilemma in unsupervised learning. More recently, predictive coding networks (Whittington & Bogacz, 2017) have shown that reconstruction-based learning can approximate backpropagation under certain conditions. Our work combines these principles—competitive learning, reconstruction, and Hebbian plasticity—in a cross-modal setting.

### 2.3 Computational Models of Development

Computational models of infant visual development have explored the emergence of perceptual biases. Rogers & McClelland (2004) modeled semantic development using connectionist networks. Sloutsky (2010) proposed that early perception is similarity-based rather than category-based. Colunga & Smith (2005) showed that shape bias in word learning emerges from statistical regularities in the environment. Our model extends this work by demonstrating that developmental trajectories can emerge from biologically-plausible learning rules without explicit environmental statistics.

### 2.4 Cross-Modal Binding in Neuroscience

The anterior temporal lobe has been identified as a "hub" for multimodal semantic integration (Patterson et al., 2007; Lambon Ralph et al., 2017). Neuroimaging studies show ATL activation during cross-modal tasks (Visser et al., 2010). Our simplified ATL module, implementing competitive learning with shared prototypes, captures the essential computational function while remaining biologically interpretable.

---

## 3. Methods

### 3.1 Synthetic Environment

To enable controlled experimentation, we designed a synthetic visual-linguistic environment with known ground truth. Visual stimuli consist of 28×28 RGB images containing geometric shapes (circle, square, triangle, star, pentagon, hexagon) rendered in different colors (red, blue, green, yellow, purple, orange) and sizes (small, medium, large). Each stimulus is paired with a linguistic label of the form "[color] [shape]" (e.g., "red circle"). This yields 108 unique combinations, providing sufficient diversity while maintaining interpretability.

The synthetic environment allows us to precisely measure color vs. shape categorization bias by computing similarity between stimuli that share color but differ in shape, versus stimuli that share shape but differ in color.

### 3.2 Neural Architecture

Our architecture consists of three brain-inspired modules:

**Visual Cortex.** A convolutional encoder-decoder network inspired by hierarchical visual processing. The encoder maps 28×28×3 images to 64-dimensional feature vectors through three convolutional layers with ReLU activations and max-pooling. The decoder reconstructs the input image, implementing predictive coding where the network learns by minimizing reconstruction error.

**Language Cortex.** A character-level recurrent network that processes linguistic labels. Text is embedded character-by-character, processed through an LSTM, and projected to a 64-dimensional feature space matching the visual representation dimensionality.

**Anterior Temporal Lobe (ATL).** A semantic hub implementing competitive learning with shared prototypes. The ATL maintains K=100 prototype vectors that represent learned concepts. For each input (visual or linguistic), the ATL computes activation as the softmax over cosine similarities to all prototypes:

$$a_k = \frac{\exp(\text{sim}(x, p_k) / \tau)}{\sum_{j=1}^{K} \exp(\text{sim}(x, p_j) / \tau)}$$

where τ=0.1 is a temperature parameter. The winning prototype is updated via Hebbian learning:

$$p_k \leftarrow p_k + \eta \cdot a_k \cdot (x - p_k)$$

Crucially, visual and linguistic inputs share the same prototype space but use modality-specific projection layers, enabling cross-modal alignment.

### 3.3 Learning Rules

Training proceeds in three developmental phases with intentional overlap, reflecting biological development where multiple learning processes occur concurrently:

**Phase 1: Visual Feature Learning (Epochs 1-10).** The visual encoder-decoder is trained using reconstruction loss:

$$\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|_2^2$$

This implements predictive coding (Rao & Ballard, 1999), where the visual cortex learns to compress and reconstruct visual input, extracting meaningful features without labels.

**Phase 2: Cross-Modal Alignment (Epochs 5-25).** Beginning while visual learning is still active, the language cortex learns to produce representations similar to visual features:

$$\mathcal{L}_{\text{align}} = 1 - \frac{f_v \cdot f_l}{\|f_v\| \|f_l\|}$$

where f_v and f_l are visual and linguistic feature vectors. Note: This phase uses gradient descent for efficiency; a fully Hebbian alternative would use cross-correlation learning.

**Phase 3: Semantic Consolidation (Epochs 10+).** Overlapping with continued visual and language learning, image-label pairs are presented to the ATL for competitive learning. This Hebbian consolidation strengthens cross-modal associations.

**Rationale for Overlap:** In biological development, visual, linguistic, and semantic learning occur simultaneously rather than sequentially (Kuhl, 2004). The overlapping phases allow learned visual features to immediately influence language alignment, and both to inform semantic consolidation.

### 3.4 Evaluation Metrics

**Color Bias Score.** We measure the ratio of average cosine similarity between stimuli sharing color (but differing in shape) to stimuli sharing shape (but differing in color):

$$\text{Bias} = \frac{\mathbb{E}[\text{sim}(x_{\text{same-color}}, x'_{\text{diff-shape}})]}{\mathbb{E}[\text{sim}(x_{\text{same-shape}}, x'_{\text{diff-color}})]}$$

A score >1 indicates color bias; <1 indicates shape bias; =1 is neutral.

**Cross-Modal Binding Rate.** The percentage of image-label pairs where both modalities activate the same ATL prototype.

**Active Concept Count.** The number of distinct prototypes activated across all stimuli, measuring representational diversity.

### 3.5 Baselines and Ablations

We compare our brain-like model against:
- **Backpropagation baseline:** A convolutional classifier trained with cross-entropy loss for color and shape classification.
- **No reconstruction:** Brain-like model without visual cortex training (random features).
- **No consolidation:** Brain-like model without ATL Hebbian updates.

### 3.6 Statistical Analysis

All experiments are repeated with n=10 random seeds for short training and n=3 seeds for extended training. We report mean ± standard deviation and perform two-tailed independent t-tests for between-condition comparisons and paired t-tests for within-model ablations. Effect sizes are reported as Cohen's d.

**Multiple Comparisons.** We conduct two primary, *a priori* planned comparisons: (1) brain-inspired vs. backprop bias, and (2) consolidation effect on binding. Since these tests address distinct, pre-specified hypotheses rather than exploratory analysis, we report uncorrected p-values (Armstrong, 2014). For completeness, both comparisons remain significant under Bonferroni correction (α = 0.025): p < 0.00001 and p = 0.01 < 0.025.

---

## 4. Results

### 4.1 Brain-Inspired Learning Achieves Cross-Modal Binding

Our brain-like model successfully learns to bind visual shapes with linguistic labels, achieving **62.5% ± 8.4%** cross-modal alignment (Table 1). The model discovers 2.7 active concepts on average, suggesting it learns to categorize the 108 unique stimuli into a small number of semantic clusters.

**Table 1: Main results across conditions (n=10 seeds)**

| Condition | Color Bias | Binding Rate | Concepts |
|-----------|------------|--------------|----------|
| Full Model (Brain-like) | 1.227 ± 0.090 | 62.5% ± 8.4% | 2.7 ± 0.8 |
| No Reconstruction | 1.020 ± 0.010 | 100.0% ± 0.0%* | 1.0 ± 0.0 |
| No Consolidation | 1.177 ± 0.151 | 44.4% ± 15.7% | 6.1 ± 1.2 |
| Backprop Baseline | 2.245 ± 0.475 | --- | --- |

*Collapsed to single concept (trivial binding).

### 4.2 Brain-Inspired Learning Shows Reduced Color Bias

A striking finding is that brain-inspired learning exhibits significantly **less color bias** than supervised backpropagation (Figure 1A). The brain-inspired model shows moderate color bias (1.23 ± 0.09), while backpropagation shows nearly double the bias (2.25 ± 0.48). This difference is highly significant (t=-6.32, **p<0.00001**).

![Figure 1: Ablation Results](figure2_ablation_results.png)
*Figure 1: Ablation study results. (A) Color categorization bias across conditions. Brain-inspired learning shows significantly lower bias than backpropagation (p<0.00001). Dashed line indicates neutral (no bias). (B) Cross-modal binding rates. ATL consolidation provides 41% improvement (p=0.01, d=1.44). Error bars show ±1 SD (n=10).*

Furthermore, brain-inspired learning shows **lower variance** across seeds (std=0.09) compared to backpropagation (std=0.48), suggesting more robust learning dynamics. This implicit regularization effect was unexpected and suggests that reconstruction-based learning with Hebbian consolidation may prevent overfitting to superficial features.

### 4.3 Ablation Studies Reveal Mechanistic Requirements

**Reconstruction is necessary.** Without visual cortex training via reconstruction loss, features remain random and the model collapses to a single concept. The 100% binding rate in this condition is trivial—everything maps to the same prototype.

**ATL consolidation is critical.** Removing Hebbian consolidation in ATL reduces binding rate by 41% (62.5% → 44.4%; t=3.27, **p=0.01**). Effect size is large (Cohen's d=1.44), indicating consolidation is essential for cross-modal alignment.

**No consolidation increases concept count.** Interestingly, removing consolidation *increases* the number of active concepts (2.7 → 6.1) while *decreasing* binding quality. This suggests consolidation performs a compression function, distilling many fragmented representations into fewer, more coherent concepts.

### 4.4 Emergent Developmental Trajectory

Extended training (100 epochs, n=3 seeds) reveals a striking developmental trajectory (Figure 2). The model progresses through three distinct phases:

**Phase 1: Neutral (Epochs 0-5).** Initial bias is near-neutral (0.99 ± 0.00), reflecting random initialization before meaningful features emerge.

**Phase 2: Color Bias Emergence (Epochs 5-25).** Bias rapidly increases to a peak of 1.32 ± 0.06, indicating the visual cortex first learns color-based representations.

**Phase 3: Bias Reduction (Epochs 25-100).** Bias gradually decreases toward neutral (1.07 ± 0.02), suggesting shape features emerge with continued training.

![Figure 2: Developmental Trajectory](figure3_developmental_trajectory.png)
*Figure 2: Developmental trajectory over extended training (100 epochs). (A) Color bias emerges early (peak at epoch 25) then reduces toward neutral, recapitulating infant visual development. (B) Cross-modal binding rate stabilizes around 55%. (C) Active concept count increases as training progresses. Thin lines: individual seeds (n=3); thick line: mean; shading: ±1 SD.*

This trajectory was not explicitly programmed—it emerges naturally from the interaction between reconstruction-based learning and cross-modal binding. The temporal dynamics closely parallel infant visual development, where color categorization dominates before 24 months, followed by shape bias emergence (Smith & Heise, 1992).

---

## 5. Discussion

### 5.1 Biological Relevance

Our results provide computational support for several neuroscientific theories:

**Predictive coding.** The necessity of reconstruction loss validates predictive coding as a viable cortical learning mechanism (Rao & Ballard, 1999). Without it, visual features remain random and uninformative.

**ATL as semantic hub.** The critical role of ATL consolidation for cross-modal binding supports the hub-and-spoke model of semantic memory (Patterson et al., 2007). Our simplified ATL, implementing competitive learning with shared prototypes, captures the essential function of integrating multimodal information into unified concepts.

**Developmental trajectory.** The emergent color→shape progression matches extensive developmental psychology literature (Bornstein, 1976; Smith & Heise, 1992; Landau et al., 1988). Our model suggests this trajectory arises naturally from the learning dynamics of reconstruction-based feature extraction, where color—affecting more pixels than shape boundaries—produces stronger gradients early in training.

### 5.2 Implicit Regularization in Brain-Inspired Learning

Perhaps our most surprising finding is that brain-inspired learning produces *less* biased representations than supervised backpropagation. While backpropagation optimizes directly for classification, our approach provides implicit regularization through:

1. **Reconstruction objective** forces learning of generative features rather than discriminative shortcuts.
2. **Competitive learning** in ATL encourages balanced prototype utilization.
3. **Hebbian consolidation** averages representations across examples, smoothing noise.

This suggests that biological constraints, often viewed as limitations, may actually confer advantages for robust representation learning. Similar observations have been made in the machine learning literature regarding the regularizing effects of noise and local learning rules (Hochreiter & Schmidhuber, 1997).

### 5.3 Mechanistic Insights

Why does color dominate early learning? We propose a simple explanation based on gradient magnitude:

$$\text{MSE}(\text{red circle}, \text{blue circle}) \gg \text{MSE}(\text{red circle}, \text{red square})$$

Changing color affects nearly all pixels, while changing shape affects only boundary pixels. Thus, reconstruction loss produces larger gradients for color differences, leading to faster learning of color features. Shape features require more training to emerge because their gradients are smaller.

### 5.4 Limitations

Several limitations warrant mention:

- **Simplified environment.** Our synthetic stimuli lack the complexity of natural images. The shapes are uniform-colored with no texture, shading, or occlusion.
- **Limited vocabulary.** With only 108 unique stimuli and simple two-word labels, we cannot assess compositional generalization or complex language understanding.
- **Architectural simplifications.** Our model omits many brain structures (hippocampus, prefrontal cortex, basal ganglia) that likely contribute to human cross-modal learning.
- **Learning rule approximations.** Language alignment uses gradient descent rather than purely Hebbian learning, limiting claims about full biological plausibility.
- **Sample size for trajectory analysis.** Extended training used n=3 seeds due to computational constraints; larger samples would strengthen trajectory claims.
- **No human behavioral comparison.** While we draw parallels to infant development, we did not directly compare to human experimental data.
- **Single modality pair.** We only tested vision-language binding; auditory or tactile modalities may show different dynamics.

### 5.5 Future Directions

1. **Compositional learning:** Can the model learn "red" and "circle" as separable concepts?
2. **Natural images:** Does the developmental trajectory persist with realistic visual input?
3. **Longer training:** Does shape bias eventually dominate, matching adult perception?
4. **Hippocampal contributions:** How does episodic memory interact with semantic consolidation?

---

## 6. Conclusion

We have demonstrated that brain-inspired learning rules—reconstruction-based predictive coding and Hebbian consolidation—can achieve meaningful cross-modal binding from scratch. Statistical validation (n=10 seeds) reveals that our approach shows significantly reduced color bias compared to supervised backpropagation (1.23 vs 2.25, p<0.00001), suggesting reconstruction-based learning with Hebbian consolidation provides implicit regularization. Extended training reveals an emergent developmental trajectory that recapitulates infant visual development, transitioning from neutral through peak color bias to reduction.

Our ablation studies identify critical components: reconstruction prevents representational collapse, while ATL consolidation improves cross-modal binding by 41% (p=0.01, d=1.44). These findings provide mechanistic insights into both biological learning and the development of perceptual biases.

Brain-inspired learning achieves not only functional performance but also *recapitulates known developmental phenomena*—a hallmark of biologically-relevant computational models. This work takes a step toward understanding how the brain learns to integrate information across modalities using local plasticity rules and reconstruction-based objectives.

---

## References

1. Armstrong, R. A. (2014). When to use the Bonferroni correction. *Ophthalmic and Physiological Optics*, 34(5), 502-508.

2. Bornstein, M. H. (1976). Infants are trichromats. *Journal of Experimental Child Psychology*, 21(3), 425-445.

3. Colunga, E., & Smith, L. B. (2005). From the lexicon to expectations about kinds: A role for associative learning. *Psychological Review*, 112(2), 347-382.

4. Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815-836.

5. Grossberg, S. (1987). Competitive learning: From interactive activation to adaptive resonance. *Cognitive Science*, 11(1), 23-63.

6. Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.

7. Hochreiter, S., & Schmidhuber, J. (1997). Flat minima. *Neural Computation*, 9(1), 1-42.

8. Jia, C., Yang, Y., Xia, Y., et al. (2021). Scaling up visual and vision-language representation learning with noisy text supervision. *ICML*, 4904-4916.

9. Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1), 59-69.

10. Kuhl, P. K. (2004). Early language acquisition: Cracking the speech code. *Nature Reviews Neuroscience*, 5(11), 831-843.

11. Lambon Ralph, M. A., Jefferies, E., Patterson, K., & Rogers, T. T. (2017). The neural and computational bases of semantic cognition. *Nature Reviews Neuroscience*, 18(1), 42-55.

12. Landau, B., Smith, L. B., & Jones, S. S. (1988). The importance of shape in early lexical learning. *Cognitive Development*, 3(3), 299-321.

13. Patterson, K., Nestor, P. J., & Rogers, T. T. (2007). Where do you know what you know? *Nature Reviews Neuroscience*, 8(12), 976-987.

14. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*, 8748-8763.

15. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79-87.

16. Rogers, T. T., & McClelland, J. L. (2004). *Semantic Cognition: A Parallel Distributed Processing Approach*. MIT Press.

17. Singh, A., Hu, R., Goswami, V., et al. (2022). FLAVA: A foundational language and vision alignment model. *CVPR*, 15638-15650.

18. Sloutsky, V. M. (2010). From perceptual categories to concepts: What develops? *Cognitive Science*, 34(7), 1244-1286.

19. Smith, L. B., & Heise, D. (1992). Perceptual similarity and conceptual structure. *Advances in Psychology*, 93, 233-272.

20. Visser, M., Jefferies, E., & Lambon Ralph, M. A. (2010). Semantic processing in the anterior temporal lobes: A meta-analysis of the functional neuroimaging literature. *Journal of Cognitive Neuroscience*, 22(6), 1083-1094.

21. Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*, 29(5), 1229-1262.

---

## Supplementary Material

### A. Detailed Ablation Statistics

| Comparison | t-statistic | p-value | Cohen's d | Significant |
|------------|-------------|---------|-----------|-------------|
| Brain-like vs Backprop (bias) | -6.316 | <0.00001 | 2.98 | Yes |
| Full vs No-Consolidation (binding) | 3.267 | 0.010 | 1.44 | Yes |

### B. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Visual feature dimension | 64 |
| Number of ATL prototypes | 100 |
| ATL temperature τ | 0.1 |
| Hebbian learning rate η | 0.1 |
| Visual reconstruction epochs | 10 |
| Language alignment epochs | 15 |
| Cross-modal binding epochs | 10 |
| Adam learning rate (cortices) | 0.001 |

### C. Code Availability

All code, data, and trained models are available at: https://github.com/Diimoo/CHPL
