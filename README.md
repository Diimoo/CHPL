# CHPL: Compositional Hebbian Predictive Learning

Official code for **Distributed Semantic Binding: From Synthetic Composition to Natural Scenes**

> A unified architecture that develops from infant-level perception to adult-level cognition in ~20 minutes.

---

## 🎯 Key Results

| Capability | Target | Achieved | Status |
|------------|--------|----------|--------|
| **Grounded Vocabulary** | 50,000 | 275,527 (94.9% coverage) | ✅ 551% |
| **Knowledge Patterns** | 3,000 | 3,665 (hierarchical) | ✅ 122% |
| **Observation Events** | 4,000 | 128,788 | ✅ 3,220% |
| **Grammar Checking** | Working | Complete | ✅ |

**Total training time: ~20 minutes on consumer GPU (RTX 4080)**

---

## 📁 Repository Structure

```
CHPL/
├── 📄 Core Architecture
│   ├── brain_crossmodal_learner.py    # Main CHPL model
│   ├── synthetic_environment.py        # Visual stimulus generator
│   └── synthetic_environment_hierarchical.py
│
├── 🧪 experiments/                     # All experimental scripts
│   ├── Phase 1-2: Perception & Prediction
│   │   ├── predictive_atl.py          # Temporal prediction
│   │   ├── train_predictive_atl.py    # Train prediction
│   │   └── test_predictive_hard.py    # Evaluate prediction
│   │
│   ├── Phase 3: Causal Reasoning
│   │   ├── causal_atl.py              # Causal inference module
│   │   ├── train_causal_atl.py        # Train causal reasoning
│   │   └── synthetic_causal.py        # Causal environment
│   │
│   ├── Phase 4: Language & Composition
│   │   ├── language_atl.py            # Language-vision binding
│   │   ├── train_language_atl.py      # Train language
│   │   ├── hierarchical_atl.py        # Hierarchical composition
│   │   └── train_hierarchical_atl.py  # Train hierarchy
│   │
│   ├── Phase 5-8: Adult-Level Scaling
│   │   ├── distributional_language.py # Wikipedia training
│   │   ├── language_bootstrap.py      # Vocabulary expansion
│   │   ├── ground_vocabulary.py       # COCO visual grounding
│   │   ├── ground_vocabulary_multipass.py  # Semantic propagation
│   │   ├── knowledge_graph.py         # Knowledge acquisition
│   │   ├── video_learner.py           # Video pattern learning
│   │   ├── batch_video_processor.py   # Batch video processing
│   │   ├── dialogue_system.py         # Grammar-checked dialogue
│   │   └── continuous_observer.py     # Real-time observation
│   │
│   └── Evaluation & Analysis
│       ├── fair_baseline_comparison.py
│       ├── mechanistic_analysis.py
│       └── test_coco_subset.py
│
├── 📊 results/                         # Experimental results
│   ├── ADULT_LEVEL_RESULTS.md         # Final adult-level metrics
│   ├── ROADMAP_TO_CHILD.md            # Development phases 1-4
│   ├── CHILD_TO_ADULT_RESULTS.md      # Scaling results
│   └── FINAL_RESULTS.md               # Summary
│
├── 📝 paper/                           # Publication materials
│   ├── distributed_atl_paper.tex      # Main LaTeX paper
│   ├── distributed_atl_paper.pdf      # Compiled PDF
│   └── figures/                        # All figures
│
└── 🗃️ models/                          # Trained models & data
    └── knowledge_graph_full.pkl        # 3,665 knowledge patterns
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Diimoo/CHPL.git
cd CHPL
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA recommended

### Run the Full Development Pipeline

```bash
# Phase 1-2: Perception & Prediction (2 min)
python experiments/train_predictive_atl.py

# Phase 3: Causal Reasoning (2 min)
python experiments/train_causal_atl.py

# Phase 4: Language & Composition (4 min)
python experiments/train_language_atl.py
python experiments/train_hierarchical_atl.py

# Phase 5-8: Adult Scaling (11 min)
python experiments/distributional_language.py  # Wikipedia vocabulary
python experiments/ground_vocabulary.py        # COCO grounding
python experiments/ground_vocabulary_multipass.py  # Semantic propagation
```

### Quick Demo

```python
from brain_crossmodal_learner import BrainCrossModalLearner
from synthetic_environment import create_stimulus
import torch

# Create model
model = BrainCrossModalLearner(feature_dim=64, n_concepts=100)

# Process visual input
img = create_stimulus(shape='circle', color='red', size='medium')
img_t = torch.tensor(img, dtype=torch.float32).cuda()

with torch.no_grad():
    features = model.visual(img_t)
    _, concept = model.atl.activate(features, 'visual')
    
print(f"Activated concept: {concept}")
```

---

## 📖 Development Phases

### Phase 1-2: Perception & Prediction
- **Goal:** Learn to predict next visual state
- **Metric:** Prediction accuracy 0.946
- **Time:** ~2 minutes

### Phase 3: Causal Reasoning  
- **Goal:** Infer cause-effect from observation
- **Metric:** Causal accuracy 1.000
- **Time:** ~2 minutes

### Phase 4: Language & Composition
- **Goal:** Bind language to vision, compose hierarchically
- **Metrics:** Visual QA 0.860, Analogical reasoning 1.000
- **Time:** ~4 minutes

### Phase 5-8: Adult-Level Scaling
- **Goal:** Scale to real-world vocabulary and knowledge
- **Metrics:** 275,527 grounded words, 3,665 knowledge patterns
- **Time:** ~11 minutes

See [`results/ROADMAP_TO_CHILD.md`](results/ROADMAP_TO_CHILD.md) for detailed phase documentation.

---

## 🔬 Key Experiments

| Experiment | Script | Description |
|------------|--------|-------------|
| **Compositional Generalization** | `experiments/train_two_object_distributed.py` | Multi-object scene understanding |
| **COCO Natural Images** | `experiments/test_coco_subset.py` | Real image evaluation |
| **Vocabulary Grounding** | `experiments/ground_vocabulary_multipass.py` | 275k word grounding |
| **Video Knowledge** | `experiments/batch_video_processor.py` | Extract patterns from video |
| **Dialogue System** | `experiments/dialogue_system.py` | Grammar-checked conversation |
| **Continuous Observation** | `experiments/continuous_observer.py` | Real-time event detection |

---

## 📈 Reproducing Paper Results

### Main Validation (Table 1)
```bash
python validation_study.py  # 10 seeds, 4 conditions
```

### Compositional Generalization (Table 2)
```bash
python experiments/train_two_object_distributed.py
python experiments/test_hierarchical_composition.py
```

### Adult-Level Scaling (Table 3)
```bash
# Full pipeline - see results/ADULT_LEVEL_RESULTS.md
python experiments/distributional_language.py
python experiments/ground_vocabulary_multipass.py
python experiments/batch_video_processor.py
```

### Generate Figures
```bash
python generate_figures.py
python experiments/make_figures.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`results/ROADMAP_TO_CHILD.md`](results/ROADMAP_TO_CHILD.md) | Development phases 1-4 roadmap |
| [`results/ADULT_LEVEL_RESULTS.md`](results/ADULT_LEVEL_RESULTS.md) | Final adult-level metrics |
| [`results/CHILD_TO_ADULT_RESULTS.md`](results/CHILD_TO_ADULT_RESULTS.md) | Scaling experiments |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Key scientific findings |
| [`paper/distributed_atl_paper.tex`](paper/distributed_atl_paper.tex) | Full paper (LaTeX) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CHPL Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Visual Cortex          Language Cortex                     │
│  ┌──────────┐           ┌──────────┐                        │
│  │ CNN/ViT  │           │ Word2Vec │                        │
│  │ Encoder  │           │ Embedder │                        │
│  └────┬─────┘           └────┬─────┘                        │
│       │                      │                              │
│       └──────────┬───────────┘                              │
│                  ▼                                          │
│         ┌───────────────┐                                   │
│         │ Distributed   │  Soft activation over             │
│         │     ATL       │  200 prototypes (τ=0.2)           │
│         │  (Semantic    │  Hebbian learning                 │
│         │    Hub)       │                                   │
│         └───────┬───────┘                                   │
│                 │                                           │
│    ┌────────────┼────────────┐                              │
│    ▼            ▼            ▼                              │
│ Prediction  Causality   Knowledge                           │
│  Module      Module       Graph                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Citation

```bibtex
@article{trabelsi2026distributed,
  title={Distributed Semantic Binding: From Synthetic Composition to Natural Scenes},
  author={Trabelsi, Ahmed},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on principles from computational neuroscience (anterior temporal lobe semantic hub) and developmental psychology (infant cognitive development trajectories).
