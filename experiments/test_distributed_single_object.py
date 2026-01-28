#!/usr/bin/env python3
"""
Sanity check: Distributed ATL on single-object task.
Should achieve similar binding as winner-takes-all.
"""

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

import numpy as np
import torch
import torch.nn.functional as F

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import generate_training_pairs


def train_and_evaluate(brain, pairs, n_epochs_visual=10, n_epochs_lang=15, n_epochs_binding=10):
    """Train brain and return final binding metric."""
    data = [(p.image, p.label) for p in pairs]

    # Phase 1: Visual reconstruction
    vis_opt = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(n_epochs_visual):
        np.random.shuffle(data)
        losses = []
        for img, _ in data:
            vis_opt.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(img).float())
            loss.backward()
            vis_opt.step()
            losses.append(loss.item())
        print(f"  Visual epoch {epoch+1}: loss={np.mean(losses):.4f}")

    # Phase 2: Language alignment
    lang_opt = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(n_epochs_lang):
        np.random.shuffle(data)
        losses = []
        for img, label in data:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(img).float())
            lang_feat = brain.language(label)
            loss = 1 - F.cosine_similarity(vis_feat.unsqueeze(0), lang_feat.unsqueeze(0))
            lang_opt.zero_grad()
            loss.backward()
            lang_opt.step()
            losses.append(loss.item())
        print(f"  Language epoch {epoch+1}: loss={np.mean(losses):.4f}")

    # Phase 3: Binding
    binding_metrics = []
    for epoch in range(n_epochs_binding):
        np.random.shuffle(data)

        if brain.use_distributed_atl:
            similarities = []
            for img, label in data:
                img_t = torch.from_numpy(img).float()
                vis_feat = brain.visual(img_t)
                lang_feat = brain.language(label)
                brain.atl.consolidate(vis_feat.detach(), lang_feat.detach())
                sim = brain.atl.similarity(vis_feat, lang_feat)
                similarities.append(sim)
            metric = np.mean(similarities)
            print(f"  Binding epoch {epoch+1}: pattern_sim={metric:.3f}, concepts={brain.atl.get_active_concepts()}")
        else:
            correct = 0
            for img, label in data:
                img_t = torch.from_numpy(img).float()
                res = brain.experience(img_t, label, consolidate=True, train_cortices=False)
                if res['same_concept']:
                    correct += 1
            metric = correct / len(data)
            print(f"  Binding epoch {epoch+1}: winner_match={metric:.3f}, concepts={brain.atl.get_active_concepts()}")

        binding_metrics.append(metric)

    return binding_metrics[-1]


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("Generating single-object data...")
    pairs = generate_training_pairs(n_per_combination=5)
    print(f"Generated {len(pairs)} training examples")
    print(f"Device: {DEVICE}")

    # Train distributed ATL
    print("\n=== DISTRIBUTED ATL ===")
    brain_dist = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=50,
        use_distributed_atl=True,
        distributed_temperature=0.1,
    )
    dist_metric = train_and_evaluate(brain_dist, pairs)

    # Train baseline ATL
    print("\n=== BASELINE ATL ===")
    np.random.seed(42)
    torch.manual_seed(42)
    brain_base = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=50,
        use_distributed_atl=False,
    )
    base_metric = train_and_evaluate(brain_base, pairs)

    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Distributed ATL final: {dist_metric:.3f}")
    print(f"Baseline ATL final:    {base_metric:.3f}")

    if dist_metric > 0.7:
        print("\n✓ Distributed ATL works on simple task!")
    elif dist_metric > 0.5:
        print("\n? Distributed ATL partially works - may need tuning")
    else:
        print("\n✗ Distributed ATL fails on simple task - need to debug")


if __name__ == '__main__':
    main()
