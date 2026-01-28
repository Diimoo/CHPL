#!/usr/bin/env python3
"""
Train distributed ATL on two-object compositional scenes.
Compare pattern similarity to winner-takes-all baseline.
"""

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import generate_two_object_pairs


def _parse_label(label: str):
    parts = label.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Unexpected label format: {label}")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--n_concepts', type=int, default=200)
    ap.add_argument('--n_epochs_visual', type=int, default=10)
    ap.add_argument('--n_epochs_lang', type=int, default=15)
    ap.add_argument('--n_epochs_binding', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square']
    relations = ['above', 'left_of']
    n_per_combination = 20
    held_out_color = 'green'
    canvas_size = 56

    out_dir = Path('two_object_results')
    out_dir.mkdir(exist_ok=True)
    run_id = f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_json = out_dir / f'distributed_atl_{run_id}.json'
    out_ckpt = out_dir / f'distributed_atl_{run_id}.pt'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {DEVICE}")

    brain = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=args.n_concepts,
        visual_input_size=canvas_size,
        use_distributed_atl=True,
        distributed_temperature=args.temperature,
    )

    pairs = generate_two_object_pairs(
        colors=colors,
        shapes=shapes,
        relations=relations,
        n_per_combination=n_per_combination,
        canvas_size=canvas_size,
    )

    data = [(p.image, p.label) for p in pairs]

    combos = sorted({_parse_label(lbl) for (_, lbl) in data})
    train_combos = {c for c in combos if c[0] != held_out_color}
    test_combos = {c for c in combos if c[0] == held_out_color}

    train_data = [(img, lbl) for (img, lbl) in data if _parse_label(lbl) in train_combos]
    test_data = [(img, lbl) for (img, lbl) in data if _parse_label(lbl) in test_combos]

    print(f"Train: {len(train_data)} (combos={len(train_combos)})")
    print(f"Test:  {len(test_data)} (combos={len(test_combos)})")

    # Phase 1: Visual reconstruction
    print("\nPHASE 1: Visual reconstruction (56x56)")
    vis_opt = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(args.n_epochs_visual):
        np.random.shuffle(train_data)
        losses = []
        for img, _ in train_data:
            vis_opt.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(img).float())
            loss.backward()
            vis_opt.step()
            losses.append(loss.item())
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")

    # Phase 2: Language alignment
    print("\nPHASE 2: Language alignment")
    lang_opt = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(args.n_epochs_lang):
        np.random.shuffle(train_data)
        losses = []
        for img, label in train_data:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(img).float())
            lang_feat = brain.language(label)
            loss = 1 - F.cosine_similarity(vis_feat.unsqueeze(0), lang_feat.unsqueeze(0))
            lang_opt.zero_grad()
            loss.backward()
            lang_opt.step()
            losses.append(loss.item())
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")

    # Phase 3: Distributed binding
    print("\nPHASE 3: Distributed semantic binding")
    binding_qualities = []
    kl_divergences = []

    for epoch in range(args.n_epochs_binding):
        np.random.shuffle(train_data)
        epoch_quality = []
        epoch_kl = []

        for img, label in train_data:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(label)
            result = brain.atl.consolidate(vis_feat.detach(), lang_feat.detach())
            epoch_quality.append(result['pattern_similarity'])
            epoch_kl.append(result['kl_divergence'])

        binding_qualities.append(np.mean(epoch_quality))
        kl_divergences.append(np.mean(epoch_kl))
        stats = brain.atl.get_stats()
        print(f"  Epoch {epoch+1}: pattern_sim={binding_qualities[-1]:.3f}, kl={kl_divergences[-1]:.4f}, concepts={stats['active_concepts']}")

    # Evaluate
    print("\n=== EVALUATION ===")

    def evaluate_pattern_similarity(data_subset):
        similarities = []
        for img, label in data_subset:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(label)
            sim = brain.atl.similarity(vis_feat, lang_feat)
            similarities.append(sim)
        return float(np.mean(similarities))

    train_sim = evaluate_pattern_similarity(train_data)
    test_sim = evaluate_pattern_similarity(test_data)

    print(f"Train pattern similarity:    {train_sim:.3f}")
    print(f"Held-out pattern similarity: {test_sim:.3f}")
    print(f"Gap: {train_sim - test_sim:.3f}")

    # Compare to A2 baseline
    print("\n=== vs A2 BASELINE (Winner-Takes-All) ===")
    print("A2 Baseline:")
    print("  Train cosine: 0.862, Held-out: 0.512")
    print("  Train winner: 1.000, Held-out: 0.568")
    print(f"\nDistributed ATL:")
    print(f"  Train pattern_sim: {train_sim:.3f}")
    print(f"  Held-out pattern_sim: {test_sim:.3f}")

    if test_sim > 0.7:
        print("\n✓✓ SUCCESS: Distributed ATL solves composition!")
    elif test_sim > 0.6:
        print("\n✓ PARTIAL: Better than winner-takes-all baseline")
    elif test_sim > 0.5:
        print("\n? MARGINAL: Similar to baseline, may need tuning")
    else:
        print("\n✗ WORSE: Distributed doesn't help here")

    # Save results
    results = {
        'run_id': run_id,
        'config': {
            'atl_type': 'distributed',
            'seed': args.seed,
            'temperature': args.temperature,
            'n_concepts': args.n_concepts,
            'visual_input_size': canvas_size,
            'held_out_color': held_out_color,
        },
        'training': {
            'binding_qualities': binding_qualities,
            'kl_divergences': kl_divergences,
        },
        'evaluation': {
            'train_pattern_similarity': train_sim,
            'held_out_pattern_similarity': test_sim,
            'gap': train_sim - test_sim,
        },
        'stats': brain.atl.get_stats(),
    }

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(
        {
            'config': results['config'],
            'visual_state': brain.visual.state_dict(),
            'language_state': brain.language.state_dict(),
            'atl_state': brain.atl.state_dict(),
        },
        out_ckpt,
    )

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_ckpt}")


if __name__ == '__main__':
    main()
