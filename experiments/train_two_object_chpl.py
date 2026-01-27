#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment import generate_two_object_pairs, downsample_scene


def _parse_label(label: str):
    parts = label.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Unexpected label format: {label}")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def main():
    # Small-start config (A1)
    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square']
    relations = ['above', 'left_of']

    n_per_combination = 20
    n_concepts = 200

    n_epochs_visual = 10
    n_epochs_lang = 15
    n_epochs_binding = 10

    # Compositional split: hold out all combos where obj1 color == held_out_color
    held_out_color = 'green'

    out_dir = Path('two_object_results')
    out_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = out_dir / f'two_object_a1_{run_id}.json'
    out_ckpt = out_dir / f'two_object_a1_{run_id}.pt'

    np.random.seed(0)
    torch.manual_seed(0)

    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=n_concepts)

    pairs56 = generate_two_object_pairs(
        colors=colors,
        shapes=shapes,
        relations=relations,
        n_per_combination=n_per_combination,
        canvas_size=56,
    )

    # Convert to (28x28 image, label) tuples + build compositional split
    data = []
    for p in pairs56:
        img28 = downsample_scene(p.image, out_size=28)
        data.append((img28, p.label))

    combos = sorted({
        _parse_label(lbl) for (_img, lbl) in data
    })
    train_combos = {c for c in combos if c[0] != held_out_color}
    test_combos = {c for c in combos if c[0] == held_out_color}

    train_data = [(img, lbl) for (img, lbl) in data if _parse_label(lbl) in train_combos]
    test_data = [(img, lbl) for (img, lbl) in data if _parse_label(lbl) in test_combos]

    print(f"Generated {len(data)} total examples")
    print(f"Train examples: {len(train_data)} (combos={len(train_combos)})")
    print(f"Test examples:  {len(test_data)} (combos={len(test_combos)})")
    print(f"Device: {DEVICE}")

    # ============================================================
    # Phase 1: Visual cortex reconstruction
    # ============================================================
    print("\nPHASE 1: Visual reconstruction")
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)

    for epoch in range(n_epochs_visual):
        np.random.shuffle(train_data)
        total_loss = 0.0
        n = 0
        for img28, _label in train_data:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(img28).float())
            loss.backward()
            vis_optimizer.step()
            total_loss += float(loss.item())
            n += 1
        print(f"  Epoch {epoch+1}: recon_loss={total_loss/max(1,n):.4f}")

    # ============================================================
    # Phase 2: Language alignment (teach language to match vision)
    # ============================================================
    print("\nPHASE 2: Language alignment")
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)

    for epoch in range(n_epochs_lang):
        np.random.shuffle(train_data)
        total_loss = 0.0
        n = 0
        for img28, label in train_data:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(img28).float())
            lang_feat = brain.language(label)
            loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
            lang_optimizer.zero_grad()
            loss.backward()
            lang_optimizer.step()
            total_loss += float(loss.item())
            n += 1
        print(f"  Epoch {epoch+1}: alignment_loss={total_loss/max(1,n):.4f}")

    # ============================================================
    # Phase 3: Binding / consolidation (ATL)
    # ============================================================
    print("\nPHASE 3: Cross-modal binding")

    binding_rates = []
    for epoch in range(n_epochs_binding):
        np.random.shuffle(train_data)
        n_same = 0
        for img28, label in train_data:
            result = brain.experience(
                torch.from_numpy(img28).float(),
                label,
                consolidate=True,
                train_cortices=False,
            )
            if result['same_concept']:
                n_same += 1
        binding_rate = n_same / max(1, len(train_data))
        binding_rates.append(binding_rate)
        stats = brain.get_stats()
        print(f"  Epoch {epoch+1}: binding_rate={binding_rate:.3f}, concepts={stats['atl_active_concepts']}")

    # Quick held-out eval (compositional generalization proxy)
    n_correct = 0
    for img28, label in test_data:
        res = brain.test_binding(torch.from_numpy(img28).float(), label)
        if res['same_concept']:
            n_correct += 1
    held_out_acc = n_correct / max(1, len(test_data))
    print(f"\nHeld-out compositional accuracy (obj1 color={held_out_color}): {held_out_acc:.3f}")

    payload = {
        'run_id': run_id,
        'config': {
            'colors': colors,
            'shapes': shapes,
            'relations': relations,
            'n_per_combination': n_per_combination,
            'n_concepts': n_concepts,
            'n_epochs_visual': n_epochs_visual,
            'n_epochs_lang': n_epochs_lang,
            'n_epochs_binding': n_epochs_binding,
            'a1_downsample_to': 28,
            'split_rule': 'hold_out_obj1_color',
            'held_out_color': held_out_color,
        },
        'binding_rates': binding_rates,
        'held_out_accuracy': held_out_acc,
        'stats': brain.get_stats(),
    }

    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    torch.save(
        {
            'config': payload['config'],
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
