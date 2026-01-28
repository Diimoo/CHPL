#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

# Force CPU for evaluation: BrainCrossModalLearner sets DEVICE at import time.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from brain_crossmodal_learner import BrainCrossModalLearner
from synthetic_environment import generate_two_object_pairs, downsample_scene


def _parse_label(label: str):
    # "red circle above blue square" -> (c1,s1,rel,c2,s2)
    parts = label.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Unexpected label format: {label}")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _eval_metrics(brain: BrainCrossModalLearner, data, top_k: int = 5):
    winner_match = []
    topk_overlap = []
    topk_hit = []
    cosine_feat = []
    cosine_proj = []

    for img28, label in data:
        img_t = torch.from_numpy(img28).float()
        vis_feat = brain.visual(img_t)
        lang_feat = brain.language(label)

        vis_feat_n = F.normalize(vis_feat.flatten(), dim=-1)
        lang_feat_n = F.normalize(lang_feat.flatten(), dim=-1)
        cosine_feat.append(float(torch.dot(vis_feat_n, lang_feat_n).item()))

        # Project both into ATL shared space (same space prototypes live in)
        vis_proj = F.normalize(brain.atl.vis_proj(vis_feat.flatten()), dim=-1)
        lang_proj = F.normalize(brain.atl.lang_proj(lang_feat.flatten()), dim=-1)
        cosine_proj.append(float(torch.dot(vis_proj, lang_proj).item()))

        sims_vis = torch.matmul(brain.atl.prototypes, vis_proj)
        sims_lang = torch.matmul(brain.atl.prototypes, lang_proj)

        vis_w = int(sims_vis.argmax().item())
        lang_w = int(sims_lang.argmax().item())
        winner_match.append(1.0 if vis_w == lang_w else 0.0)

        k = min(int(top_k), int(sims_vis.numel()))
        vis_top = torch.topk(sims_vis, k=k).indices.tolist()
        lang_top = torch.topk(sims_lang, k=k).indices.tolist()
        overlap = len(set(vis_top).intersection(set(lang_top)))
        topk_overlap.append(overlap / max(1, k))

        # weaker-but-informative: does either winner appear in the other modality's top-k?
        hit = (vis_w in lang_top) or (lang_w in vis_top)
        topk_hit.append(1.0 if hit else 0.0)

    return {
        'n': int(len(data)),
        'winner_match': float(np.mean(winner_match)) if winner_match else 0.0,
        'topk_overlap': float(np.mean(topk_overlap)) if topk_overlap else 0.0,
        'topk_hit': float(np.mean(topk_hit)) if topk_hit else 0.0,
        'cosine_feat': float(np.mean(cosine_feat)) if cosine_feat else 0.0,
        'cosine_proj': float(np.mean(cosine_proj)) if cosine_proj else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint saved by train_two_object_chpl.py')
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt.get('config', {})
    colors = list(config.get('colors', ['red', 'blue', 'green']))
    shapes = list(config.get('shapes', ['circle', 'square']))
    relations = list(config.get('relations', ['above', 'left_of']))
    held_out_color = str(config.get('held_out_color', 'green'))
    n_concepts = int(config.get('n_concepts', 200))
    n_per_combination = int(config.get('n_per_combination', 20))
    visual_input_size = int(config.get('visual_input_size', 28))
    atl_variant = str(config.get('atl_variant', 'baseline'))

    from brain_crossmodal_learner import ATLVariant

    brain = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=n_concepts,
        visual_input_size=visual_input_size,
        atl_variant=ATLVariant[atl_variant.upper()],
    )
    brain.visual.load_state_dict(ckpt['visual_state'])
    brain.language.load_state_dict(ckpt['language_state'])
    brain.atl.load_state_dict(ckpt['atl_state'])

    pairs56 = generate_two_object_pairs(
        colors=colors,
        shapes=shapes,
        relations=relations,
        n_per_combination=n_per_combination,
        canvas_size=56,
    )

    data = []
    for p in pairs56:
        if visual_input_size == 56:
            img = p.image
        else:
            img = downsample_scene(p.image, out_size=28)
        data.append((img, p.label))

    combos = sorted({
        _parse_label(lbl) for (_img, lbl) in data
    })
    train_combos = {c for c in combos if c[0] != held_out_color}
    test_combos = {c for c in combos if c[0] == held_out_color}

    train_data = [(img, lbl) for (img, lbl) in data if _parse_label(lbl) in train_combos]
    test_data = [(img, lbl) for (img, lbl) in data if _parse_label(lbl) in test_combos]

    train_metrics = _eval_metrics(brain, train_data, top_k=args.top_k)
    test_metrics = _eval_metrics(brain, test_data, top_k=args.top_k)

    print("=== Compositional Generalization Metrics ===")
    print(f"Train n={train_metrics['n']}")
    print(f"  winner_match: {train_metrics['winner_match']:.3f}")
    print(f"  top{args.top_k}_overlap: {train_metrics['topk_overlap']:.3f}")
    print(f"  top{args.top_k}_hit: {train_metrics['topk_hit']:.3f}")
    print(f"  cosine_feat: {train_metrics['cosine_feat']:.3f}")
    print(f"  cosine_proj: {train_metrics['cosine_proj']:.3f}")

    print(f"Held-out (obj1 color={held_out_color}) n={test_metrics['n']}")
    print(f"  winner_match: {test_metrics['winner_match']:.3f}")
    print(f"  top{args.top_k}_overlap: {test_metrics['topk_overlap']:.3f}")
    print(f"  top{args.top_k}_hit: {test_metrics['topk_hit']:.3f}")
    print(f"  cosine_feat: {test_metrics['cosine_feat']:.3f}")
    print(f"  cosine_proj: {test_metrics['cosine_proj']:.3f}")

    out = {
        'checkpoint': str(args.checkpoint),
        'config': config,
        'train_metrics': train_metrics,
        'held_out_metrics': test_metrics,
        'split': {
            'held_out_color': held_out_color,
            'train_combos': [' '.join(c) for c in sorted(train_combos)],
            'test_combos': [' '.join(c) for c in sorted(test_combos)],
        }
    }

    out_path = Path('two_object_results') / 'two_object_compositional_eval.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
