#!/usr/bin/env python3
"""
Harder compositional splits for Distributed ATL.
Tests:
1. Relation holdout: Train {above, left_of}, Test {below, right_of}
2. Swap generalization: Train A above B, Test B above A
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
    return {
        'color1': parts[0],
        'shape1': parts[1],
        'relation': parts[2],
        'color2': parts[3],
        'shape2': parts[4],
    }


def run_experiment(
    split_type: str,
    temperature: float = 0.2,
    seed: int = 0,
    n_epochs_visual: int = 10,
    n_epochs_lang: int = 15,
    n_epochs_binding: int = 10,
):
    """Run experiment with specified split type."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    out_dir = Path('two_object_results')
    out_dir.mkdir(exist_ok=True)
    run_id = f"{split_type}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate data with all relations
    all_colors = ['red', 'blue', 'green']
    all_shapes = ['circle', 'square']
    all_relations = ['above', 'left_of', 'below', 'right_of']
    canvas_size = 56
    
    pairs = generate_two_object_pairs(
        colors=all_colors,
        shapes=all_shapes,
        relations=all_relations,
        n_per_combination=15,
        canvas_size=canvas_size,
    )
    
    data = [(p.image, p.label) for p in pairs]
    
    # Split based on type
    if split_type == 'relation_holdout':
        train_relations = {'above', 'left_of'}
        test_relations = {'below', 'right_of'}
        train_data = [(img, lbl) for img, lbl in data 
                      if _parse_label(lbl)['relation'] in train_relations]
        test_data = [(img, lbl) for img, lbl in data 
                     if _parse_label(lbl)['relation'] in test_relations]
        split_desc = f"Train: {train_relations}, Test: {test_relations}"
        
    elif split_type == 'swap_generalization':
        # Train: obj1 has specific colors, Test: swapped
        # e.g., Train: red above blue, Test: blue above red
        train_data = []
        test_data = []
        for img, lbl in data:
            p = _parse_label(lbl)
            # Train: color1 < color2 alphabetically
            if p['color1'] < p['color2']:
                train_data.append((img, lbl))
            else:
                test_data.append((img, lbl))
        split_desc = "Train: color1 < color2, Test: color1 >= color2 (swapped order)"
        
    elif split_type == 'novel_combination':
        # Train: seen color+shape combos, Test: unseen
        # Hold out green shapes in position 1
        train_data = []
        test_data = []
        for img, lbl in data:
            p = _parse_label(lbl)
            if p['color1'] == 'green':
                test_data.append((img, lbl))
            else:
                train_data.append((img, lbl))
        split_desc = "Train: obj1 != green, Test: obj1 = green"
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    print(f"Split: {split_type}")
    print(f"Description: {split_desc}")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"Temperature: {temperature}, Seed: {seed}")
    print(f"Device: {DEVICE}")
    
    # Create model
    brain = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=canvas_size,
        use_distributed_atl=True,
        distributed_temperature=temperature,
    )
    
    # Phase 1: Visual reconstruction
    print("\nPHASE 1: Visual reconstruction")
    vis_opt = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(n_epochs_visual):
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
    for epoch in range(n_epochs_lang):
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
    
    # Phase 3: Binding
    print("\nPHASE 3: Distributed binding")
    for epoch in range(n_epochs_binding):
        np.random.shuffle(train_data)
        qualities = []
        for img, label in train_data:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(label)
            result = brain.atl.consolidate(vis_feat.detach(), lang_feat.detach())
            qualities.append(result['pattern_similarity'])
        stats = brain.atl.get_stats()
        print(f"  Epoch {epoch+1}: pattern_sim={np.mean(qualities):.3f}, concepts={stats['active_concepts']}")
    
    # Evaluate
    def eval_similarity(dataset):
        sims = []
        for img, label in dataset:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(label)
            sim = brain.atl.similarity(vis_feat, lang_feat)
            sims.append(sim)
        return float(np.mean(sims))
    
    train_sim = eval_similarity(train_data)
    test_sim = eval_similarity(test_data)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {split_type}")
    print("=" * 60)
    print(f"Train pattern similarity:    {train_sim:.3f}")
    print(f"Test pattern similarity:     {test_sim:.3f}")
    print(f"Gap: {train_sim - test_sim:.3f}")
    print(f"Active concepts: {brain.atl.get_stats()['active_concepts']}")
    
    # Interpret results
    if test_sim > 0.6:
        verdict = "SUCCESS - generalizes to novel split"
    elif test_sim > 0.5:
        verdict = "PARTIAL - some generalization"
    else:
        verdict = "FAIL - does not generalize"
    print(f"\nVerdict: {verdict}")
    
    # Save
    results = {
        'run_id': run_id,
        'split_type': split_type,
        'split_desc': split_desc,
        'config': {
            'temperature': temperature,
            'seed': seed,
            'n_train': len(train_data),
            'n_test': len(test_data),
        },
        'evaluation': {
            'train_similarity': train_sim,
            'test_similarity': test_sim,
            'gap': train_sim - test_sim,
        },
        'stats': brain.atl.get_stats(),
        'verdict': verdict,
    }
    
    out_json = out_dir / f'harder_split_{run_id}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['relation_holdout', 'swap_generalization', 'novel_combination', 'all'],
                    default='all')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    
    splits = ['relation_holdout', 'swap_generalization', 'novel_combination'] if args.split == 'all' else [args.split]
    
    all_results = []
    for split in splits:
        print("\n" + "#" * 70)
        print(f"# SPLIT: {split}")
        print("#" * 70)
        result = run_experiment(split, args.temperature, args.seed)
        all_results.append(result)
    
    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY: All Harder Splits")
        print("=" * 70)
        print(f"{'Split':<25} {'Train':<10} {'Test':<10} {'Gap':<10} {'Verdict'}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['split_type']:<25} {r['evaluation']['train_similarity']:<10.3f} "
                  f"{r['evaluation']['test_similarity']:<10.3f} {r['evaluation']['gap']:<10.3f} "
                  f"{r['verdict']}")


if __name__ == '__main__':
    main()
