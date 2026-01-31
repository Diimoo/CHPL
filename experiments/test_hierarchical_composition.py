#!/usr/bin/env python3
"""
Phase B Test: Hierarchical Composition

Test if Distributed ATL can handle nested relations:
- Train on depth 1-2, test on depth 3 (hierarchy generalization)
- Train on depth 1-3 mixed, test on held-out combinations
"""

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL-exploration')

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE
from synthetic_environment_hierarchical import generate_hierarchical_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_per_depth', type=int, default=300)
    ap.add_argument('--split', choices=['depth', 'mixed'], default='depth',
                    help='depth: train 1-2, test 3. mixed: random split all depths')
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    out_dir = Path('hierarchical_results')
    out_dir.mkdir(exist_ok=True)
    run_id = f"hierarchical_{args.split}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Phase B: Hierarchical Composition Test")
    print(f"Split: {args.split}")
    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {DEVICE}")
    
    # Generate data
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']
    canvas_size = 56
    
    print("\nGenerating hierarchical scenes...")
    all_pairs = generate_hierarchical_pairs(
        colors=colors,
        shapes=shapes,
        depths=[1, 2, 3],
        n_per_depth=args.n_per_depth,
        canvas_size=canvas_size,
    )
    
    # Count by depth
    for d in [1, 2, 3]:
        count = len([p for p in all_pairs if p.hierarchy_depth == d])
        print(f"  Depth {d}: {count} scenes")
    
    # Split data
    if args.split == 'depth':
        # Train on depth 1-2, test on depth 3 (hierarchy generalization)
        train_pairs = [p for p in all_pairs if p.hierarchy_depth <= 2]
        test_pairs = [p for p in all_pairs if p.hierarchy_depth == 3]
        split_desc = "Train: depth 1-2, Test: depth 3 (novel hierarchy level)"
    else:
        # Mixed: random 80/20 split across all depths
        np.random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * 0.8)
        train_pairs = all_pairs[:split_idx]
        test_pairs = all_pairs[split_idx:]
        split_desc = "Random 80/20 split across all depths"
    
    # Convert to (image, label) format
    # Use hierarchical label (with parentheses) for language
    train_data = [(p.image, p.label) for p in train_pairs]
    test_data = [(p.image, p.label) for p in test_pairs]
    
    # Also prepare flat-label version for comparison
    train_data_flat = [(p.image, p.flat_label) for p in train_pairs]
    test_data_flat = [(p.image, p.flat_label) for p in test_pairs]
    
    print(f"\n{split_desc}")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Show example labels
    print("\nExample hierarchical labels:")
    for d in [1, 2, 3]:
        examples = [p.label for p in all_pairs if p.hierarchy_depth == d][:1]
        if examples:
            print(f"  Depth {d}: {examples[0]}")
    
    # Create model
    brain = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=300,  # More for hierarchical
        visual_input_size=canvas_size,
        use_distributed_atl=True,
        distributed_temperature=args.temperature,
    )
    
    # Phase 1: Visual reconstruction
    print("\nPHASE 1: Visual reconstruction")
    vis_opt = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(12):
        np.random.shuffle(train_data)
        losses = []
        for img, _ in train_data:
            vis_opt.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(img).float())
            loss.backward()
            vis_opt.step()
            losses.append(loss.item())
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")
    
    # Phase 2: Language alignment (with hierarchical labels)
    print("\nPHASE 2: Language alignment (hierarchical)")
    lang_opt = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(20):
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
    print("\nPHASE 3: Distributed binding")
    for epoch in range(12):
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
    
    def eval_by_depth(pairs):
        results = {}
        for d in [1, 2, 3]:
            depth_data = [(p.image, p.label) for p in pairs if p.hierarchy_depth == d]
            if depth_data:
                results[d] = eval_similarity(depth_data)
        return results
    
    print("\n" + "=" * 60)
    print("RESULTS: Hierarchical Composition")
    print("=" * 60)
    
    train_sim = eval_similarity(train_data)
    test_sim = eval_similarity(test_data)
    
    print(f"\nOverall:")
    print(f"  Train: {train_sim:.3f}")
    print(f"  Test:  {test_sim:.3f}")
    print(f"  Gap:   {train_sim - test_sim:.3f}")
    
    # Per-depth breakdown
    print("\nPer-depth (on test set):")
    test_by_depth = eval_by_depth(test_pairs)
    for d, sim in sorted(test_by_depth.items()):
        marker = "(TRAIN DEPTH)" if (args.split == 'depth' and d <= 2) else "(TEST DEPTH)"
        if args.split == 'mixed':
            marker = ""
        print(f"  Depth {d}: {sim:.3f} {marker}")
    
    # Verdict
    if args.split == 'depth':
        depth3_sim = test_by_depth.get(3, 0)
        if depth3_sim > 0.55:
            verdict = "SUCCESS - generalizes to deeper hierarchy!"
        elif depth3_sim > 0.45:
            verdict = "PARTIAL - some hierarchy generalization"
        else:
            verdict = "FAIL - cannot generalize to deeper hierarchy"
    else:
        if test_sim > 0.6:
            verdict = "SUCCESS - learns hierarchical composition"
        elif test_sim > 0.5:
            verdict = "PARTIAL - some hierarchical learning"
        else:
            verdict = "FAIL - struggles with hierarchy"
    
    print(f"\nVerdict: {verdict}")
    
    # Save results
    results = {
        'run_id': run_id,
        'split': args.split,
        'split_desc': split_desc,
        'config': {
            'temperature': args.temperature,
            'seed': args.seed,
            'n_per_depth': args.n_per_depth,
        },
        'evaluation': {
            'train_similarity': train_sim,
            'test_similarity': test_sim,
            'gap': train_sim - test_sim,
            'test_by_depth': {str(k): v for k, v in test_by_depth.items()},
        },
        'stats': brain.atl.get_stats(),
        'verdict': verdict,
    }
    
    out_json = out_dir / f'{run_id}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")
    
    return results


if __name__ == '__main__':
    main()
