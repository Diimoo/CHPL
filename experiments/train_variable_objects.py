#!/usr/bin/env python3
"""
Milestone 2: Variable Objects (1-4)
Train on 1-3 objects, test on 4 objects (count generalization)
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
from synthetic_environment import generate_variable_object_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_epochs_visual', type=int, default=15)
    ap.add_argument('--n_epochs_lang', type=int, default=20)
    ap.add_argument('--n_epochs_binding', type=int, default=15)
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    out_dir = Path('two_object_results')
    out_dir.mkdir(exist_ok=True)
    run_id = f"variable_objects_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square', 'triangle']
    canvas_size = 56
    
    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {DEVICE}")
    
    # Generate data with 1-4 objects
    print("\nGenerating variable object scenes...")
    all_pairs = generate_variable_object_pairs(
        colors=colors,
        shapes=shapes,
        n_objects_range=(1, 4),
        n_per_config=20,
        canvas_size=canvas_size,
    )
    
    # Split: train on 1-3 objects, test on 4 objects
    train_data = [(p.image, p.label) for p in all_pairs if p.n_objects <= 3]
    test_data = [(p.image, p.label) for p in all_pairs if p.n_objects == 4]
    
    # Also create per-count test sets
    test_by_count = {
        n: [(p.image, p.label) for p in all_pairs if p.n_objects == n]
        for n in [1, 2, 3, 4]
    }
    
    print(f"\nData summary:")
    for n in [1, 2, 3, 4]:
        count = len([p for p in all_pairs if p.n_objects == n])
        print(f"  {n} objects: {count} scenes")
    print(f"\nTrain (1-3 objects): {len(train_data)}")
    print(f"Test (4 objects): {len(test_data)}")
    
    # Show example labels
    print("\nExample labels:")
    for n in [1, 2, 3, 4]:
        examples = [p.label for p in all_pairs if p.n_objects == n][:2]
        print(f"  {n} obj: {examples}")
    
    # Create model
    brain = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=300,  # More concepts for more complex scenes
        visual_input_size=canvas_size,
        use_distributed_atl=True,
        distributed_temperature=args.temperature,
    )
    
    # Phase 1: Visual reconstruction
    print("\nPHASE 1: Visual reconstruction")
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
    
    # Phase 3: Binding
    print("\nPHASE 3: Distributed binding")
    for epoch in range(args.n_epochs_binding):
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
        if not dataset:
            return 0.0
        sims = []
        for img, label in dataset:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(label)
            sim = brain.atl.similarity(vis_feat, lang_feat)
            sims.append(sim)
        return float(np.mean(sims))
    
    print("\n" + "=" * 60)
    print("RESULTS: Variable Object Generalization")
    print("=" * 60)
    
    results_by_count = {}
    for n in [1, 2, 3, 4]:
        sim = eval_similarity(test_by_count[n])
        results_by_count[n] = sim
        marker = "(TRAIN)" if n <= 3 else "(TEST - novel count!)"
        print(f"  {n} objects: {sim:.3f} {marker}")
    
    train_sim = eval_similarity(train_data)
    test_sim = eval_similarity(test_data)
    
    print(f"\nOverall:")
    print(f"  Train (1-3 obj): {train_sim:.3f}")
    print(f"  Test (4 obj):    {test_sim:.3f}")
    print(f"  Gap: {train_sim - test_sim:.3f}")
    
    # Verdict
    if test_sim > 0.6:
        verdict = "SUCCESS - generalizes to novel object count!"
    elif test_sim > 0.5:
        verdict = "PARTIAL - some generalization to 4 objects"
    else:
        verdict = "FAIL - does not generalize to 4 objects"
    print(f"\nVerdict: {verdict}")
    
    # Save results
    results = {
        'run_id': run_id,
        'config': {
            'temperature': args.temperature,
            'seed': args.seed,
            'train_counts': [1, 2, 3],
            'test_count': 4,
        },
        'per_count_similarity': results_by_count,
        'evaluation': {
            'train_similarity': train_sim,
            'test_similarity': test_sim,
            'gap': train_sim - test_sim,
        },
        'stats': brain.atl.get_stats(),
        'verdict': verdict,
    }
    
    out_json = out_dir / f'{run_id}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    out_ckpt = out_dir / f'{run_id}.pt'
    torch.save({
        'config': results['config'],
        'visual_state': brain.visual.state_dict(),
        'language_state': brain.language.state_dict(),
        'atl_state': brain.atl.state_dict(),
    }, out_ckpt)
    
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_ckpt}")


if __name__ == '__main__':
    main()
