#!/usr/bin/env python3
"""
Multi-seed validation for Distributed ATL.
Runs training with multiple seeds and aggregates results.
"""

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

import json
import subprocess
from pathlib import Path
import numpy as np

SEEDS = [0, 42, 123, 456, 999]
TEMPERATURE = 0.2

def main():
    results_dir = Path('two_object_results')
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    print("=" * 60)
    print("MULTI-SEED VALIDATION: Distributed ATL (temp=0.2)")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print()
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        cmd = [
            'python3', 'experiments/train_two_object_distributed.py',
            '--seed', str(seed),
            '--temperature', str(TEMPERATURE),
        ]
        
        result = subprocess.run(
            cmd,
            cwd='/home/ahmed/Dokumente/Neuroscience/CHPL',
            capture_output=False,
        )
        
        if result.returncode != 0:
            print(f"ERROR: Seed {seed} failed!")
            continue
        
        # Find the most recent result file for this seed
        json_files = sorted(results_dir.glob(f'distributed_atl_seed{seed}_*.json'))
        if json_files:
            with open(json_files[-1]) as f:
                data = json.load(f)
            all_results.append({
                'seed': seed,
                'train_sim': data['evaluation']['train_pattern_similarity'],
                'held_out_sim': data['evaluation']['held_out_pattern_similarity'],
                'gap': data['evaluation']['gap'],
                'active_concepts': data['stats']['active_concepts'],
            })
    
    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    
    if not all_results:
        print("No results collected!")
        return
    
    train_sims = [r['train_sim'] for r in all_results]
    held_out_sims = [r['held_out_sim'] for r in all_results]
    gaps = [r['gap'] for r in all_results]
    concepts = [r['active_concepts'] for r in all_results]
    
    print(f"\nN seeds: {len(all_results)}")
    print(f"\nTrain pattern_similarity:")
    print(f"  Mean: {np.mean(train_sims):.3f} ± {np.std(train_sims):.3f}")
    print(f"  Range: [{min(train_sims):.3f}, {max(train_sims):.3f}]")
    
    print(f"\nHeld-out pattern_similarity:")
    print(f"  Mean: {np.mean(held_out_sims):.3f} ± {np.std(held_out_sims):.3f}")
    print(f"  Range: [{min(held_out_sims):.3f}, {max(held_out_sims):.3f}]")
    
    print(f"\nGap (train - held_out):")
    print(f"  Mean: {np.mean(gaps):.3f} ± {np.std(gaps):.3f}")
    
    print(f"\nActive concepts:")
    print(f"  Mean: {np.mean(concepts):.1f} ± {np.std(concepts):.1f}")
    
    # Per-seed breakdown
    print("\n" + "-" * 60)
    print("Per-seed breakdown:")
    print("-" * 60)
    print(f"{'Seed':<8} {'Train':<10} {'Held-out':<10} {'Gap':<10} {'Concepts':<10}")
    for r in all_results:
        print(f"{r['seed']:<8} {r['train_sim']:<10.3f} {r['held_out_sim']:<10.3f} {r['gap']:<10.3f} {r['active_concepts']:<10}")
    
    # Comparison with baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH A2 BASELINE")
    print("=" * 60)
    print(f"A2 Baseline (winner-takes-all): Held-out = 0.512")
    print(f"Distributed ATL (mean):         Held-out = {np.mean(held_out_sims):.3f}")
    print(f"Improvement: {(np.mean(held_out_sims) - 0.512) / 0.512 * 100:.1f}%")
    
    # Robustness check
    robust = all(h > 0.65 for h in held_out_sims)
    if robust:
        print("\n✓✓ ROBUST: All seeds > 0.65 held-out similarity")
    else:
        failed = [r['seed'] for r in all_results if r['held_out_sim'] < 0.65]
        print(f"\n⚠ WARNING: Seeds {failed} below 0.65 threshold")
    
    # Save aggregated results
    agg_file = results_dir / 'multiseed_validation_summary.json'
    with open(agg_file, 'w') as f:
        json.dump({
            'seeds': SEEDS,
            'temperature': TEMPERATURE,
            'per_seed': all_results,
            'aggregated': {
                'train_mean': float(np.mean(train_sims)),
                'train_std': float(np.std(train_sims)),
                'held_out_mean': float(np.mean(held_out_sims)),
                'held_out_std': float(np.std(held_out_sims)),
                'gap_mean': float(np.mean(gaps)),
                'gap_std': float(np.std(gaps)),
                'concepts_mean': float(np.mean(concepts)),
                'concepts_std': float(np.std(concepts)),
            },
            'baseline_comparison': {
                'a2_baseline_held_out': 0.512,
                'distributed_held_out_mean': float(np.mean(held_out_sims)),
                'improvement_pct': float((np.mean(held_out_sims) - 0.512) / 0.512 * 100),
            },
            'robust': robust,
        }, f, indent=2)
    
    print(f"\nSaved summary: {agg_file}")


if __name__ == '__main__':
    main()
