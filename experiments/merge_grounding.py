#!/usr/bin/env python3
"""
Merge Multiple Grounding Sources

Combines grounding from:
1. COCO captions (highest priority - direct visual)
2. ImageNet classes (high priority - direct visual)
3. Multi-pass propagation (lower priority - derived)

Priority order ensures direct visual grounding takes precedence.
"""

import sys
sys.path.insert(0, '..')

import os
import glob
import pickle
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_grounding_file(path: str) -> Dict[str, Any]:
    """Load a grounding file and return word -> activation dict."""
    print(f"Loading: {path}")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    grounding = {}
    
    if isinstance(data, dict):
        for word, value in data.items():
            # Check if it's a GroundedWord object
            if hasattr(value, 'activation'):
                activation = value.activation
            else:
                activation = value
            
            # Convert to numpy if tensor
            if isinstance(activation, torch.Tensor):
                activation = activation.cpu().numpy()
            
            grounding[word] = activation
    
    print(f"  Loaded {len(grounding)} words")
    return grounding


def find_grounding_files(base_dir: str) -> Dict[str, str]:
    """Find all available grounding files."""
    files = {}
    
    # Look for different grounding sources
    patterns = [
        ('coco', '**/grounded_vocabulary*coco*.pkl'),
        ('imagenet', '**/grounded_vocabulary*imagenet*.pkl'),
        ('multipass', '**/grounded_vocabulary*multipass*.pkl'),
        ('default', '**/grounded_vocabulary.pkl'),
    ]
    
    for source, pattern in patterns:
        matches = glob.glob(str(Path(base_dir) / pattern), recursive=True)
        if matches:
            # Get most recent
            latest = max(matches, key=os.path.getmtime)
            files[source] = latest
    
    return files


def merge_grounding_sources(
    sources: Dict[str, str],
    output_path: str,
    priority_order: list = None
):
    """
    Merge multiple grounding sources with priority.
    
    Args:
        sources: Dict mapping source name -> file path
        output_path: Output path for merged grounding
        priority_order: Order of priority (first = lowest, last = highest)
    """
    if priority_order is None:
        # Default priority: multipass < default < imagenet < coco
        priority_order = ['multipass', 'default', 'imagenet', 'coco']
    
    print("=" * 60)
    print("MERGE GROUNDING SOURCES")
    print("=" * 60)
    print(f"Found sources: {list(sources.keys())}")
    print(f"Priority order: {priority_order}")
    print()
    
    # Load and merge in priority order
    merged = {}
    source_counts = {}
    
    for source in priority_order:
        if source not in sources:
            continue
        
        grounding = load_grounding_file(sources[source])
        
        # Track what's new vs overwritten
        new_words = 0
        overwritten = 0
        
        for word, activation in grounding.items():
            if word in merged:
                overwritten += 1
            else:
                new_words += 1
            merged[word] = activation
        
        source_counts[source] = {
            'total': len(grounding),
            'new': new_words,
            'overwritten': overwritten
        }
        
        print(f"  {source}: {len(grounding)} words ({new_words} new, {overwritten} updated)")
    
    print(f"\nFinal merged vocabulary: {len(merged)} words")
    
    # Save merged grounding
    with open(output_path, 'wb') as f:
        pickle.dump(merged, f)
    
    print(f"Saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    for source, counts in source_counts.items():
        print(f"  {source}: {counts['total']} total, {counts['new']} contributed unique")
    print(f"  FINAL: {len(merged)} grounded words")
    
    return merged


def run_merge():
    """Run the merge process."""
    base_dir = Path(__file__).parent.parent
    
    # Find available grounding files
    sources = find_grounding_files(str(base_dir))
    
    if not sources:
        print("No grounding files found!")
        print("Run grounding scripts first:")
        print("  - ground_vocabulary.py (COCO)")
        print("  - ground_vocabulary_multipass.py")
        return None
    
    # Output path
    output_path = str(base_dir / 'grounded_vocabulary_final.pkl')
    
    # Merge
    merged = merge_grounding_sources(sources, output_path)
    
    return merged


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge grounding sources')
    parser.add_argument('--coco', type=str, help='COCO grounding file')
    parser.add_argument('--imagenet', type=str, help='ImageNet grounding file')
    parser.add_argument('--multipass', type=str, help='Multi-pass grounding file')
    parser.add_argument('--output', type=str, default='grounded_vocabulary_final.pkl',
                        help='Output path')
    
    args = parser.parse_args()
    
    # Build sources from args or auto-discover
    if args.coco or args.imagenet or args.multipass:
        sources = {}
        if args.coco:
            sources['coco'] = args.coco
        if args.imagenet:
            sources['imagenet'] = args.imagenet
        if args.multipass:
            sources['multipass'] = args.multipass
        
        merge_grounding_sources(sources, args.output)
    else:
        run_merge()
