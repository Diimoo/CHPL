#!/usr/bin/env python3
"""
Phase A: Quick COCO Feasibility Test

Tests if Distributed ATL can handle natural images from COCO.
Uses a small subset (100-500 images) for fast iteration.
"""

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from brain_crossmodal_learner import BrainCrossModalLearner, DEVICE

try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


class VisualCortex224(nn.Module):
    """Visual cortex for 224x224 natural images."""
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Encoder: 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> feature
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),   # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), # 56 -> 28
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride=2, padding=2),# 28 -> 14
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride=2, padding=2),# 14 -> 7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 1, 1)),
            nn.ConvTranspose2d(128, 128, 7),            # 1 -> 7
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),    # 112 -> 224
            nn.Sigmoid(),
        )
        
        self._init_weights()
        self.to(DEVICE)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = x.to(DEVICE)
        feat = self.encoder(x)
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0) if feat.size(0) == 1 else feat
    
    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        return self.decoder(feat)
    
    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.to(DEVICE)
        feat = self.encoder(x)
        recon = self.decoder(feat)
        return F.mse_loss(recon, x)


def load_coco_subset(coco_root: str, n_samples: int = 100, split: str = 'train'):
    """Load a subset of COCO for quick testing."""
    coco_root = Path(os.path.expanduser(coco_root))
    
    ann_file = coco_root / 'annotations' / f'instances_{split}2017.json'
    cap_file = coco_root / 'annotations' / f'captions_{split}2017.json'
    
    print(f"Loading COCO {split} annotations...")
    coco = COCO(str(ann_file))
    coco_caps = COCO(str(cap_file))
    
    cat_id_to_name = {
        cat['id']: cat['name'] 
        for cat in coco.loadCats(coco.getCatIds())
    }
    
    # Filter for 2-4 object scenes
    samples = []
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        categories = set(cat_id_to_name[ann['category_id']] for ann in anns)
        n_cats = len(categories)
        
        if 2 <= n_cats <= 4:
            cap_ids = coco_caps.getAnnIds(imgIds=img_id)
            caps = coco_caps.loadAnns(cap_ids)
            caption = caps[0]['caption'] if caps else ""
            
            img_info = coco.loadImgs(img_id)[0]
            # Try both path structures (images/train2017 or train2017)
            img_path = coco_root / f'{split}2017' / img_info['file_name']
            if not img_path.exists():
                img_path = coco_root / 'images' / f'{split}2017' / img_info['file_name']
            
            if img_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'caption': caption,
                    'categories': list(categories),
                    'n_objects': n_cats,
                })
            
            if len(samples) >= n_samples:
                break
    
    print(f"Loaded {len(samples)} COCO samples")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco_root', type=str, default='~/Datasets/coco')
    ap.add_argument('--subset_size', type=int, default=100)
    ap.add_argument('--epochs_visual', type=int, default=8)
    ap.add_argument('--epochs_lang', type=int, default=12)
    ap.add_argument('--epochs_binding', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--image_size', type=int, default=224)
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    out_dir = Path('coco_results')
    out_dir.mkdir(exist_ok=True)
    run_id = f"coco_quick_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 60)
    print("Phase A: COCO Quick Feasibility Test")
    print("=" * 60)
    print(f"Subset size: {args.subset_size}")
    print(f"Image size: {args.image_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {DEVICE}")
    
    # Load data
    samples = load_coco_subset(args.coco_root, args.subset_size)
    
    # Image transform
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
    ])
    
    def load_image(path):
        img = Image.open(path).convert('RGB')
        img_t = transform(img)
        return img_t.permute(1, 2, 0).numpy()  # [H, W, C]
    
    # Load all images
    print("\nLoading images...")
    data = []
    for s in samples:
        try:
            img = load_image(s['image_path'])
            data.append((img, s['caption']))
        except Exception as e:
            print(f"  Skip {s['image_path']}: {e}")
    print(f"Loaded {len(data)} images")
    
    # Split train/test
    np.random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create model with custom visual cortex
    print("\nCreating model...")
    visual_cortex = VisualCortex224(feature_dim=64)
    
    # Create brain with placeholder visual (we'll replace it)
    brain = BrainCrossModalLearner(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=56,  # Placeholder
        use_distributed_atl=True,
        distributed_temperature=args.temperature,
    )
    # Replace with our 224 visual cortex
    brain.visual = visual_cortex
    
    # Phase 1: Visual reconstruction
    print("\nPHASE 1: Visual reconstruction")
    vis_opt = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    for epoch in range(args.epochs_visual):
        np.random.shuffle(train_data)
        losses = []
        for img, _ in train_data:
            vis_opt.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(img).float())
            loss.backward()
            vis_opt.step()
            losses.append(loss.item())
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")
    
    final_vis_loss = np.mean(losses)
    
    # Phase 2: Language alignment
    print("\nPHASE 2: Language alignment")
    lang_opt = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    for epoch in range(args.epochs_lang):
        np.random.shuffle(train_data)
        losses = []
        for img, caption in train_data:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(img).float())
            lang_feat = brain.language(caption)
            loss = 1 - F.cosine_similarity(vis_feat.unsqueeze(0), lang_feat.unsqueeze(0))
            lang_opt.zero_grad()
            loss.backward()
            lang_opt.step()
            losses.append(loss.item())
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")
    
    final_lang_loss = np.mean(losses)
    
    # Phase 3: Distributed binding
    print("\nPHASE 3: Distributed binding")
    for epoch in range(args.epochs_binding):
        np.random.shuffle(train_data)
        qualities = []
        for img, caption in train_data:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(caption)
            result = brain.atl.consolidate(vis_feat.detach(), lang_feat.detach())
            qualities.append(result['pattern_similarity'])
        stats = brain.atl.get_stats()
        print(f"  Epoch {epoch+1}: pattern_sim={np.mean(qualities):.3f}, concepts={stats['active_concepts']}")
    
    # Evaluate
    def eval_similarity(dataset):
        sims = []
        for img, caption in dataset:
            img_t = torch.from_numpy(img).float()
            vis_feat = brain.visual(img_t)
            lang_feat = brain.language(caption)
            sim = brain.atl.similarity(vis_feat, lang_feat)
            sims.append(sim)
        return float(np.mean(sims))
    
    print("\n" + "=" * 60)
    print("RESULTS: COCO Quick Test")
    print("=" * 60)
    
    train_sim = eval_similarity(train_data)
    test_sim = eval_similarity(test_data)
    
    print(f"\nPattern Similarity:")
    print(f"  Train: {train_sim:.3f}")
    print(f"  Test:  {test_sim:.3f}")
    print(f"  Gap:   {train_sim - test_sim:.3f}")
    
    print(f"\nFinal losses:")
    print(f"  Visual reconstruction: {final_vis_loss:.4f}")
    print(f"  Language alignment: {final_lang_loss:.4f}")
    
    # Verdict
    if train_sim > 0.5:
        verdict = "SUCCESS - Distributed ATL works on natural images!"
        recommendation = "INTEGRATE into paper as Section 4.4"
    elif train_sim > 0.35:
        verdict = "PARTIAL - Some learning, but needs more work"
        recommendation = "Mention in Discussion/Future Work, don't integrate results"
    else:
        verdict = "FAIL - Natural images need fundamental changes"
        recommendation = "Skip for Paper 1, Phase A = long-term project"
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'run_id': run_id,
        'config': {
            'subset_size': args.subset_size,
            'image_size': args.image_size,
            'temperature': args.temperature,
            'epochs_visual': args.epochs_visual,
            'epochs_lang': args.epochs_lang,
            'epochs_binding': args.epochs_binding,
            'seed': args.seed,
        },
        'data': {
            'train_samples': len(train_data),
            'test_samples': len(test_data),
        },
        'evaluation': {
            'train_similarity': train_sim,
            'test_similarity': test_sim,
            'gap': train_sim - test_sim,
            'visual_loss': final_vis_loss,
            'language_loss': final_lang_loss,
        },
        'stats': brain.atl.get_stats(),
        'verdict': verdict,
        'recommendation': recommendation,
    }
    
    out_json = out_dir / f'{run_id}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")
    
    # Save checkpoint
    out_pt = out_dir / f'{run_id}.pt'
    torch.save({
        'visual_state': brain.visual.state_dict(),
        'language_state': brain.language.state_dict(),
        'atl_state': brain.atl.state_dict() if hasattr(brain.atl, 'state_dict') else None,
        'config': results['config'],
    }, out_pt)
    print(f"Saved: {out_pt}")
    
    return results


if __name__ == '__main__':
    main()
