#!/usr/bin/env python3
"""
Phase A: COCO Data Loader

Loads COCO images with 2-4 objects for compositional scene understanding.
Uses captions as language supervision.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("Warning: pycocotools not installed. Run: pip install pycocotools")


@dataclass
class COCOScenePair:
    """A COCO scene with image and caption."""
    image: np.ndarray  # [H, W, 3] normalized to [0, 1]
    caption: str
    image_id: int
    n_objects: int
    categories: List[str]


class COCOScenes(Dataset):
    """
    COCO dataset for compositional scene understanding.
    
    Filters for images with 2-4 distinct object categories.
    Uses captions as language supervision.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        min_objects: int = 2,
        max_objects: int = 4,
        image_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root: Path to COCO dataset (containing 'images/' and 'annotations/')
            split: 'train' or 'val'
            min_objects: Minimum number of distinct object categories
            max_objects: Maximum number of distinct object categories
            image_size: Resize images to this size
            max_samples: Limit number of samples (for quick testing)
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        if not HAS_COCO:
            raise ImportError("pycocotools required. Install with: pip install pycocotools")
        
        # Load annotations
        ann_file = self.root / 'annotations' / f'instances_{split}2017.json'
        cap_file = self.root / 'annotations' / f'captions_{split}2017.json'
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(str(ann_file))
        self.coco_caps = COCO(str(cap_file)) if cap_file.exists() else None
        
        # Get category names
        self.cat_id_to_name = {
            cat['id']: cat['name'] 
            for cat in self.coco.loadCats(self.coco.getCatIds())
        }
        
        # Filter images by number of distinct object categories
        self.samples = []
        print(f"Filtering images with {min_objects}-{max_objects} object categories...")
        
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Count distinct categories
            categories = set(self.cat_id_to_name[ann['category_id']] for ann in anns)
            n_cats = len(categories)
            
            if min_objects <= n_cats <= max_objects:
                # Get caption
                caption = ""
                if self.coco_caps:
                    cap_ids = self.coco_caps.getAnnIds(imgIds=img_id)
                    caps = self.coco_caps.loadAnns(cap_ids)
                    if caps:
                        caption = caps[0]['caption']
                
                self.samples.append({
                    'image_id': img_id,
                    'n_objects': n_cats,
                    'categories': list(categories),
                    'caption': caption,
                })
                
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        print(f"Found {len(self.samples)} images with {min_objects}-{max_objects} categories")
        
        # Image transform
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> COCOScenePair:
        sample = self.samples[idx]
        
        # Load image
        img_info = self.coco.loadImgs(sample['image_id'])[0]
        img_path = self.root / 'images' / f'{self.split}2017' / img_info['file_name']
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Convert to [H, W, C] numpy for compatibility
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        return COCOScenePair(
            image=image_np,
            caption=sample['caption'],
            image_id=sample['image_id'],
            n_objects=sample['n_objects'],
            categories=sample['categories'],
        )
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        n_by_count = {}
        for sample in self.samples:
            n = sample['n_objects']
            n_by_count[n] = n_by_count.get(n, 0) + 1
        
        return {
            'total': len(self.samples),
            'by_object_count': n_by_count,
            'split': self.split,
        }


class SimpleCOCOLoader:
    """
    Simplified COCO loader that doesn't require pycocotools.
    Uses pre-processed subset for quick testing.
    """
    
    def __init__(self, subset_path: str):
        """
        Args:
            subset_path: Path to pre-processed subset JSON file
        """
        self.subset_path = Path(subset_path)
        
        if not self.subset_path.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_path}")
        
        with open(subset_path) as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from subset")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_coco_subset(
    coco_root: str,
    output_path: str,
    n_samples: int = 1000,
    min_objects: int = 2,
    max_objects: int = 4,
):
    """
    Create a small COCO subset for quick testing.
    Saves image paths and captions to JSON.
    """
    dataset = COCOScenes(
        root=coco_root,
        split='train',
        min_objects=min_objects,
        max_objects=max_objects,
        max_samples=n_samples,
    )
    
    subset = []
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        img_info = dataset.coco.loadImgs(sample['image_id'])[0]
        
        subset.append({
            'image_path': f"images/train2017/{img_info['file_name']}",
            'caption': sample['caption'],
            'categories': sample['categories'],
            'n_objects': sample['n_objects'],
        })
    
    with open(output_path, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"Saved {len(subset)} samples to {output_path}")


if __name__ == '__main__':
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco_root', type=str, default='~/Datasets/coco',
                    help='Path to COCO dataset')
    ap.add_argument('--action', choices=['stats', 'subset', 'test'], default='stats')
    ap.add_argument('--n_samples', type=int, default=1000)
    args = ap.parse_args()
    
    coco_root = os.path.expanduser(args.coco_root)
    
    if args.action == 'stats':
        # Show dataset statistics
        try:
            dataset = COCOScenes(coco_root, max_samples=10000)
            stats = dataset.get_stats()
            print(f"\nDataset stats:")
            print(f"  Total: {stats['total']}")
            print(f"  By object count: {stats['by_object_count']}")
        except Exception as e:
            print(f"Error: {e}")
            print("\nTo use COCO, download it first:")
            print("  mkdir -p ~/Datasets/coco")
            print("  cd ~/Datasets/coco")
            print("  wget http://images.cocodataset.org/zips/train2017.zip")
            print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            print("  unzip train2017.zip")
            print("  unzip annotations_trainval2017.zip")
    
    elif args.action == 'subset':
        # Create subset for quick testing
        create_coco_subset(
            coco_root=coco_root,
            output_path='coco_subset.json',
            n_samples=args.n_samples,
        )
    
    elif args.action == 'test':
        # Quick test
        try:
            dataset = COCOScenes(coco_root, max_samples=10)
            sample = dataset[0]
            print(f"\nSample:")
            print(f"  Image shape: {sample.image.shape}")
            print(f"  Caption: {sample.caption}")
            print(f"  Categories: {sample.categories}")
            print(f"  N objects: {sample.n_objects}")
        except Exception as e:
            print(f"Error: {e}")
