#!/usr/bin/env python3
"""
Batch Video Processor for Knowledge Graph Construction

Process multiple educational videos in parallel to extract
temporal patterns and build a comprehensive knowledge graph.

Target: 150+ videos → 3000+ knowledge patterns
"""

import sys
sys.path.insert(0, '..')

import os
import glob
import json
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

# Fix CUDA multiprocessing issue
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available, using synthetic video processing")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class VideoPattern:
    """A pattern extracted from video."""
    pattern_id: str
    video_source: str
    timestamp: float
    pattern_type: str  # 'motion', 'appearance', 'disappearance', 'transition'
    before_state: Optional[np.ndarray]
    after_state: Optional[np.ndarray]
    change_magnitude: float
    description: str


@dataclass
class KnowledgePattern:
    """Higher-level knowledge pattern."""
    pattern_id: str
    pattern_type: str  # 'atomic', 'rule', 'principle'
    domain: str  # 'physics', 'biology', 'chemistry', etc.
    description: str
    evidence_count: int
    confidence: float
    embedding: Optional[np.ndarray] = None


class VideoPatternExtractor:
    """Extract temporal patterns from video frames."""
    
    def __init__(self, feature_dim=64, change_threshold=0.1):
        self.feature_dim = feature_dim
        self.change_threshold = change_threshold
        
        # Simple feature extractor (can be replaced with brain.visual)
        self.feature_extractor = self._create_feature_extractor()
    
    def _create_feature_extractor(self):
        """Create a simple CNN feature extractor."""
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.feature_dim)
        )
        return model.to(DEVICE)
    
    def extract_features(self, frame):
        """Extract features from a single frame."""
        if isinstance(frame, np.ndarray):
            # Convert to tensor
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            
            # HWC -> CHW
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = np.transpose(frame, (2, 0, 1))
            
            frame = torch.tensor(frame, device=DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor(frame)
            features = F.normalize(features, dim=-1)
        
        return features.squeeze(0)
    
    def process_video(self, video_path: str, sample_interval: int = 30) -> List[VideoPattern]:
        """
        Process a video and extract temporal patterns.
        
        Args:
            video_path: Path to video file
            sample_interval: Sample every N frames
        
        Returns:
            List of VideoPattern objects
        """
        patterns = []
        
        if HAS_CV2:
            patterns = self._process_with_cv2(video_path, sample_interval)
        else:
            patterns = self._process_synthetic(video_path)
        
        return patterns
    
    def _process_with_cv2(self, video_path: str, sample_interval: int) -> List[VideoPattern]:
        """Process video using OpenCV."""
        patterns = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            return patterns
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prev_features = None
        frame_idx = 0
        pattern_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # Resize frame
                frame_resized = cv2.resize(frame, (56, 56))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = self.extract_features(frame_rgb)
                
                if prev_features is not None:
                    # Compute change
                    change = 1.0 - F.cosine_similarity(
                        prev_features.unsqueeze(0),
                        features.unsqueeze(0)
                    ).item()
                    
                    if change > self.change_threshold:
                        pattern = VideoPattern(
                            pattern_id=f"{Path(video_path).stem}_{pattern_count}",
                            video_source=video_path,
                            timestamp=frame_idx / fps,
                            pattern_type='transition',
                            before_state=prev_features.cpu().numpy(),
                            after_state=features.cpu().numpy(),
                            change_magnitude=change,
                            description=f"Transition at {frame_idx/fps:.1f}s (change={change:.3f})"
                        )
                        patterns.append(pattern)
                        pattern_count += 1
                
                prev_features = features
            
            frame_idx += 1
        
        cap.release()
        return patterns
    
    def _process_synthetic(self, video_path: str) -> List[VideoPattern]:
        """Generate synthetic patterns when cv2 not available."""
        patterns = []
        
        # Generate synthetic patterns based on video name
        video_name = Path(video_path).stem.lower()
        
        # Determine domain from path or name
        if 'physics' in video_path.lower() or any(w in video_name for w in ['motion', 'force', 'energy']):
            domain = 'physics'
            pattern_types = ['collision', 'acceleration', 'equilibrium']
        elif 'biology' in video_path.lower() or any(w in video_name for w in ['cell', 'organism', 'life']):
            domain = 'biology'
            pattern_types = ['growth', 'division', 'metabolism']
        elif 'chemistry' in video_path.lower() or any(w in video_name for w in ['reaction', 'molecule', 'atom']):
            domain = 'chemistry'
            pattern_types = ['reaction', 'bonding', 'catalyst']
        else:
            domain = 'general'
            pattern_types = ['change', 'sequence', 'pattern']
        
        # Generate 10-20 patterns per video
        n_patterns = np.random.randint(10, 21)
        
        for i in range(n_patterns):
            pattern = VideoPattern(
                pattern_id=f"{Path(video_path).stem}_{i}",
                video_source=video_path,
                timestamp=float(i * 30),  # Every 30 seconds
                pattern_type=np.random.choice(pattern_types),
                before_state=np.random.randn(self.feature_dim).astype(np.float32),
                after_state=np.random.randn(self.feature_dim).astype(np.float32),
                change_magnitude=np.random.uniform(0.1, 0.8),
                description=f"{domain} pattern {i}: {np.random.choice(pattern_types)}"
            )
            patterns.append(pattern)
        
        return patterns


class HierarchicalKnowledgeGraph:
    """
    Knowledge graph with hierarchical organization:
    Atomic patterns → Rules → Principles
    """
    
    def __init__(self, db_path: str = None):
        self.patterns: Dict[str, KnowledgePattern] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (source, target, relation)
        self.db_path = db_path
    
    def add_video_patterns(self, video_patterns: List[VideoPattern], domain: str = 'general'):
        """Convert video patterns to knowledge patterns."""
        for vp in video_patterns:
            # Create atomic pattern
            kp = KnowledgePattern(
                pattern_id=vp.pattern_id,
                pattern_type='atomic',
                domain=domain,
                description=vp.description,
                evidence_count=1,
                confidence=min(1.0, vp.change_magnitude * 2),
                embedding=vp.after_state
            )
            self.patterns[kp.pattern_id] = kp
    
    def abstract_to_rules(self, similarity_threshold: float = 0.7):
        """Group similar atomic patterns into rules."""
        atomic_patterns = [p for p in self.patterns.values() if p.pattern_type == 'atomic']
        
        if len(atomic_patterns) < 2:
            return 0
        
        # Cluster similar patterns
        embeddings = []
        pattern_ids = []
        
        for p in atomic_patterns:
            if p.embedding is not None:
                embeddings.append(p.embedding)
                pattern_ids.append(p.pattern_id)
        
        if len(embeddings) < 2:
            return 0
        
        embeddings = np.array(embeddings)
        
        # Simple clustering: find groups with high similarity
        n_rules = 0
        used = set()
        
        for i in range(len(embeddings)):
            if pattern_ids[i] in used:
                continue
            
            # Find similar patterns
            similarities = np.dot(embeddings, embeddings[i])
            similar_idx = np.where(similarities > similarity_threshold)[0]
            
            if len(similar_idx) >= 2:
                # Create rule
                rule_id = f"rule_{n_rules}"
                member_ids = [pattern_ids[j] for j in similar_idx]
                
                # Average embedding
                avg_embedding = embeddings[similar_idx].mean(axis=0)
                
                rule = KnowledgePattern(
                    pattern_id=rule_id,
                    pattern_type='rule',
                    domain=self.patterns[member_ids[0]].domain,
                    description=f"Rule grouping {len(member_ids)} similar patterns",
                    evidence_count=len(member_ids),
                    confidence=float(similarities[similar_idx].mean()),
                    embedding=avg_embedding
                )
                
                self.patterns[rule_id] = rule
                
                # Add edges
                for mid in member_ids:
                    self.edges.append((mid, rule_id, 'supports'))
                    used.add(mid)
                
                n_rules += 1
        
        return n_rules
    
    def abstract_to_principles(self, min_rules: int = 3):
        """Abstract rules into higher-level principles."""
        rules = [p for p in self.patterns.values() if p.pattern_type == 'rule']
        
        if len(rules) < min_rules:
            return 0
        
        # Group by domain
        domains = {}
        for rule in rules:
            if rule.domain not in domains:
                domains[rule.domain] = []
            domains[rule.domain].append(rule)
        
        n_principles = 0
        
        for domain, domain_rules in domains.items():
            if len(domain_rules) >= min_rules:
                # Create principle
                principle_id = f"principle_{domain}_{n_principles}"
                
                # Combine embeddings
                embeddings = [r.embedding for r in domain_rules if r.embedding is not None]
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                else:
                    avg_embedding = None
                
                principle = KnowledgePattern(
                    pattern_id=principle_id,
                    pattern_type='principle',
                    domain=domain,
                    description=f"{domain.title()} principle from {len(domain_rules)} rules",
                    evidence_count=sum(r.evidence_count for r in domain_rules),
                    confidence=float(np.mean([r.confidence for r in domain_rules])),
                    embedding=avg_embedding
                )
                
                self.patterns[principle_id] = principle
                
                # Add edges
                for rule in domain_rules:
                    self.edges.append((rule.pattern_id, principle_id, 'abstracts_to'))
                
                n_principles += 1
        
        return n_principles
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the knowledge graph."""
        stats = {
            'total_patterns': len(self.patterns),
            'atomic': sum(1 for p in self.patterns.values() if p.pattern_type == 'atomic'),
            'rules': sum(1 for p in self.patterns.values() if p.pattern_type == 'rule'),
            'principles': sum(1 for p in self.patterns.values() if p.pattern_type == 'principle'),
            'edges': len(self.edges)
        }
        
        # Domain breakdown
        domains = {}
        for p in self.patterns.values():
            if p.domain not in domains:
                domains[p.domain] = 0
            domains[p.domain] += 1
        stats['domains'] = domains
        
        return stats
    
    def save(self, output_path: str):
        """Save knowledge graph to file."""
        save_data = {
            'patterns': {
                pid: {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type,
                    'domain': p.domain,
                    'description': p.description,
                    'evidence_count': p.evidence_count,
                    'confidence': p.confidence,
                    'embedding': p.embedding.tolist() if p.embedding is not None else None
                }
                for pid, p in self.patterns.items()
            },
            'edges': self.edges,
            'stats': self.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved knowledge graph to {output_path}")
    
    @classmethod
    def load(cls, input_path: str) -> 'HierarchicalKnowledgeGraph':
        """Load knowledge graph from file."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        kg = cls()
        
        for pid, pdata in data['patterns'].items():
            embedding = np.array(pdata['embedding']) if pdata['embedding'] else None
            kg.patterns[pid] = KnowledgePattern(
                pattern_id=pdata['pattern_id'],
                pattern_type=pdata['pattern_type'],
                domain=pdata['domain'],
                description=pdata['description'],
                evidence_count=pdata['evidence_count'],
                confidence=pdata['confidence'],
                embedding=embedding
            )
        
        kg.edges = data['edges']
        return kg


def process_video_worker(args):
    """Worker function for parallel video processing."""
    video_path, feature_dim, change_threshold = args
    
    extractor = VideoPatternExtractor(
        feature_dim=feature_dim,
        change_threshold=change_threshold
    )
    
    patterns = extractor.process_video(video_path)
    
    return video_path, patterns


def batch_process_videos(
    video_dir: str,
    output_path: str,
    workers: int = 4,
    feature_dim: int = 64,
    change_threshold: float = 0.1
):
    """
    Process all videos in directory and build knowledge graph.
    
    Args:
        video_dir: Directory containing videos
        output_path: Output path for knowledge graph
        workers: Number of parallel workers
        feature_dim: Feature dimension for patterns
        change_threshold: Minimum change to detect pattern
    """
    print("=" * 60)
    print("BATCH VIDEO PROCESSOR")
    print("=" * 60)
    
    # Find all videos
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(f"{video_dir}/**/{ext}", recursive=True))
    
    if not video_files:
        print(f"No videos found in {video_dir}")
        print("Generating synthetic video patterns for demonstration...")
        
        # Create synthetic video list
        video_files = [
            f"{video_dir}/physics/motion_{i}.mp4" for i in range(50)
        ] + [
            f"{video_dir}/biology/cell_{i}.mp4" for i in range(50)
        ] + [
            f"{video_dir}/chemistry/reaction_{i}.mp4" for i in range(50)
        ]
    
    print(f"Found {len(video_files)} videos")
    
    # Prepare arguments
    args = [(v, feature_dim, change_threshold) for v in video_files]
    
    # Process videos
    all_patterns = []
    
    if workers > 1 and len(video_files) > 10:
        print(f"Processing with {workers} workers...")
        
        with mp.Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_video_worker, args),
                total=len(args),
                desc="Processing videos"
            ))
        
        for video_path, patterns in results:
            all_patterns.extend(patterns)
    else:
        print("Processing sequentially...")
        
        for video_path, fd, ct in tqdm(args, desc="Processing videos"):
            extractor = VideoPatternExtractor(feature_dim=fd, change_threshold=ct)
            patterns = extractor.process_video(video_path)
            all_patterns.extend(patterns)
    
    print(f"\nExtracted {len(all_patterns)} patterns from {len(video_files)} videos")
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    kg = HierarchicalKnowledgeGraph()
    
    # Add patterns by domain
    for pattern in all_patterns:
        # Infer domain from video path
        video_path = pattern.video_source.lower()
        if 'physics' in video_path:
            domain = 'physics'
        elif 'biology' in video_path:
            domain = 'biology'
        elif 'chemistry' in video_path:
            domain = 'chemistry'
        else:
            domain = 'general'
        
        kp = KnowledgePattern(
            pattern_id=pattern.pattern_id,
            pattern_type='atomic',
            domain=domain,
            description=pattern.description,
            evidence_count=1,
            confidence=min(1.0, pattern.change_magnitude * 2),
            embedding=pattern.after_state
        )
        kg.patterns[kp.pattern_id] = kp
    
    # Abstract to higher levels
    print("Abstracting to rules...")
    n_rules = kg.abstract_to_rules()
    print(f"Created {n_rules} rules")
    
    print("Abstracting to principles...")
    n_principles = kg.abstract_to_principles()
    print(f"Created {n_principles} principles")
    
    # Save
    kg.save(output_path)
    
    # Summary
    stats = kg.get_stats()
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("=" * 60)
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"  Atomic: {stats['atomic']}")
    print(f"  Rules: {stats['rules']}")
    print(f"  Principles: {stats['principles']}")
    print(f"Edges: {stats['edges']}")
    print(f"Domains: {stats['domains']}")
    print(f"Output: {output_path}")
    
    return kg


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch video processor')
    parser.add_argument('--input', type=str, default='data/videos',
                        help='Input directory containing videos')
    parser.add_argument('--output', type=str, default='knowledge_graph_full.pkl',
                        help='Output path for knowledge graph')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Feature dimension')
    parser.add_argument('--change_threshold', type=float, default=0.1,
                        help='Minimum change to detect pattern')
    
    args = parser.parse_args()
    
    batch_process_videos(
        video_dir=args.input,
        output_path=args.output,
        workers=args.workers,
        feature_dim=args.feature_dim,
        change_threshold=args.change_threshold
    )
