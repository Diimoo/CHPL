#!/usr/bin/env python3
"""
Hierarchical Knowledge Graph for Pattern Storage

Organize 3000+ learned patterns hierarchically:
- Atomic facts: "ball falls down"
- Rules: "all objects fall due to gravity"  
- Laws: "F = ma"

Enables:
- Efficient storage (not RAM-limited)
- Inference (combine facts → conclusions)
- Explanation (why does X happen?)
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import sqlite3
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pickle

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed, using simple graph")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Pattern:
    """A knowledge pattern."""
    id: int
    pattern_type: str  # atomic, rule, law
    domain: str  # physics, biology, chemistry, general
    description: str
    embedding: Optional[torch.Tensor] = None
    confidence: float = 1.0
    occurrences: int = 1
    parent_ids: List[int] = field(default_factory=list)
    child_ids: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class KnowledgeGraph:
    """
    Hierarchical knowledge graph with disk storage.
    
    Supports 3000+ patterns without memory issues.
    """
    
    def __init__(self, db_path: str = 'knowledge.db'):
        self.db_path = db_path
        self.patterns: Dict[int, Pattern] = {}
        self.next_id = 0
        
        # In-memory graph for fast traversal
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            self.edges = defaultdict(list)  # parent → [children]
            self.reverse_edges = defaultdict(list)  # child → [parents]
        
        # Domain indices
        self.domain_index: Dict[str, Set[int]] = defaultdict(set)
        self.type_index: Dict[str, Set[int]] = defaultdict(set)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                domain TEXT,
                description TEXT,
                embedding BLOB,
                confidence REAL,
                occurrences INTEGER,
                metadata TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                parent_id INTEGER,
                child_id INTEGER,
                edge_type TEXT,
                FOREIGN KEY (parent_id) REFERENCES patterns(id),
                FOREIGN KEY (child_id) REFERENCES patterns(id)
            )
        ''')
        
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_domain ON patterns(domain)
        ''')
        
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_type ON patterns(pattern_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_pattern(
        self,
        pattern_type: str,
        domain: str,
        description: str,
        embedding: Optional[torch.Tensor] = None,
        parent_ids: Optional[List[int]] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Add pattern to knowledge graph.
        
        Returns: pattern ID
        """
        pattern_id = self.next_id
        self.next_id += 1
        
        pattern = Pattern(
            id=pattern_id,
            pattern_type=pattern_type,
            domain=domain,
            description=description,
            embedding=embedding,
            confidence=confidence,
            parent_ids=parent_ids or [],
            metadata=metadata or {},
        )
        
        self.patterns[pattern_id] = pattern
        
        # Update indices
        self.domain_index[domain].add(pattern_id)
        self.type_index[pattern_type].add(pattern_id)
        
        # Update graph
        if HAS_NETWORKX:
            self.graph.add_node(pattern_id)
        
        if parent_ids:
            for parent_id in parent_ids:
                if parent_id in self.patterns:
                    self.patterns[parent_id].child_ids.append(pattern_id)
                    
                    if HAS_NETWORKX:
                        self.graph.add_edge(parent_id, pattern_id)
                    else:
                        self.edges[parent_id].append(pattern_id)
                        self.reverse_edges[pattern_id].append(parent_id)
        
        # Persist to database
        self._save_pattern(pattern)
        
        return pattern_id
    
    def _save_pattern(self, pattern: Pattern):
        """Save pattern to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        embedding_blob = None
        if pattern.embedding is not None:
            embedding_blob = pickle.dumps(pattern.embedding.cpu())
        
        c.execute('''
            INSERT OR REPLACE INTO patterns 
            (id, pattern_type, domain, description, embedding, confidence, occurrences, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.id,
            pattern.pattern_type,
            pattern.domain,
            pattern.description,
            embedding_blob,
            pattern.confidence,
            pattern.occurrences,
            json.dumps(pattern.metadata),
        ))
        
        # Save edges
        for parent_id in pattern.parent_ids:
            c.execute('''
                INSERT INTO edges (parent_id, child_id, edge_type)
                VALUES (?, ?, 'derives')
            ''', (parent_id, pattern.id))
        
        conn.commit()
        conn.close()
    
    def find_similar(
        self,
        query_embedding: torch.Tensor,
        domain: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Find patterns similar to query embedding.
        """
        candidates = []
        
        if domain:
            pattern_ids = self.domain_index.get(domain, set())
        else:
            pattern_ids = set(self.patterns.keys())
        
        for pid in pattern_ids:
            pattern = self.patterns[pid]
            if pattern.embedding is not None:
                sim = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    pattern.embedding.unsqueeze(0)
                ).item()
                candidates.append((pid, sim))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def find_by_description(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 10,
    ) -> List[int]:
        """
        Find patterns by text search in description.
        """
        query_words = set(query.lower().split())
        
        if domain:
            pattern_ids = self.domain_index.get(domain, set())
        else:
            pattern_ids = set(self.patterns.keys())
        
        scores = []
        for pid in pattern_ids:
            pattern = self.patterns[pid]
            desc_words = set(pattern.description.lower().split())
            overlap = len(query_words & desc_words)
            if overlap > 0:
                scores.append((pid, overlap))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, score in scores[:top_k]]
    
    def get_children(self, pattern_id: int) -> List[int]:
        """Get child patterns."""
        if pattern_id not in self.patterns:
            return []
        return self.patterns[pattern_id].child_ids
    
    def get_parents(self, pattern_id: int) -> List[int]:
        """Get parent patterns."""
        if pattern_id not in self.patterns:
            return []
        return self.patterns[pattern_id].parent_ids
    
    def explain(self, pattern_id: int, depth: int = 3) -> Dict:
        """
        Explain why a pattern holds.
        
        Traverses graph to find supporting evidence.
        """
        if pattern_id not in self.patterns:
            return {'error': 'Pattern not found'}
        
        pattern = self.patterns[pattern_id]
        
        explanation = {
            'pattern': pattern.description,
            'type': pattern.pattern_type,
            'domain': pattern.domain,
            'confidence': pattern.confidence,
            'evidence': [],
        }
        
        if depth > 0:
            for parent_id in pattern.parent_ids:
                parent_explanation = self.explain(parent_id, depth - 1)
                explanation['evidence'].append(parent_explanation)
        
        return explanation
    
    def abstract_patterns(
        self,
        pattern_ids: List[int],
        domain: str,
    ) -> Optional[int]:
        """
        Find common structure across patterns to create higher-level rule.
        
        Example:
        - Atomic: "red ball falls", "blue ball falls"
        - Rule: "all objects fall"
        """
        if len(pattern_ids) < 2:
            return None
        
        patterns = [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        
        if len(patterns) < 2:
            return None
        
        # Find common words in descriptions
        word_sets = [set(p.description.lower().split()) for p in patterns]
        common_words = word_sets[0]
        for ws in word_sets[1:]:
            common_words &= ws
        
        if len(common_words) < 2:
            return None
        
        # Create rule description
        rule_desc = f"Pattern: {' '.join(sorted(common_words))} (from {len(patterns)} examples)"
        
        # Average embeddings if available
        embeddings = [p.embedding for p in patterns if p.embedding is not None]
        if embeddings:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
        else:
            avg_embedding = None
        
        # Create rule pattern
        rule_id = self.add_pattern(
            pattern_type='rule',
            domain=domain,
            description=rule_desc,
            embedding=avg_embedding,
            parent_ids=pattern_ids,
            confidence=len(patterns) / 10.0,  # Confidence based on evidence
        )
        
        return rule_id
    
    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        stats = {
            'total_patterns': len(self.patterns),
            'by_type': {t: len(ids) for t, ids in self.type_index.items()},
            'by_domain': {d: len(ids) for d, ids in self.domain_index.items()},
        }
        
        if HAS_NETWORKX:
            stats['edges'] = self.graph.number_of_edges()
        else:
            stats['edges'] = sum(len(children) for children in self.edges.values())
        
        return stats
    
    def save(self, path: str):
        """Save in-memory state to file."""
        data = {
            'next_id': self.next_id,
            'patterns': {
                pid: {
                    'id': p.id,
                    'pattern_type': p.pattern_type,
                    'domain': p.domain,
                    'description': p.description,
                    'embedding': p.embedding.cpu() if p.embedding is not None else None,
                    'confidence': p.confidence,
                    'occurrences': p.occurrences,
                    'parent_ids': p.parent_ids,
                    'child_ids': p.child_ids,
                    'metadata': p.metadata,
                }
                for pid, p in self.patterns.items()
            },
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.patterns)} patterns to {path}")
    
    def load(self, path: str):
        """Load state from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.next_id = data['next_id']
        
        for pid, pdata in data['patterns'].items():
            pattern = Pattern(
                id=pdata['id'],
                pattern_type=pdata['pattern_type'],
                domain=pdata['domain'],
                description=pdata['description'],
                embedding=pdata['embedding'],
                confidence=pdata['confidence'],
                occurrences=pdata['occurrences'],
                parent_ids=pdata['parent_ids'],
                child_ids=pdata['child_ids'],
                metadata=pdata['metadata'],
            )
            self.patterns[int(pid)] = pattern
            self.domain_index[pattern.domain].add(pattern.id)
            self.type_index[pattern.pattern_type].add(pattern.id)
        
        print(f"Loaded {len(self.patterns)} patterns from {path}")


class ScalableVideoLearner:
    """
    Learn from 400+ videos, store in knowledge graph.
    
    Processes videos in batches, extracts patterns,
    organizes hierarchically.
    """
    
    def __init__(self, brain, db_path: str = 'video_knowledge.db'):
        self.brain = brain
        self.knowledge = KnowledgeGraph(db_path)
        
        # Processing stats
        self.videos_processed = 0
        self.total_frames = 0
        self.patterns_extracted = 0
    
    def process_video(
        self,
        video_path: str,
        domain: str,
        sample_rate: int = 10,
    ) -> List[int]:
        """
        Process single video, extract patterns.
        
        Args:
            video_path: Path to video file
            domain: Knowledge domain
            sample_rate: Sample every N frames
        
        Returns:
            List of pattern IDs created
        """
        try:
            import cv2
        except ImportError:
            print("cv2 not available, using synthetic frames")
            return self._process_synthetic(domain)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            return []
        
        pattern_ids = []
        frame_count = 0
        previous_activation = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % sample_rate != 0:
                continue
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)
            
            # Resize
            frame_tensor = F.interpolate(
                frame_tensor.unsqueeze(0),
                size=(56, 56),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).to(DEVICE)
            
            # Get activation
            activation = self.brain.get_scene_activation(frame_tensor)
            
            if previous_activation is not None:
                # Detect change
                change = activation - previous_activation
                change_magnitude = torch.norm(change).item()
                
                if change_magnitude > 0.1:  # Significant change
                    # Create pattern
                    pattern_id = self.knowledge.add_pattern(
                        pattern_type='atomic',
                        domain=domain,
                        description=f"Visual change at frame {frame_count}",
                        embedding=change.cpu(),
                        confidence=min(change_magnitude, 1.0),
                        metadata={'frame': frame_count, 'video': video_path},
                    )
                    pattern_ids.append(pattern_id)
                    self.patterns_extracted += 1
            
            previous_activation = activation
            self.total_frames += 1
        
        cap.release()
        self.videos_processed += 1
        
        return pattern_ids
    
    def _process_synthetic(self, domain: str) -> List[int]:
        """Process synthetic video when cv2 not available."""
        from synthetic_environment import create_stimulus_on_canvas
        
        pattern_ids = []
        
        # Create synthetic sequence
        for i in range(10):
            canvas = np.ones((56, 56, 3), dtype=np.float32)
            
            obj = create_stimulus_on_canvas(
                shape='circle',
                color='red',
                size='small',
                canvas_size=56,
                center=(10 + i * 4, 28),
            )
            
            is_obj = np.any(obj < 0.95, axis=-1)
            canvas[is_obj] = obj[is_obj]
            
            frame_tensor = torch.from_numpy(canvas).float().permute(2, 0, 1).to(DEVICE)
            activation = self.brain.get_scene_activation(frame_tensor)
            
            if i > 0:
                pattern_id = self.knowledge.add_pattern(
                    pattern_type='atomic',
                    domain=domain,
                    description=f"Object moved right (frame {i})",
                    embedding=activation.cpu(),
                )
                pattern_ids.append(pattern_id)
                self.patterns_extracted += 1
        
        self.videos_processed += 1
        self.total_frames += 10
        
        return pattern_ids
    
    def process_curriculum(
        self,
        video_dir: str,
        domain: str,
    ) -> Dict:
        """
        Process entire curriculum of videos.
        """
        video_dir = Path(video_dir)
        
        if not video_dir.exists():
            print(f"Directory not found: {video_dir}")
            # Use synthetic videos
            pattern_ids = []
            for i in range(10):
                pids = self._process_synthetic(domain)
                pattern_ids.extend(pids)
            return {'videos': 10, 'patterns': len(pattern_ids)}
        
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
        
        all_pattern_ids = []
        
        for video_file in video_files:
            print(f"  Processing: {video_file.name}")
            pattern_ids = self.process_video(str(video_file), domain)
            all_pattern_ids.extend(pattern_ids)
        
        # Abstract to rules
        if len(all_pattern_ids) > 5:
            self.abstract_domain_patterns(all_pattern_ids, domain)
        
        return {
            'videos': len(video_files),
            'patterns': len(all_pattern_ids),
        }
    
    def abstract_domain_patterns(
        self,
        pattern_ids: List[int],
        domain: str,
    ):
        """Create rules from atomic patterns."""
        # Cluster similar patterns
        # For now: simple abstraction every 10 patterns
        
        for i in range(0, len(pattern_ids), 10):
            chunk = pattern_ids[i:i+10]
            if len(chunk) >= 3:
                rule_id = self.knowledge.abstract_patterns(chunk, domain)
                if rule_id:
                    print(f"    Created rule {rule_id} from {len(chunk)} patterns")


def run_knowledge_graph_demo():
    """Demo the knowledge graph system."""
    print("=" * 70)
    print("KNOWLEDGE GRAPH DEMO")
    print("Hierarchical storage for 3000+ patterns")
    print("=" * 70)
    
    # Create knowledge graph
    kg = KnowledgeGraph('demo_knowledge.db')
    
    # Add some physics patterns
    print("\nAdding physics patterns...")
    
    physics_patterns = [
        "ball falls down due to gravity",
        "ball bounces after hitting ground",
        "objects accelerate as they fall",
        "heavier objects fall at same rate as light ones",
        "momentum is conserved in collisions",
    ]
    
    pattern_ids = []
    for desc in physics_patterns:
        pid = kg.add_pattern(
            pattern_type='atomic',
            domain='physics',
            description=desc,
            embedding=torch.randn(64),
        )
        pattern_ids.append(pid)
        print(f"  Added pattern {pid}: {desc}")
    
    # Create rule from patterns
    print("\nAbstracting to rules...")
    rule_id = kg.abstract_patterns(pattern_ids[:3], 'physics')
    if rule_id:
        print(f"  Created rule {rule_id}")
    
    # Query patterns
    print("\nQuerying patterns...")
    results = kg.find_by_description("fall", domain='physics')
    print(f"  Patterns containing 'fall': {results}")
    
    # Get stats
    stats = kg.get_stats()
    print(f"\nKnowledge graph stats:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By domain: {stats['by_domain']}")
    
    # Explain
    if pattern_ids:
        explanation = kg.explain(pattern_ids[0])
        print(f"\nExplanation for pattern 0:")
        print(f"  {explanation}")
    
    print("\nDemo complete!")
    
    return kg


if __name__ == "__main__":
    kg = run_knowledge_graph_demo()
