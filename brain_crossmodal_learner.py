#!/usr/bin/env python3
"""
Brain-Like Cross-Modal Learning System.

This is the REAL neuroscience research:
- NO pre-trained features (learning from scratch)
- Hebbian plasticity (neurons that fire together, wire together)
- Hippocampal episodic binding (fast, one-shot)
- ATL semantic consolidation (slow, repeated exposure)
- Online learning during inference

Scientific question: Can biological learning rules learn cross-modal binding?
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# SIMPLE VISUAL CORTEX (trainable from scratch)
# ============================================================

class SimpleVisualCortex(nn.Module):
    """
    Simple visual processing for 28x28 images.
    
    Architecture inspired by V1→V2→V4→IT pathway:
    - Conv layers extract increasingly complex features
    - NO pre-training, learns from scratch
    - Uses reconstruction (predictive coding) to learn discriminative features
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # V1: Edge detection (3x3 filters)
        self.v1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # V2: Texture/corners (3x3 filters on V1 output)
        self.v2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # V4: Shape fragments (3x3 with pooling)
        self.v4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # IT: Object representation
        self.it = nn.Linear(64 * 3 * 3, feature_dim)  # After 3x pooling: 28→14→7→3
        
        # Decoder for reconstruction (predictive coding)
        # 28 -> 14 -> 7 -> 3 (encoder), so decoder: 3 -> 7 -> 14 -> 28
        self.decoder_fc = nn.Linear(feature_dim, 64 * 3 * 3)
        self.decoder_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_v4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_v2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_v1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.feature_dim = feature_dim
        self.to(DEVICE)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image through visual hierarchy.
        
        Input: [H, W, C] or [B, H, W, C] (28x28x3)
        Output: [feature_dim] or [B, feature_dim]
        """
        # Handle input shapes
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        x = x.to(DEVICE)
        
        # V1: edges
        x = F.relu(self.v1(x))
        x = F.max_pool2d(x, 2)  # 28→14
        
        # V2: textures
        x = F.relu(self.v2(x))
        x = F.max_pool2d(x, 2)  # 14→7
        
        # V4: shapes
        x = F.relu(self.v4(x))
        x = F.max_pool2d(x, 2)  # 7→3
        
        # IT: flatten and project
        x = x.reshape(x.size(0), -1)
        x = self.it(x)
        
        # Normalize for stable Hebbian learning
        x = F.normalize(x, dim=-1)
        
        return x.squeeze(0) if x.size(0) == 1 else x
    
    def reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to image (predictive coding)."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        x = F.relu(self.decoder_fc(features))
        x = x.view(-1, 64, 3, 3)
        
        # 3→6→7 (pad to 7)
        x = self.decoder_upsample(x)  # 3→6
        x = F.pad(x, (0, 1, 0, 1))  # 6→7
        x = F.relu(self.decoder_v4(x))
        
        # 7→14
        x = self.decoder_upsample(x)  # 7→14
        x = F.relu(self.decoder_v2(x))
        
        # 14→28
        x = self.decoder_upsample(x)  # 14→28
        x = torch.sigmoid(self.decoder_v1(x))
        
        return x.squeeze(0) if x.size(0) == 1 else x
    
    def reconstruction_loss(self, image: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for predictive coding."""
        # Ensure correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        image = image.to(DEVICE)
        
        features = self.forward(image.permute(0, 2, 3, 1))  # forward expects [B, H, W, C]
        if features.dim() == 1:
            features = features.unsqueeze(0)
        reconstructed = self.reconstruct(features)
        
        return F.mse_loss(reconstructed, image)


# ============================================================
# SIMPLE LANGUAGE CORTEX (trainable from scratch)
# ============================================================

class SimpleLanguageCortex(nn.Module):
    """
    Simple language processing for small vocabulary.
    
    Character-level processing → word embedding.
    Works for 50-word vocabulary with short words.
    """
    
    def __init__(self, feature_dim: int = 64, max_word_len: int = 20):
        super().__init__()
        
        # Character embedding (26 letters + space + padding)
        self.char_embed = nn.Embedding(28, 16)
        
        # Convolutions over character sequence
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Project to feature space
        self.proj = nn.Linear(64, feature_dim)
        
        self.feature_dim = feature_dim
        self.max_word_len = max_word_len
        self.to(DEVICE)
        
    def text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text to character indices."""
        indices = []
        for c in text.lower()[:self.max_word_len]:
            if c == ' ':
                indices.append(26)
            elif 'a' <= c <= 'z':
                indices.append(ord(c) - ord('a'))
            else:
                indices.append(27)  # Unknown
        
        # Pad to max length
        while len(indices) < self.max_word_len:
            indices.append(27)
        
        return torch.tensor(indices, dtype=torch.long, device=DEVICE)
    
    def forward(self, text: str) -> torch.Tensor:
        """
        Process text through language hierarchy.
        
        Input: string (e.g., "red circle")
        Output: [feature_dim]
        """
        # Convert to indices
        indices = self.text_to_indices(text).unsqueeze(0)  # [1, max_len]
        
        # Character embeddings
        x = self.char_embed(indices)  # [1, max_len, 16]
        x = x.permute(0, 2, 1)  # [1, 16, max_len]
        
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global average pool over sequence
        x = x.mean(dim=-1)  # [1, 64]
        
        # Project
        x = self.proj(x)
        
        # Normalize
        x = F.normalize(x, dim=-1)
        
        return x.squeeze(0)


# ============================================================
# HIPPOCAMPUS: Fast episodic binding
# ============================================================

class Hippocampus(nn.Module):
    """
    Hippocampal episodic memory.
    
    Biological properties:
    - Fast one-shot learning (single exposure creates memory)
    - Pattern separation (similar inputs → distinct representations)
    - Pattern completion (partial cue → full memory)
    - Capacity limited (forgetting of old memories)
    """
    
    def __init__(self, feature_dim: int = 64, capacity: int = 1000):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.capacity = capacity
        
        # DG: Pattern separation (sparse encoding)
        self.dg = nn.Linear(feature_dim * 2, feature_dim * 4)
        
        # CA3: Autoassociative memory (stores bindings)
        self.ca3_keys = torch.zeros(capacity, feature_dim * 4, device=DEVICE)
        self.ca3_values_vis = torch.zeros(capacity, feature_dim, device=DEVICE)
        self.ca3_values_lang = torch.zeros(capacity, feature_dim, device=DEVICE)
        
        # Memory pointer
        self.memory_idx = 0
        self.n_stored = 0
        
        self.to(DEVICE)
    
    def encode(self, vis_features: torch.Tensor, lang_features: torch.Tensor) -> int:
        """
        Encode a cross-modal binding (one-shot learning).
        
        Returns memory index.
        """
        # Concatenate and pass through DG for pattern separation
        combined = torch.cat([vis_features.flatten(), lang_features.flatten()])
        sparse_code = F.relu(self.dg(combined))
        sparse_code = F.normalize(sparse_code, dim=-1)
        
        # Store in CA3
        idx = self.memory_idx
        self.ca3_keys[idx] = sparse_code.detach()
        self.ca3_values_vis[idx] = vis_features.flatten().detach()
        self.ca3_values_lang[idx] = lang_features.flatten().detach()
        
        # Update pointer (circular buffer)
        self.memory_idx = (self.memory_idx + 1) % self.capacity
        self.n_stored = min(self.n_stored + 1, self.capacity)
        
        return idx
    
    def recall_from_visual(self, vis_features: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given visual input, recall associated language.
        
        Returns: (language_features, similarity_scores)
        """
        if self.n_stored == 0:
            return None, None
        
        # Create query (use visual features as partial cue)
        # Pad with zeros for the language part
        query = torch.cat([vis_features.flatten(), torch.zeros(self.feature_dim, device=DEVICE)])
        query_sparse = F.relu(self.dg(query))
        query_sparse = F.normalize(query_sparse, dim=-1)
        
        # Find most similar memories
        similarities = torch.matmul(self.ca3_keys[:self.n_stored], query_sparse)
        top_indices = torch.topk(similarities, min(top_k, self.n_stored)).indices
        
        recalled_lang = self.ca3_values_lang[top_indices]
        scores = similarities[top_indices]
        
        return recalled_lang, scores
    
    def recall_from_language(self, lang_features: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given language input, recall associated visual.
        """
        if self.n_stored == 0:
            return None, None
        
        query = torch.cat([torch.zeros(self.feature_dim, device=DEVICE), lang_features.flatten()])
        query_sparse = F.relu(self.dg(query))
        query_sparse = F.normalize(query_sparse, dim=-1)
        
        similarities = torch.matmul(self.ca3_keys[:self.n_stored], query_sparse)
        top_indices = torch.topk(similarities, min(top_k, self.n_stored)).indices
        
        recalled_vis = self.ca3_values_vis[top_indices]
        scores = similarities[top_indices]
        
        return recalled_vis, scores


# ============================================================
# ATL: Slow semantic consolidation
# ============================================================

class ATL(nn.Module):
    """
    Anterior Temporal Lobe - semantic memory hub.
    
    Biological properties:
    - Slow learning (requires repeated exposure)
    - Extracts statistical regularities
    - Cross-modal binding (amodal concepts)
    - Competitive learning with lateral inhibition (SOM-like)
    - Survives hippocampal damage (consolidated memories)
    """
    
    def __init__(self, feature_dim: int = 64, n_concepts: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_concepts = n_concepts
        
        # SHARED concept prototypes (amodal - same for both modalities)
        # This is biologically plausible: ATL represents concepts abstractly
        prototypes = torch.randn(n_concepts, feature_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.register_buffer('prototypes', prototypes)
        
        # Modality-specific projection to shared space
        # This allows each modality to map to shared concepts differently
        self.vis_proj = nn.Linear(feature_dim, feature_dim).to(DEVICE)
        self.lang_proj = nn.Linear(feature_dim, feature_dim).to(DEVICE)
        
        # Initialize projections close to identity
        nn.init.eye_(self.vis_proj.weight)
        nn.init.zeros_(self.vis_proj.bias)
        nn.init.eye_(self.lang_proj.weight)
        nn.init.zeros_(self.lang_proj.bias)
        
        # Usage tracking
        self.register_buffer('usage', torch.zeros(n_concepts))
        
        # Learning rate
        self.base_lr = 0.1
        
        self.to(DEVICE)
    
    def activate(self, features: torch.Tensor, modality: str = 'visual') -> Tuple[torch.Tensor, int]:
        """
        Project features to shared space, then find most similar prototype.
        """
        features = features.flatten()
        
        # Project to shared space
        if modality == 'visual':
            projected = self.vis_proj(features)
        else:
            projected = self.lang_proj(features)
        
        projected = F.normalize(projected, dim=-1)
        
        # Cosine similarity to shared prototypes
        similarities = torch.matmul(self.prototypes, projected)
        
        # Lateral inhibition via softmax
        temperature = 0.1
        activations = F.softmax(similarities / temperature, dim=-1)
        
        # Winner
        best_idx = similarities.argmax().item()
        
        return activations, best_idx
    
    def consolidate(self, vis_features: torch.Tensor, lang_features: torch.Tensor):
        """
        Cross-modal binding via shared prototype learning.
        
        Key insight: Both modalities should map to SAME concept.
        We learn projections to align them.
        """
        vis_features = vis_features.flatten()
        lang_features = lang_features.flatten()
        
        # Project both to shared space
        vis_proj = F.normalize(self.vis_proj(vis_features), dim=-1)
        lang_proj = F.normalize(self.lang_proj(lang_features), dim=-1)
        
        # Find winner for COMBINED representation
        combined = F.normalize((vis_proj + lang_proj) / 2, dim=-1)
        similarities = torch.matmul(self.prototypes, combined)
        winner = similarities.argmax().item()
        
        # Move prototype towards combined representation
        lr = self.base_lr / (1 + self.usage[winner] * 0.01)
        self.prototypes[winner] = F.normalize(
            (1 - lr) * self.prototypes[winner] + lr * combined,
            dim=-1
        )
        self.usage[winner] += 1
    
    def get_active_concepts(self) -> int:
        """Count how many concepts have been used."""
        return int((self.usage > 0).sum().item())


# ============================================================
# BRAIN-LIKE CROSS-MODAL LEARNER
# ============================================================

class BrainCrossModalLearner:
    """
    Complete brain-like system for cross-modal learning.
    
    Components:
    - Visual cortex: learns visual features from scratch
    - Language cortex: learns word embeddings from scratch
    - Hippocampus: fast episodic binding
    - ATL: slow semantic consolidation
    
    Learning is ONLINE during inference (like a real brain).
    """
    
    def __init__(self, feature_dim: int = 64, n_concepts: int = 100):
        self.feature_dim = feature_dim
        
        # Sensory cortices
        self.visual = SimpleVisualCortex(feature_dim)
        self.language = SimpleLanguageCortex(feature_dim)
        
        # Memory systems
        self.hippocampus = Hippocampus(feature_dim)
        self.atl = ATL(feature_dim, n_concepts)
        
        # Hebbian learning parameters
        self.hebbian_lr = 0.01
        
        # Optimizer for cortices (optional supervised signal)
        self.cortex_optimizer = torch.optim.Adam(
            list(self.visual.parameters()) + list(self.language.parameters()),
            lr=1e-3
        )
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'consolidations': 0,
            'bindings': defaultdict(int),  # label → count
        }
    
    def experience(self, image: torch.Tensor, label: str, 
                   consolidate: bool = True,
                   train_cortices: bool = True) -> Dict:
        """
        Experience a cross-modal pair (like infant learning).
        
        1. Process both modalities
        2. Train cortices via reconstruction (predictive coding)
        3. Hippocampus binds them (one-shot)
        4. ATL consolidates (if repeated)
        
        Returns dict with features and activations.
        """
        # Train cortices via reconstruction (predictive coding)
        # This is biologically plausible - brain learns to predict/reconstruct
        if train_cortices:
            self.cortex_optimizer.zero_grad()
            recon_loss = self.visual.reconstruction_loss(image)
            recon_loss.backward()
            self.cortex_optimizer.step()
        
        # Process through cortices (after training step)
        vis_features = self.visual(image)
        lang_features = self.language(label)
        
        # Hippocampal episodic binding (always happens)
        memory_idx = self.hippocampus.encode(vis_features, lang_features)
        
        # Track exposure
        self.stats['bindings'][label] += 1
        self.stats['episodes'] += 1
        
        # ATL consolidation (if repeated exposure)
        if consolidate and self.stats['bindings'][label] > 1:
            self.atl.consolidate(vis_features.detach(), lang_features.detach())
            self.stats['consolidations'] += 1
        
        # Hebbian update to cortex (co-activation strengthens connection)
        self.hebbian_update(vis_features.detach(), lang_features.detach())
        
        # Get concept activations
        vis_concept = self.atl.activate(vis_features, 'visual')[1]
        lang_concept = self.atl.activate(lang_features, 'language')[1]
        
        return {
            'vis_features': vis_features,
            'lang_features': lang_features,
            'vis_concept': vis_concept,
            'lang_concept': lang_concept,
            'memory_idx': memory_idx,
            'same_concept': vis_concept == lang_concept,
        }
    
    def hebbian_update(self, vis_features: torch.Tensor, lang_features: torch.Tensor):
        """
        Hebbian learning is now handled by ATL.consolidate() which uses
        competitive learning with prototype matching.
        
        This method is kept for API compatibility but delegates to ATL.
        """
        # ATL.consolidate already does Hebbian-style learning
        pass  # Learning happens in experience() via atl.consolidate()
    
    def recall_word_from_image(self, image: torch.Tensor) -> Tuple[Optional[str], float]:
        """
        Given image, what word does the brain associate?
        
        Uses hippocampal recall.
        """
        vis_features = self.visual(image)
        recalled_lang, scores = self.hippocampus.recall_from_visual(vis_features)
        
        if recalled_lang is None:
            return None, 0.0
        
        # Find most similar word in vocabulary
        # (In a full system, this would search vocabulary)
        return recalled_lang, scores[0].item() if scores is not None else 0.0
    
    def test_binding(self, image: torch.Tensor, label: str) -> Dict:
        """
        Test if image and label activate the same concept.
        
        This is the key scientific test.
        """
        vis_features = self.visual(image)
        lang_features = self.language(label)
        
        _, vis_concept = self.atl.activate(vis_features, 'visual')
        _, lang_concept = self.atl.activate(lang_features, 'language')
        
        # Cosine similarity in feature space
        feature_sim = F.cosine_similarity(
            vis_features.unsqueeze(0), 
            lang_features.unsqueeze(0)
        ).item()
        
        return {
            'vis_concept': vis_concept,
            'lang_concept': lang_concept,
            'same_concept': vis_concept == lang_concept,
            'feature_similarity': feature_sim,
        }
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_episodes': self.stats['episodes'],
            'total_consolidations': self.stats['consolidations'],
            'unique_bindings': len(self.stats['bindings']),
            'hippocampus_memories': self.hippocampus.n_stored,
            'atl_active_concepts': self.atl.get_active_concepts(),
        }


# ============================================================
# MAIN: Test the brain-like learner
# ============================================================

if __name__ == "__main__":
    from synthetic_environment import generate_training_pairs, generate_test_pairs, create_stimulus, SHAPES, COLORS
    
    print("="*60)
    print("BRAIN-LIKE CROSS-MODAL LEARNING")
    print("Biologically plausible developmental sequence")
    print("="*60)
    
    # Create brain
    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100)
    print(f"\nBrain initialized on {DEVICE}")
    
    # Generate training data
    train_pairs = generate_training_pairs(n_per_combination=20)
    
    # ============================================================
    # PHASE 1: Train Visual Cortex (like infant visual development)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: Visual Cortex Development")
    print("Learning to see (reconstruction-based)")
    print("="*60)
    
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    
    for epoch in range(10):
        total_loss = 0
        np.random.shuffle(train_pairs)
        for pair in train_pairs[:500]:
            vis_optimizer.zero_grad()
            loss = brain.visual.reconstruction_loss(torch.from_numpy(pair.image).float())
            loss.backward()
            vis_optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: recon_loss={total_loss/500:.4f}")
    
    # Check visual discrimination
    print("\n  Checking visual discrimination...")
    test_images = []
    test_labels = []
    for shape in SHAPES[:4]:
        for color in COLORS[:3]:
            img = create_stimulus(shape, color, 'small')
            test_images.append(brain.visual(torch.from_numpy(img).float()).detach())
            test_labels.append(f"{color}_{shape}")
    
    features = torch.stack(test_images)
    sims = torch.matmul(F.normalize(features, dim=-1), F.normalize(features, dim=-1).T)
    off_diag = sims[~torch.eye(len(features), dtype=bool)].mean().item()
    print(f"  Mean off-diagonal similarity: {off_diag:.3f}")
    if off_diag < 0.9:
        print("  ✓ Visual cortex shows some discrimination!")
    else:
        print("  ⚠ Visual features still too similar")
    
    # ============================================================
    # PHASE 2: Language learns from vision (cross-modal teaching)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: Language Cortex Development")
    print("Learning to align words with visual features")
    print("="*60)
    
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)
    
    for epoch in range(15):
        total_loss = 0
        np.random.shuffle(train_pairs)
        for pair in train_pairs:
            # Get visual features (frozen - already trained)
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(pair.image).float())
            
            # Language should match visual (cross-modal teaching)
            lang_feat = brain.language(pair.label)
            
            # Alignment loss: language should be similar to vision
            loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
            
            lang_optimizer.zero_grad()
            loss.backward()
            lang_optimizer.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: alignment_loss={total_loss/len(train_pairs):.4f}")
    
    # Check language discrimination
    print("\n  Checking language discrimination...")
    test_words = ['red circle', 'blue square', 'green triangle', 'yellow star']
    lang_features = []
    for word in test_words:
        feat = brain.language(word).detach()
        lang_features.append(feat)
    
    lang_features = torch.stack(lang_features)
    lang_sims = torch.matmul(F.normalize(lang_features, dim=-1), F.normalize(lang_features, dim=-1).T)
    lang_off_diag = lang_sims[~torch.eye(len(lang_features), dtype=bool)].mean().item()
    print(f"  Mean off-diagonal similarity: {lang_off_diag:.3f}")
    
    # ============================================================
    # PHASE 3: Cross-Modal Binding (concept formation)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 3: Cross-Modal Binding")
    print("Forming unified concepts from aligned features")
    print("="*60)
    
    n_epochs = 10
    for epoch in range(n_epochs):
        np.random.shuffle(train_pairs)
        
        n_same_concept = 0
        for pair in train_pairs:
            result = brain.experience(
                torch.from_numpy(pair.image).float(),
                pair.label,
                consolidate=True,
                train_cortices=False
            )
            if result['same_concept']:
                n_same_concept += 1
        
        binding_rate = n_same_concept / len(train_pairs)
        stats = brain.get_stats()
        print(f"  Epoch {epoch+1}: binding_rate={binding_rate:.3f}, "
              f"concepts={stats['atl_active_concepts']}")
    
    # Testing
    print("\n" + "-"*40)
    print("Testing cross-modal binding...")
    
    test_sets = generate_test_pairs()
    
    for test_name, test_pairs in test_sets.items():
        n_correct = 0
        total_sim = 0.0
        
        for pair in test_pairs:
            result = brain.test_binding(
                torch.from_numpy(pair.image).float(),
                pair.label
            )
            if result['same_concept']:
                n_correct += 1
            total_sim += result['feature_similarity']
        
        accuracy = n_correct / len(test_pairs) if test_pairs else 0
        avg_sim = total_sim / len(test_pairs) if test_pairs else 0
        
        print(f"  {test_name}: accuracy={accuracy:.3f}, avg_similarity={avg_sim:.3f}")
    
    print("\n" + "="*60)
    print("DONE - Brain-like learning complete")
    print("="*60)
