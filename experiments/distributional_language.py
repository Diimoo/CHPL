#!/usr/bin/env python3
"""
Adult-Level Language Learning via Distributional Semantics

Learn 50,000+ words from Wikipedia using Word2Vec-style co-occurrence.

Method: Skip-gram with negative sampling
- "The red circle is above the blue square"
- Learn: red ≈ blue (both colors), circle ≈ square (both shapes)
- Ground to CHPL's visual concepts

This enables vocabulary expansion from 320 → 50,000+ words.
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import pickle

# Try to import gensim for efficient Word2Vec
try:
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("Warning: gensim not installed, using pure PyTorch implementation")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DistributionalLanguage:
    """
    Learn words from context, not definitions.
    
    Uses Word2Vec skip-gram to learn word embeddings from text corpus,
    then grounds them to CHPL's visual concepts.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        
        # Vocabulary
        self.vocab: Dict[str, int] = {}  # word → index
        self.idx_to_word: Dict[int, str] = {}  # index → word
        self.word_counts: Dict[str, int] = defaultdict(int)
        
        # Embeddings
        self.embeddings: Optional[torch.Tensor] = None
        
        # Grounding to CHPL
        self.grounded_words: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self.total_words_processed = 0
        self.corpus_size = 0
    
    def clean_word(self, word: str) -> str:
        """Clean and normalize a word."""
        word = word.lower().strip()
        word = re.sub(r'[^a-z\'-]', '', word)
        return word
    
    def tokenize_line(self, line: str) -> List[str]:
        """Tokenize a line of text."""
        words = line.split()
        cleaned = []
        for word in words:
            word = self.clean_word(word)
            if len(word) > 1:  # Skip single characters
                cleaned.append(word)
        return cleaned
    
    def build_vocab_from_corpus(
        self, 
        corpus_path: str,
        min_count: int = 10,
        max_vocab: int = 64000,
    ) -> int:
        """
        Build vocabulary from text corpus.
        
        Args:
            corpus_path: Path to corpus directory or file
            min_count: Minimum word frequency
            max_vocab: Maximum vocabulary size
        
        Returns:
            Vocabulary size
        """
        print(f"Building vocabulary from: {corpus_path}")
        
        corpus_path = Path(corpus_path)
        
        # Collect all text files
        if corpus_path.is_dir():
            files = list(corpus_path.rglob('*.txt')) + list(corpus_path.rglob('wiki_*'))
        else:
            files = [corpus_path]
        
        print(f"Found {len(files)} files")
        
        # Count words
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        words = self.tokenize_line(line)
                        for word in words:
                            self.word_counts[word] += 1
                        self.total_words_processed += len(words)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        print(f"Processed {self.total_words_processed:,} words")
        print(f"Unique words before filtering: {len(self.word_counts):,}")
        
        # Filter by frequency
        vocab_list = [
            (word, count) 
            for word, count in self.word_counts.items() 
            if count >= min_count
        ]
        
        # Sort by frequency
        vocab_list.sort(key=lambda x: x[1], reverse=True)
        
        # Limit size
        vocab_list = vocab_list[:max_vocab]
        
        # Build vocab dict
        self.vocab = {word: i for i, (word, count) in enumerate(vocab_list)}
        self.idx_to_word = {i: word for word, i in self.vocab.items()}
        
        print(f"Final vocabulary: {len(self.vocab):,} words")
        
        # Initialize random embeddings
        self.embeddings = torch.randn(len(self.vocab), self.embedding_dim)
        self.embeddings = F.normalize(self.embeddings, dim=1)
        
        self.corpus_size = self.total_words_processed
        
        return len(self.vocab)
    
    def train_word2vec_gensim(
        self,
        corpus_path: str,
        epochs: int = 5,
        window: int = 5,
        min_count: int = 10,
        workers: int = 4,
    ) -> None:
        """
        Train Word2Vec using gensim (fast C implementation).
        """
        if not HAS_GENSIM:
            print("gensim not available, falling back to PyTorch")
            self.train_skipgram_pytorch(corpus_path, epochs)
            return
        
        print("Training Word2Vec with gensim...")
        
        corpus_path = Path(corpus_path)
        
        # Collect sentences
        class SentenceIterator:
            def __init__(self, path):
                self.path = path
            
            def __iter__(self):
                if self.path.is_dir():
                    files = list(self.path.rglob('*.txt')) + list(self.path.rglob('wiki_*'))
                else:
                    files = [self.path]
                
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                words = line.lower().split()
                                words = [re.sub(r'[^a-z\'-]', '', w) for w in words]
                                words = [w for w in words if len(w) > 1]
                                if len(words) > 2:
                                    yield words
                    except:
                        continue
        
        sentences = SentenceIterator(corpus_path)
        
        # Progress callback
        class ProgressCallback(CallbackAny2Vec):
            def __init__(self):
                self.epoch = 0
            
            def on_epoch_end(self, model):
                self.epoch += 1
                print(f"  Epoch {self.epoch} complete")
        
        # Train model
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1,  # Skip-gram
            callbacks=[ProgressCallback()],
        )
        
        # Extract embeddings
        self.vocab = {word: i for i, word in enumerate(model.wv.index_to_key)}
        self.idx_to_word = {i: word for word, i in self.vocab.items()}
        
        self.embeddings = torch.tensor(model.wv.vectors, dtype=torch.float32)
        self.embeddings = F.normalize(self.embeddings, dim=1)
        
        print(f"Trained embeddings for {len(self.vocab):,} words")
    
    def train_skipgram_pytorch(
        self,
        corpus_path: str,
        epochs: int = 5,
        window: int = 5,
        batch_size: int = 4096,
        neg_samples: int = 5,
    ) -> None:
        """
        Train skip-gram with negative sampling in pure PyTorch.
        """
        print("Training skip-gram with PyTorch...")
        
        if len(self.vocab) == 0:
            raise ValueError("Build vocabulary first!")
        
        # Skip-gram model with negative sampling
        class SkipGramNS(nn.Module):
            def __init__(self, vocab_size, embedding_dim):
                super().__init__()
                self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
                
                # Initialize
                nn.init.xavier_uniform_(self.center_embeddings.weight)
                nn.init.xavier_uniform_(self.context_embeddings.weight)
            
            def forward(self, center, context, negative):
                # center: [batch]
                # context: [batch]
                # negative: [batch, neg_samples]
                
                center_emb = self.center_embeddings(center)  # [batch, dim]
                context_emb = self.context_embeddings(context)  # [batch, dim]
                neg_emb = self.context_embeddings(negative)  # [batch, neg, dim]
                
                # Positive score
                pos_score = torch.sum(center_emb * context_emb, dim=1)  # [batch]
                pos_loss = F.logsigmoid(pos_score)
                
                # Negative score
                neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # [batch, neg]
                neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [batch]
                
                return -(pos_loss + neg_loss).mean()
        
        model = SkipGramNS(len(self.vocab), self.embedding_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate training pairs
        print("Generating training pairs...")
        pairs = []
        
        corpus_path = Path(corpus_path)
        if corpus_path.is_dir():
            files = list(corpus_path.rglob('*.txt')) + list(corpus_path.rglob('wiki_*'))
        else:
            files = [corpus_path]
        
        for file_path in files[:100]:  # Limit files for speed
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        words = self.tokenize_line(line)
                        word_indices = [self.vocab[w] for w in words if w in self.vocab]
                        
                        for i, center_idx in enumerate(word_indices):
                            start = max(0, i - window)
                            end = min(len(word_indices), i + window + 1)
                            
                            for j in range(start, end):
                                if i != j:
                                    pairs.append((center_idx, word_indices[j]))
            except:
                continue
        
        print(f"Generated {len(pairs):,} training pairs")
        
        if len(pairs) == 0:
            print("No training pairs generated!")
            return
        
        # Shuffle
        np.random.shuffle(pairs)
        pairs = pairs[:5000000]  # Limit for memory
        
        # Negative sampling distribution (unigram^0.75)
        word_freqs = np.zeros(len(self.vocab))
        for word, idx in self.vocab.items():
            word_freqs[idx] = self.word_counts.get(word, 1)
        
        neg_probs = word_freqs ** 0.75
        neg_probs /= neg_probs.sum()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                
                centers = torch.tensor([p[0] for p in batch], device=DEVICE)
                contexts = torch.tensor([p[1] for p in batch], device=DEVICE)
                
                # Sample negatives
                negatives = np.random.choice(
                    len(self.vocab), 
                    size=(len(batch), neg_samples),
                    p=neg_probs
                )
                negatives = torch.tensor(negatives, device=DEVICE)
                
                # Forward
                loss = model(centers, contexts, negatives)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            print(f"  Epoch {epoch+1}/{epochs}: loss = {total_loss/n_batches:.4f}")
        
        # Extract embeddings (average of center and context)
        self.embeddings = (
            model.center_embeddings.weight.data.cpu() + 
            model.context_embeddings.weight.data.cpu()
        ) / 2
        self.embeddings = F.normalize(self.embeddings, dim=1)
        
        print(f"Training complete. Embeddings shape: {self.embeddings.shape}")
    
    def ground_to_chpl(
        self,
        chpl_vocab: Dict[str, torch.Tensor],
        similarity_threshold: float = 0.5,
    ) -> int:
        """
        Ground learned embeddings to CHPL's visual concepts.
        
        Args:
            chpl_vocab: Dictionary of word → activation from CHPL
            similarity_threshold: Minimum similarity for grounding
        
        Returns:
            Number of grounded words
        """
        print(f"Grounding to CHPL vocabulary ({len(chpl_vocab)} words)...")
        
        grounded = 0
        
        # First: Direct matches
        for word in self.vocab:
            if word in chpl_vocab:
                self.grounded_words[word] = chpl_vocab[word]
                grounded += 1
        
        print(f"  Direct matches: {grounded}")
        
        # Build embedding matrix for CHPL words
        chpl_words = [w for w in chpl_vocab if w in self.vocab]
        if len(chpl_words) == 0:
            print("  No overlapping vocabulary!")
            return grounded
        
        chpl_indices = [self.vocab[w] for w in chpl_words]
        chpl_embeddings = self.embeddings[chpl_indices]  # [n_chpl, dim]
        
        # Second: Similar words
        for word, idx in self.vocab.items():
            if word in self.grounded_words:
                continue
            
            word_emb = self.embeddings[idx:idx+1]  # [1, dim]
            
            # Compute similarities
            similarities = torch.mm(word_emb, chpl_embeddings.t()).squeeze(0)
            
            best_idx = similarities.argmax().item()
            best_sim = similarities[best_idx].item()
            
            if best_sim > similarity_threshold:
                best_word = chpl_words[best_idx]
                self.grounded_words[word] = chpl_vocab[best_word]
                grounded += 1
        
        print(f"  Total grounded: {grounded}")
        
        return grounded
    
    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words."""
        if word not in self.vocab:
            return []
        
        word_idx = self.vocab[word]
        word_emb = self.embeddings[word_idx:word_idx+1]
        
        similarities = torch.mm(word_emb, self.embeddings.t()).squeeze(0)
        
        top_indices = similarities.argsort(descending=True)[:top_k+1]
        
        results = []
        for idx in top_indices:
            idx = idx.item()
            if idx != word_idx:
                results.append((self.idx_to_word[idx], similarities[idx].item()))
        
        return results[:top_k]
    
    def solve_analogy(
        self, 
        a: str, 
        b: str, 
        c: str, 
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy: a:b :: c:?
        
        Uses vector arithmetic: ? = b - a + c
        """
        if a not in self.vocab or b not in self.vocab or c not in self.vocab:
            return []
        
        a_emb = self.embeddings[self.vocab[a]]
        b_emb = self.embeddings[self.vocab[b]]
        c_emb = self.embeddings[self.vocab[c]]
        
        # ? = b - a + c
        target = b_emb - a_emb + c_emb
        target = F.normalize(target.unsqueeze(0), dim=1)
        
        similarities = torch.mm(target, self.embeddings.t()).squeeze(0)
        
        # Exclude a, b, c
        exclude = {self.vocab[a], self.vocab[b], self.vocab[c]}
        
        top_indices = similarities.argsort(descending=True)
        
        results = []
        for idx in top_indices:
            idx = idx.item()
            if idx not in exclude:
                results.append((self.idx_to_word[idx], similarities[idx].item()))
                if len(results) >= top_k:
                    break
        
        return results
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            'vocab': self.vocab,
            'idx_to_word': self.idx_to_word,
            'embeddings': self.embeddings,
            'grounded_words': {k: v.cpu() for k, v in self.grounded_words.items()},
            'corpus_size': self.corpus_size,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab = data['vocab']
        self.idx_to_word = data['idx_to_word']
        self.embeddings = data['embeddings']
        self.grounded_words = data['grounded_words']
        self.corpus_size = data.get('corpus_size', 0)
        
        print(f"Loaded {len(self.vocab):,} words from {path}")


def run_distributional_experiment(
    corpus_path: str,
    output_dir: str = '../language_model',
    use_gensim: bool = True,
):
    """Run the full distributional semantics experiment."""
    print("=" * 70)
    print("ADULT LANGUAGE: DISTRIBUTIONAL SEMANTICS")
    print("Learning 50,000+ words from Wikipedia")
    print("=" * 70)
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Corpus: {corpus_path}")
    
    # Create model
    model = DistributionalLanguage(embedding_dim=64)
    
    # Build vocabulary
    print("\n" + "=" * 60)
    print("Step 1: Building vocabulary...")
    print("=" * 60)
    
    vocab_size = model.build_vocab_from_corpus(
        corpus_path,
        min_count=5,
        max_vocab=64000,
    )
    
    # Train embeddings
    print("\n" + "=" * 60)
    print("Step 2: Training word embeddings...")
    print("=" * 60)
    
    if use_gensim and HAS_GENSIM:
        model.train_word2vec_gensim(corpus_path, epochs=5)
    else:
        model.train_skipgram_pytorch(corpus_path, epochs=5)
    
    # Test embeddings
    print("\n" + "=" * 60)
    print("Step 3: Testing learned embeddings...")
    print("=" * 60)
    
    test_words = ['red', 'blue', 'circle', 'square', 'big', 'small', 'above', 'below']
    
    print("\nSimilar words:")
    for word in test_words:
        if word in model.vocab:
            similar = model.get_similar_words(word, top_k=5)
            similar_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
            print(f"  {word}: {similar_str}")
    
    print("\nAnalogies:")
    analogies = [
        ('red', 'blue', 'circle'),  # red:blue :: circle:?
        ('big', 'small', 'above'),  # big:small :: above:?
        ('man', 'woman', 'king'),   # man:woman :: king:?
    ]
    
    for a, b, c in analogies:
        if all(w in model.vocab for w in [a, b, c]):
            results = model.solve_analogy(a, b, c, top_k=3)
            if results:
                answer = results[0][0]
                print(f"  {a}:{b} :: {c}:{answer}")
    
    # Ground to CHPL
    print("\n" + "=" * 60)
    print("Step 4: Grounding to CHPL vocabulary...")
    print("=" * 60)
    
    # Load CHPL vocabulary (from Phase 5)
    chpl_vocab = {}
    vocab_file = Path('../bootstrap_results')
    if vocab_file.exists():
        vocab_files = list(vocab_file.glob('vocabulary_*.txt'))
        if vocab_files:
            vocab_files.sort(reverse=True)
            with open(vocab_files[0], 'r') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        # Placeholder activation
                        chpl_vocab[word] = torch.randn(200)
    
    if len(chpl_vocab) > 0:
        grounded = model.ground_to_chpl(chpl_vocab, similarity_threshold=0.4)
    else:
        print("  No CHPL vocabulary found for grounding")
        grounded = 0
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_dir / f'distributional_model_{timestamp}.pkl'
    model.save(str(model_path))
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    print(f"\n  Duration: {duration:.1f} seconds")
    print(f"  Vocabulary size: {len(model.vocab):,}")
    print(f"  Corpus size: {model.corpus_size:,} words")
    print(f"  Grounded words: {grounded:,}")
    print(f"  Model saved: {model_path}")
    
    # Save results
    results = {
        'vocab_size': len(model.vocab),
        'corpus_size': model.corpus_size,
        'grounded_words': grounded,
        'duration_seconds': duration,
    }
    
    results_path = output_dir / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='data/wikipedia',
                       help='Path to corpus')
    parser.add_argument('--output', type=str, default='../language_model',
                       help='Output directory')
    parser.add_argument('--no-gensim', action='store_true',
                       help='Use PyTorch instead of gensim')
    
    args = parser.parse_args()
    
    model, results = run_distributional_experiment(
        corpus_path=args.corpus,
        output_dir=args.output,
        use_gensim=not args.no_gensim,
    )
