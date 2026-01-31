#!/usr/bin/env python3
"""
Multi-pass Vocabulary Grounding

Aggressive propagation with lower thresholds to expand grounded vocabulary.
Original: threshold=0.5, max_hops=3
Aggressive: threshold=0.3, max_hops=5
"""

import sys
sys.path.insert(0, '..')

import os
import pickle
import glob
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiPassGrounder:
    """
    Multi-pass vocabulary grounding with aggressive propagation.
    """
    
    def __init__(self, grounded_words: dict, word2vec_model):
        """
        Args:
            grounded_words: Dict mapping word -> activation tensor
            word2vec_model: Word2Vec model with vocab and get_similar_words
        """
        self.grounded = {}
        for word, activation in grounded_words.items():
            if isinstance(activation, np.ndarray):
                self.grounded[word] = torch.tensor(activation, device=DEVICE)
            elif isinstance(activation, torch.Tensor):
                self.grounded[word] = activation.to(DEVICE)
            else:
                self.grounded[word] = activation
        
        self.word2vec = word2vec_model
        self.vocab = set(word2vec_model.vocab.keys()) if hasattr(word2vec_model, 'vocab') else set()
        
        print(f"Loaded {len(self.grounded)} grounded words")
        print(f"Word2Vec vocab size: {len(self.vocab)}")
    
    def get_similar_words(self, word, n=20):
        """Get similar words from word2vec model."""
        try:
            if hasattr(self.word2vec, 'get_similar_words'):
                return self.word2vec.get_similar_words(word, n=n)
            elif hasattr(self.word2vec, 'wv'):
                # gensim model
                return self.word2vec.wv.most_similar(word, topn=n)
            elif hasattr(self.word2vec, 'model') and hasattr(self.word2vec.model, 'wv'):
                return self.word2vec.model.wv.most_similar(word, topn=n)
            else:
                return []
        except KeyError:
            return []
    
    def propagate_with_threshold(self, threshold=0.3, max_hops=5, min_neighbors=2):
        """
        Aggressive propagation with lower similarity threshold.
        
        Args:
            threshold: Minimum similarity to consider a neighbor
            max_hops: Maximum propagation depth
            min_neighbors: Minimum grounded neighbors required
        
        Returns:
            Number of newly grounded words
        """
        total_newly_grounded = 0
        
        print(f"\nStarting multi-pass propagation:")
        print(f"  Threshold: {threshold}")
        print(f"  Max hops: {max_hops}")
        print(f"  Min neighbors: {min_neighbors}")
        print(f"  Initial grounded: {len(self.grounded)}")
        
        for hop in range(max_hops):
            hop_grounded = 0
            candidates = []
            
            # Find ungrounded words
            for word in self.vocab:
                if word not in self.grounded:
                    candidates.append(word)
            
            print(f"\nHop {hop+1}: Processing {len(candidates)} ungrounded words...")
            
            for word in tqdm(candidates, desc=f"Hop {hop+1}"):
                # Find grounded neighbors
                similar = self.get_similar_words(word, n=20)
                
                grounded_neighbors = []
                for neighbor_word, score in similar:
                    if neighbor_word in self.grounded and score > threshold:
                        grounded_neighbors.append((neighbor_word, score))
                
                if len(grounded_neighbors) >= min_neighbors:
                    # Weighted average of neighbor activations
                    weights = torch.tensor(
                        [s for _, s in grounded_neighbors],
                        device=DEVICE
                    )
                    weights = weights / weights.sum()
                    
                    neighbor_acts = torch.stack([
                        self.grounded[w] if isinstance(self.grounded[w], torch.Tensor) 
                        else torch.tensor(self.grounded[w], device=DEVICE)
                        for w, _ in grounded_neighbors
                    ])
                    
                    # Weighted combination
                    grounded_act = (neighbor_acts * weights.unsqueeze(1)).sum(dim=0)
                    grounded_act = F.normalize(grounded_act, dim=0)
                    
                    self.grounded[word] = grounded_act
                    hop_grounded += 1
            
            total_newly_grounded += hop_grounded
            print(f"Hop {hop+1}: +{hop_grounded} words (total: {len(self.grounded)})")
            
            if hop_grounded == 0:
                print("No new words grounded, stopping early")
                break
        
        print(f"\nMulti-pass complete:")
        print(f"  Total newly grounded: {total_newly_grounded}")
        print(f"  Final grounded vocabulary: {len(self.grounded)}")
        
        return total_newly_grounded
    
    def save(self, output_path):
        """Save grounded vocabulary to file."""
        # Convert tensors to numpy for serialization
        save_dict = {}
        for word, activation in self.grounded.items():
            if isinstance(activation, torch.Tensor):
                save_dict[word] = activation.cpu().numpy()
            else:
                save_dict[word] = activation
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved {len(save_dict)} grounded words to {output_path}")


def find_latest_file(pattern):
    """Find the most recent file matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_word2vec_model(model_path):
    """Load word2vec model from pickle file."""
    print(f"Loading word2vec model from {model_path}")
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create a simple wrapper
    class Word2VecWrapper:
        def __init__(self, data):
            if isinstance(data, dict):
                self.vocab = data.get('vocab', {})
                self.idx_to_word = data.get('idx_to_word', {})
                self.embeddings = data.get('embeddings', None)
                self.model = data.get('model', None)
                
                # Convert embeddings to numpy if needed
                if self.embeddings is not None:
                    if isinstance(self.embeddings, torch.Tensor):
                        self.embeddings = self.embeddings.cpu().numpy()
                    # Normalize for cosine similarity
                    norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    self.normalized_embeddings = self.embeddings / norms
            else:
                # Assume it's a gensim model
                self.model = data
                self.vocab = {w: i for i, w in enumerate(data.wv.index_to_key)}
                self.embeddings = None
                self.normalized_embeddings = None
        
        def get_similar_words(self, word, n=20):
            # Try gensim model first
            if self.model is not None and hasattr(self.model, 'wv'):
                try:
                    return self.model.wv.most_similar(word, topn=n)
                except KeyError:
                    return []
            
            # Use raw embeddings
            if self.embeddings is not None and word in self.vocab:
                word_idx = self.vocab[word]
                word_vec = self.normalized_embeddings[word_idx]
                
                # Compute similarities
                similarities = np.dot(self.normalized_embeddings, word_vec)
                
                # Get top n+1 (excluding self)
                top_indices = np.argsort(similarities)[::-1][:n+1]
                
                results = []
                for idx in top_indices:
                    if idx != word_idx and idx in self.idx_to_word:
                        similar_word = self.idx_to_word[idx]
                        score = float(similarities[idx])
                        results.append((similar_word, score))
                        if len(results) >= n:
                            break
                
                return results
            
            return []
    
    return Word2VecWrapper(data)


def load_grounded_vocabulary(grounding_path):
    """Load existing grounded vocabulary."""
    print(f"Loading grounded vocabulary from {grounding_path}")
    
    with open(grounding_path, 'rb') as f:
        data = pickle.load(f)
    
    result = {}
    
    # Handle different formats
    if isinstance(data, dict):
        # Check if nested under 'grounded' key
        if 'grounded' in data and isinstance(data['grounded'], dict):
            data = data['grounded']
        
        for word, value in data.items():
            # Skip non-word keys
            if word in ['stats', 'timestamp', 'config']:
                continue
            
            # Handle different value formats
            activation = None
            if isinstance(value, dict) and 'activation' in value:
                activation = value['activation']
            elif hasattr(value, 'activation'):
                activation = value.activation
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                activation = value
            
            # Convert to numpy if tensor
            if isinstance(activation, torch.Tensor):
                activation = activation.cpu().numpy()
            
            if activation is not None:
                result[word] = activation
    
    print(f"  Loaded {len(result)} words")
    return result


def run_multipass_grounding(
    grounding_path=None,
    word2vec_path=None,
    output_path=None,
    threshold=0.3,
    max_hops=5,
    min_neighbors=2
):
    """
    Run multi-pass grounding experiment.
    """
    base_dir = Path(__file__).parent.parent
    
    # Find input files
    if grounding_path is None:
        grounding_path = find_latest_file(str(base_dir / 'grounded_vocabulary*.pkl'))
        if grounding_path is None:
            grounding_path = find_latest_file(str(base_dir / 'results/grounding*/grounded_vocabulary.pkl'))
    
    if word2vec_path is None:
        word2vec_path = find_latest_file(str(base_dir / 'language_model/distributional_model*.pkl'))
        if word2vec_path is None:
            word2vec_path = find_latest_file(str(base_dir / 'results/adult*/distributional_model.pkl'))
    
    if output_path is None:
        output_path = str(base_dir / 'grounded_vocabulary_multipass.pkl')
    
    print("=" * 60)
    print("MULTI-PASS VOCABULARY GROUNDING")
    print("=" * 60)
    print(f"Grounding input: {grounding_path}")
    print(f"Word2Vec model: {word2vec_path}")
    print(f"Output: {output_path}")
    print()
    
    if grounding_path is None or not os.path.exists(grounding_path):
        print("ERROR: No grounding file found!")
        print("Run ground_vocabulary.py first to create initial grounding.")
        return None
    
    if word2vec_path is None or not os.path.exists(word2vec_path):
        print("ERROR: No word2vec model found!")
        print("Run distributional_language.py first to create word embeddings.")
        return None
    
    # Load data
    grounded_words = load_grounded_vocabulary(grounding_path)
    word2vec_model = load_word2vec_model(word2vec_path)
    
    # Run multi-pass grounding
    grounder = MultiPassGrounder(grounded_words, word2vec_model)
    
    added = grounder.propagate_with_threshold(
        threshold=threshold,
        max_hops=max_hops,
        min_neighbors=min_neighbors
    )
    
    # Save results
    grounder.save(output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Initial grounded words: {len(grounded_words)}")
    print(f"Added by multi-pass: {added}")
    print(f"Final grounded words: {len(grounder.grounded)}")
    print(f"Output saved to: {output_path}")
    
    return grounder


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-pass vocabulary grounding')
    parser.add_argument('--grounding', type=str, help='Path to existing grounding file')
    parser.add_argument('--word2vec', type=str, help='Path to word2vec model')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold')
    parser.add_argument('--max_hops', type=int, default=5, help='Maximum propagation hops')
    parser.add_argument('--min_neighbors', type=int, default=2, help='Minimum grounded neighbors')
    
    args = parser.parse_args()
    
    run_multipass_grounding(
        grounding_path=args.grounding,
        word2vec_path=args.word2vec,
        output_path=args.output,
        threshold=args.threshold,
        max_hops=args.max_hops,
        min_neighbors=args.min_neighbors
    )
