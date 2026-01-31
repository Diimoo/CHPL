#!/usr/bin/env python3
"""
Phase 5: Language Bootstrapping

Learn vocabulary from dictionary definitions using visual grounding.

Strategy:
1. Start with visually-grounded words (50 words: colors, shapes, relations)
2. Read dictionary definitions
3. Learn new words from definitions containing known words
4. Iterate: 50 → 500 → 2000+ words

This enables vocabulary expansion without additional visual training.
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass

from language_atl import LanguageBrain, VOCAB, IDX_TO_WORD, tokenize, DEVICE


@dataclass
class DictionaryEntry:
    """A dictionary word and its definition."""
    word: str
    definition: str
    pos: str = ""  # Part of speech


class LanguageBootstrapper:
    """
    Learn new words from definitions of known words.
    
    Uses visual grounding: if we know "red" visually, and the definition
    says "crimson: a deep red color", we can learn "crimson" ≈ "red".
    """
    
    def __init__(self, brain: LanguageBrain):
        self.brain = brain
        self.n_concepts = brain.atl.n_concepts
        
        # Initialize with visually grounded vocabulary
        self.vocabulary: Set[str] = set(VOCAB.keys())
        self.grounded_words: Set[str] = set(VOCAB.keys())
        
        # Word embeddings: word → activation pattern
        self.word_embeddings: Dict[str, torch.Tensor] = {}
        
        # Initialize embeddings for grounded words
        self._initialize_grounded_embeddings()
        
        # Learning statistics
        self.words_learned = 0
        self.words_failed = 0
        self.learning_history: List[Dict] = []
    
    def _initialize_grounded_embeddings(self):
        """Create embeddings for visually grounded words."""
        print("Initializing grounded word embeddings...")
        
        # For each word in the base vocabulary, create an activation pattern
        for word, idx in VOCAB.items():
            if word.startswith('<'):  # Skip special tokens
                continue
            
            # Create a one-hot-ish embedding based on word index
            # These will be refined during bootstrapping
            embedding = torch.zeros(self.n_concepts, device=DEVICE)
            
            # Spread activation across a few prototypes based on word semantics
            # Colors activate color-related prototypes
            if word in ['red', 'blue', 'green', 'yellow', 'purple', 'orange']:
                base = 10 + hash(word) % 20
                embedding[base:base+5] = 1.0
            # Shapes activate shape-related prototypes
            elif word in ['circle', 'square', 'triangle', 'star']:
                base = 40 + hash(word) % 20
                embedding[base:base+5] = 1.0
            # Relations activate relation prototypes
            elif word in ['above', 'below', 'left', 'right', 'next', 'to']:
                base = 70 + hash(word) % 20
                embedding[base:base+5] = 1.0
            # Other words get general embedding
            else:
                base = hash(word) % 150
                embedding[base:base+3] = 1.0
            
            # Normalize to probability distribution
            embedding = F.softmax(embedding / 0.2, dim=0)
            self.word_embeddings[word] = embedding
        
        print(f"  Initialized {len(self.word_embeddings)} grounded embeddings")
    
    def extract_known_words(self, text: str) -> List[str]:
        """Extract words from text that are in our vocabulary."""
        words = text.lower().replace(',', '').replace('.', '').replace(';', '').split()
        known = [w for w in words if w in self.vocabulary]
        return known
    
    def get_definition_embedding(self, definition: str) -> Optional[torch.Tensor]:
        """
        Compute embedding for a definition from known word embeddings.
        
        Returns weighted average of known word embeddings.
        """
        known_words = self.extract_known_words(definition)
        
        if len(known_words) == 0:
            return None
        
        # Get embeddings for known words
        embeddings = []
        for word in known_words:
            if word in self.word_embeddings:
                embeddings.append(self.word_embeddings[word])
        
        if len(embeddings) == 0:
            return None
        
        # Average embeddings
        stacked = torch.stack(embeddings)
        averaged = stacked.mean(dim=0)
        
        # Renormalize
        averaged = F.softmax(averaged / 0.2, dim=0)
        
        return averaged
    
    def learn_from_definition(self, word: str, definition: str) -> bool:
        """
        Learn a new word from its definition.
        
        Args:
            word: The new word to learn
            definition: Definition using known words
        
        Returns:
            True if learning succeeded, False otherwise
        """
        # Skip if already known
        if word in self.vocabulary:
            return True
        
        # Get embedding from definition
        embedding = self.get_definition_embedding(definition)
        
        if embedding is None:
            self.words_failed += 1
            return False
        
        # Store the learned embedding
        self.word_embeddings[word] = embedding
        self.vocabulary.add(word)
        self.words_learned += 1
        
        # Record learning history
        known_words = self.extract_known_words(definition)
        self.learning_history.append({
            'word': word,
            'definition': definition,
            'known_words_used': known_words,
            'n_known': len(known_words),
        })
        
        return True
    
    def bootstrap_from_dictionary(
        self,
        entries: List[DictionaryEntry],
        max_iterations: int = 5,
    ) -> Dict:
        """
        Learn from a dictionary, iterating until no more words can be learned.
        
        Args:
            entries: List of dictionary entries
            max_iterations: Maximum learning passes
        
        Returns:
            Statistics about learning
        """
        print(f"\nBootstrapping from {len(entries)} dictionary entries...")
        print(f"Starting vocabulary: {len(self.vocabulary)} words")
        
        stats = {
            'iterations': [],
            'initial_vocab': len(self.vocabulary),
            'final_vocab': 0,
        }
        
        for iteration in range(max_iterations):
            learned_this_round = 0
            failed_this_round = 0
            
            for entry in entries:
                if entry.word in self.vocabulary:
                    continue
                
                success = self.learn_from_definition(entry.word, entry.definition)
                if success:
                    learned_this_round += 1
                else:
                    failed_this_round += 1
            
            stats['iterations'].append({
                'iteration': iteration + 1,
                'learned': learned_this_round,
                'failed': failed_this_round,
                'total_vocab': len(self.vocabulary),
            })
            
            print(f"  Iteration {iteration + 1}: learned {learned_this_round}, "
                  f"failed {failed_this_round}, total vocab: {len(self.vocabulary)}")
            
            # Stop if no progress
            if learned_this_round == 0:
                print("  No new words learned, stopping.")
                break
        
        stats['final_vocab'] = len(self.vocabulary)
        stats['growth_factor'] = stats['final_vocab'] / stats['initial_vocab']
        
        print(f"\nFinal vocabulary: {stats['final_vocab']} words")
        print(f"Growth factor: {stats['growth_factor']:.1f}×")
        
        return stats
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute similarity between two words."""
        if word1 not in self.word_embeddings or word2 not in self.word_embeddings:
            return 0.0
        
        emb1 = self.word_embeddings[word1]
        emb2 = self.word_embeddings[word2]
        
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words to a given word."""
        if word not in self.word_embeddings:
            return []
        
        target = self.word_embeddings[word]
        similarities = []
        
        for other_word, other_emb in self.word_embeddings.items():
            if other_word == word:
                continue
            sim = F.cosine_similarity(target.unsqueeze(0), other_emb.unsqueeze(0)).item()
            similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
    
    def answer_analogy(self, A: str, B: str, C: str) -> Optional[str]:
        """
        Solve word analogy: A is to B as C is to ?
        
        Example: red:crimson :: blue:? → navy
        """
        if A not in self.word_embeddings or B not in self.word_embeddings:
            return None
        if C not in self.word_embeddings:
            return None
        
        emb_A = self.word_embeddings[A]
        emb_B = self.word_embeddings[B]
        emb_C = self.word_embeddings[C]
        
        # D = C + (B - A)
        emb_D = emb_C + (emb_B - emb_A)
        emb_D = F.softmax(emb_D / 0.2, dim=0)
        
        # Find closest word
        best_word = None
        best_sim = -1
        
        for word, emb in self.word_embeddings.items():
            if word in [A, B, C]:
                continue
            sim = F.cosine_similarity(emb_D.unsqueeze(0), emb.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_word = word
        
        return best_word


def create_color_dictionary() -> List[DictionaryEntry]:
    """Create dictionary of color words."""
    return [
        DictionaryEntry("crimson", "a deep red color"),
        DictionaryEntry("scarlet", "a brilliant red color"),
        DictionaryEntry("ruby", "a deep red precious color"),
        DictionaryEntry("maroon", "a dark red color"),
        DictionaryEntry("navy", "a dark blue color"),
        DictionaryEntry("azure", "a bright blue color"),
        DictionaryEntry("cobalt", "a deep blue color"),
        DictionaryEntry("teal", "a blue green color"),
        DictionaryEntry("cyan", "a bright blue green color"),
        DictionaryEntry("emerald", "a bright green color"),
        DictionaryEntry("olive", "a yellow green color"),
        DictionaryEntry("lime", "a bright green color"),
        DictionaryEntry("gold", "a bright yellow color"),
        DictionaryEntry("amber", "a warm yellow orange color"),
        DictionaryEntry("coral", "a pink orange color"),
        DictionaryEntry("magenta", "a red purple color"),
        DictionaryEntry("violet", "a blue purple color"),
        DictionaryEntry("lavender", "a light purple color"),
        DictionaryEntry("indigo", "a deep blue purple color"),
        DictionaryEntry("turquoise", "a blue green color"),
    ]


def create_shape_dictionary() -> List[DictionaryEntry]:
    """Create dictionary of shape words."""
    return [
        DictionaryEntry("oval", "a shape like a circle but longer"),
        DictionaryEntry("ellipse", "a shape like a circle stretched"),
        DictionaryEntry("rectangle", "a shape like a square but longer"),
        DictionaryEntry("diamond", "a square shape rotated"),
        DictionaryEntry("pentagon", "a shape with five sides"),
        DictionaryEntry("hexagon", "a shape with six sides"),
        DictionaryEntry("octagon", "a shape with eight sides"),
        DictionaryEntry("pyramid", "a triangle in three dimensions"),
        DictionaryEntry("cube", "a square in three dimensions"),
        DictionaryEntry("sphere", "a circle in three dimensions"),
        DictionaryEntry("cylinder", "a circle stretched upward"),
        DictionaryEntry("cone", "a circle with a point above"),
        DictionaryEntry("crescent", "a shape like a curved moon"),
        DictionaryEntry("heart", "a shape with two curves above"),
        DictionaryEntry("arrow", "a triangle with a line"),
    ]


def create_relation_dictionary() -> List[DictionaryEntry]:
    """Create dictionary of spatial and relational words."""
    return [
        DictionaryEntry("beneath", "below something"),
        DictionaryEntry("under", "below something"),
        DictionaryEntry("over", "above something"),
        DictionaryEntry("atop", "above and on something"),
        DictionaryEntry("beside", "next to something"),
        DictionaryEntry("near", "close to something"),
        DictionaryEntry("far", "not close to something"),
        DictionaryEntry("between", "in the middle of two things"),
        DictionaryEntry("among", "in the middle of many things"),
        DictionaryEntry("inside", "within something"),
        DictionaryEntry("outside", "not inside something"),
        DictionaryEntry("around", "circling something"),
        DictionaryEntry("through", "from one side to the other"),
        DictionaryEntry("toward", "in the direction of"),
        DictionaryEntry("away", "in the opposite direction"),
    ]


def create_concept_dictionary() -> List[DictionaryEntry]:
    """Create dictionary of abstract concept words."""
    return [
        DictionaryEntry("big", "large in size"),
        DictionaryEntry("tiny", "very small in size"),
        DictionaryEntry("huge", "very large in size"),
        DictionaryEntry("fast", "moving quickly"),
        DictionaryEntry("slow", "moving not quickly"),
        DictionaryEntry("quick", "fast in speed"),
        DictionaryEntry("bright", "full of light"),
        DictionaryEntry("dark", "without light"),
        DictionaryEntry("warm", "having some heat"),
        DictionaryEntry("cold", "without heat"),
        DictionaryEntry("hot", "very warm"),
        DictionaryEntry("soft", "not hard"),
        DictionaryEntry("hard", "not soft"),
        DictionaryEntry("smooth", "without rough texture"),
        DictionaryEntry("rough", "not smooth texture"),
        DictionaryEntry("round", "like a circle shape"),
        DictionaryEntry("flat", "without height"),
        DictionaryEntry("tall", "having much height"),
        DictionaryEntry("short", "having little height"),
        DictionaryEntry("wide", "having much width"),
        DictionaryEntry("narrow", "having little width"),
        DictionaryEntry("thick", "having much depth"),
        DictionaryEntry("thin", "having little depth"),
        DictionaryEntry("heavy", "having much weight"),
        DictionaryEntry("light", "having little weight"),
    ]


def create_action_dictionary() -> List[DictionaryEntry]:
    """Create dictionary of action and verb words."""
    return [
        DictionaryEntry("push", "to move something away"),
        DictionaryEntry("pull", "to move something toward"),
        DictionaryEntry("lift", "to move something above"),
        DictionaryEntry("drop", "to let something fall below"),
        DictionaryEntry("throw", "to push something fast"),
        DictionaryEntry("catch", "to stop something moving"),
        DictionaryEntry("hold", "to keep something still"),
        DictionaryEntry("release", "to let something go"),
        DictionaryEntry("rotate", "to turn in a circle"),
        DictionaryEntry("spin", "to rotate fast"),
        DictionaryEntry("slide", "to move along a surface"),
        DictionaryEntry("roll", "to rotate while moving"),
        DictionaryEntry("bounce", "to move up after hitting"),
        DictionaryEntry("fall", "to move down"),
        DictionaryEntry("rise", "to move up"),
        DictionaryEntry("stop", "to not move"),
        DictionaryEntry("start", "to begin to move"),
        DictionaryEntry("continue", "to keep moving"),
        DictionaryEntry("accelerate", "to move faster"),
        DictionaryEntry("decelerate", "to move slower"),
    ]


def create_full_dictionary() -> List[DictionaryEntry]:
    """Combine all dictionaries."""
    all_entries = []
    all_entries.extend(create_color_dictionary())
    all_entries.extend(create_shape_dictionary())
    all_entries.extend(create_relation_dictionary())
    all_entries.extend(create_concept_dictionary())
    all_entries.extend(create_action_dictionary())
    return all_entries


def create_extended_full_dictionary() -> List[DictionaryEntry]:
    """Load extended dictionary with 310+ entries."""
    from extended_dictionary import create_full_extended_dictionary
    return create_full_extended_dictionary()


def run_bootstrap_experiment():
    """Run the full language bootstrapping experiment."""
    print("=" * 70)
    print("PHASE 5: LANGUAGE BOOTSTRAPPING")
    print("Learning vocabulary from dictionary definitions")
    print("=" * 70)
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    
    # Load Phase 4 brain (or Phase 3 if 4 not available)
    from pathlib import Path
    
    brain = None
    for results_dir in ['../abstraction_results', '../language_results', '../causal_results']:
        if Path(results_dir).exists():
            model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]
            if model_files:
                model_files.sort(reverse=True)
                model_path = os.path.join(results_dir, model_files[0])
                print(f"Loading brain from: {model_path}")
                
                brain = LanguageBrain(
                    feature_dim=64,
                    n_concepts=200,
                    visual_input_size=56,
                )
                
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
                brain.visual.load_state_dict(checkpoint['visual_state'])
                
                if 'generator_state' in checkpoint:
                    brain.generator.load_state_dict(checkpoint['generator_state'])
                
                print("Loaded brain weights")
                break
    
    if brain is None:
        print("No pre-trained brain found. Creating new brain...")
        brain = LanguageBrain(feature_dim=64, n_concepts=200, visual_input_size=56)
    
    # Create bootstrapper
    bootstrapper = LanguageBootstrapper(brain)
    
    # Create dictionary - use extended version
    try:
        dictionary = create_extended_full_dictionary()
        print(f"\nUsing EXTENDED dictionary: {len(dictionary)} entries")
    except ImportError:
        dictionary = create_full_dictionary()
        print(f"\nUsing basic dictionary: {len(dictionary)} entries")
    
    # Bootstrap
    stats = bootstrapper.bootstrap_from_dictionary(dictionary, max_iterations=5)
    
    # Test learned words
    print("\n" + "=" * 60)
    print("TESTING LEARNED VOCABULARY")
    print("=" * 60)
    
    # Test word similarities
    print("\n1. Word Similarities:")
    test_pairs = [
        ('red', 'crimson'),
        ('red', 'scarlet'),
        ('blue', 'navy'),
        ('blue', 'azure'),
        ('circle', 'oval'),
        ('square', 'rectangle'),
        ('above', 'over'),
        ('below', 'beneath'),
        ('big', 'huge'),
        ('small', 'tiny'),
    ]
    
    for w1, w2 in test_pairs:
        if w2 in bootstrapper.vocabulary:
            sim = bootstrapper.similarity(w1, w2)
            print(f"  {w1} ↔ {w2}: {sim:.3f}")
    
    # Test most similar words
    print("\n2. Most Similar Words:")
    test_words = ['red', 'blue', 'circle', 'above', 'big']
    
    for word in test_words:
        similar = bootstrapper.most_similar(word, top_k=3)
        similar_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
        print(f"  {word}: {similar_str}")
    
    # Test analogies
    print("\n3. Word Analogies:")
    analogies = [
        ('red', 'crimson', 'blue'),  # red:crimson :: blue:?
        ('big', 'huge', 'small'),    # big:huge :: small:?
        ('above', 'over', 'below'),  # above:over :: below:?
    ]
    
    for A, B, C in analogies:
        answer = bootstrapper.answer_analogy(A, B, C)
        print(f"  {A}:{B} :: {C}:{answer}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} seconds")
    print(f"\n  Vocabulary Growth:")
    print(f"    Initial: {stats['initial_vocab']} words")
    print(f"    Final: {stats['final_vocab']} words")
    print(f"    Growth: {stats['growth_factor']:.1f}×")
    
    print(f"\n  Learning Statistics:")
    print(f"    Words learned: {bootstrapper.words_learned}")
    print(f"    Words failed: {bootstrapper.words_failed}")
    
    # Save results
    results_dir = Path('../bootstrap_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'stats': stats,
        'words_learned': bootstrapper.words_learned,
        'words_failed': bootstrapper.words_failed,
        'final_vocabulary_size': len(bootstrapper.vocabulary),
        'learning_history': bootstrapper.learning_history[:100],  # First 100 entries
        'duration_seconds': duration,
    }
    
    results_path = results_dir / f'bootstrap_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    # Save vocabulary
    vocab_path = results_dir / f'vocabulary_{timestamp}.txt'
    with open(vocab_path, 'w') as f:
        for word in sorted(bootstrapper.vocabulary):
            f.write(f"{word}\n")
    print(f"  Vocabulary saved to: {vocab_path}")
    
    return bootstrapper, stats


if __name__ == "__main__":
    bootstrapper, stats = run_bootstrap_experiment()
