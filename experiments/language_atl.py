#!/usr/bin/env python3
"""
Phase 3: Language Generation & Question Answering

Extends CausalATL with:
- Scene description generation
- Visual question answering
- Instruction following
- Multi-turn dialogue

This builds on Phase 1 (prediction) and Phase 2 (causal reasoning).
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from causal_atl import CausalATL, CausalBrain, DEVICE


# Simple vocabulary for scene descriptions
VOCAB = {
    '<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3,
    'a': 4, 'the': 5, 'is': 6, 'are': 7, 'there': 8,
    'red': 9, 'blue': 10, 'green': 11, 'yellow': 12, 'purple': 13, 'orange': 14,
    'circle': 15, 'square': 16, 'triangle': 17, 'star': 18,
    'small': 19, 'large': 20, 'big': 21,
    'left': 22, 'right': 23, 'above': 24, 'below': 25, 'next': 26, 'to': 27,
    'moving': 28, 'stationary': 29, 'still': 30,
    'pushes': 31, 'blocks': 32, 'touches': 33,
    'and': 34, 'with': 35, 'in': 36, 'on': 37,
    'what': 38, 'where': 39, 'why': 40, 'how': 41, 'which': 42,
    'color': 43, 'shape': 44, 'object': 45,
    'yes': 46, 'no': 47,
    'it': 48, 'they': 49,
}

# Reverse vocabulary
IDX_TO_WORD = {v: k for k, v in VOCAB.items()}

# Question types
class QuestionType(Enum):
    COLOR = 'color'      # "What color is the circle?"
    SHAPE = 'shape'      # "What shape is red?"
    LOCATION = 'location'  # "Where is the circle?"
    COUNT = 'count'      # "How many circles?"
    EXISTS = 'exists'    # "Is there a red circle?"
    RELATION = 'relation'  # "What is next to the square?"


@dataclass
class QAPair:
    """Question-answer pair for training."""
    scene_description: str
    question: str
    answer: str
    question_type: QuestionType


class GenerativeLanguageCortex(nn.Module):
    """
    Generative language model for scene descriptions.
    
    Given ATL activation, generates natural language description.
    """
    
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        hidden_dim: int = 256,
        n_concepts: int = 200,
        max_length: int = 20,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_concepts = n_concepts
        self.max_length = max_length
        
        # Project ATL activation to hidden state
        self.activation_proj = nn.Linear(n_concepts, hidden_dim)
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        self.to(DEVICE)
    
    def init_hidden(self, atl_activation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state from ATL activation."""
        # Project activation to hidden dimension
        h = self.activation_proj(atl_activation)
        
        # Shape: [num_layers, batch_size, hidden_dim]
        h = h.unsqueeze(0).unsqueeze(1).expand(2, 1, -1).contiguous()
        c = torch.zeros_like(h)
        
        return (h, c)
    
    def forward(
        self,
        atl_activation: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            atl_activation: [n_concepts] scene activation
            target_tokens: [seq_len] target token indices (for teacher forcing)
        
        Returns:
            logits: [seq_len, vocab_size] output logits
        """
        atl_activation = atl_activation.to(DEVICE)
        
        # Initialize hidden state
        hidden = self.init_hidden(atl_activation)
        
        if target_tokens is not None:
            # Teacher forcing: use target tokens as input
            target_tokens = target_tokens.to(DEVICE)
            embeddings = self.embedding(target_tokens)  # [seq_len, hidden]
            
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)  # [1, seq_len, hidden]
            
            outputs, _ = self.decoder(embeddings, hidden)
            logits = self.output_layer(outputs.squeeze(0))
            
            return logits
        else:
            return self.generate(atl_activation)
    
    def generate(
        self,
        atl_activation: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Generate description from ATL activation.
        
        Args:
            atl_activation: [n_concepts] scene activation
            max_length: maximum sequence length
            temperature: sampling temperature
        
        Returns:
            List of words forming the description
        """
        if max_length is None:
            max_length = self.max_length
        
        atl_activation = atl_activation.to(DEVICE)
        hidden = self.init_hidden(atl_activation)
        
        words = []
        token = torch.tensor([VOCAB['<start>']], device=DEVICE)
        
        for _ in range(max_length):
            # Embed current token
            emb = self.embedding(token).unsqueeze(0)  # [1, 1, hidden]
            
            # Decode
            output, hidden = self.decoder(emb, hidden)
            logits = self.output_layer(output.squeeze(0))  # [1, vocab]
            
            # Sample next token
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                token = torch.argmax(logits, dim=-1)
            
            token_id = token.item()
            
            if token_id == VOCAB['<end>']:
                break
            
            if token_id in IDX_TO_WORD:
                words.append(IDX_TO_WORD[token_id])
        
        return words
    
    def describe(self, atl_activation: torch.Tensor) -> str:
        """Generate string description."""
        words = self.generate(atl_activation, temperature=0.7)
        return ' '.join(words)


class QuestionAnswerer(nn.Module):
    """
    Visual Question Answering module.
    
    Given scene activation and question, produce answer.
    """
    
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        hidden_dim: int = 256,
        n_concepts: int = 200,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Question encoder
        self.question_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.question_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        # Scene projection
        self.scene_proj = nn.Linear(n_concepts, hidden_dim)
        
        # Combined reasoning
        self.reasoning = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Answer decoder (simple: classify into answer vocab)
        self.answer_classifier = nn.Linear(hidden_dim, vocab_size)
        
        self.to(DEVICE)
    
    def encode_question(self, question_tokens: torch.Tensor) -> torch.Tensor:
        """Encode question into vector."""
        question_tokens = question_tokens.to(DEVICE)
        
        if question_tokens.dim() == 1:
            question_tokens = question_tokens.unsqueeze(0)
        
        embeddings = self.question_embedding(question_tokens)
        _, (h, _) = self.question_encoder(embeddings)
        
        return h.squeeze(0)
    
    def forward(
        self,
        scene_activation: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Answer a question about the scene.
        
        Args:
            scene_activation: [n_concepts] scene representation
            question_tokens: [seq_len] question token indices
        
        Returns:
            answer_logits: [vocab_size] logits over answer vocabulary
        """
        scene_activation = scene_activation.to(DEVICE)
        
        # Encode scene
        scene_vec = self.scene_proj(scene_activation)
        
        # Encode question
        question_vec = self.encode_question(question_tokens)
        if question_vec.dim() == 2:
            question_vec = question_vec.squeeze(0)
        
        # Combine and reason
        combined = torch.cat([scene_vec, question_vec])
        reasoning = self.reasoning(combined)
        
        # Classify answer
        answer_logits = self.answer_classifier(reasoning)
        
        return answer_logits
    
    def answer(
        self,
        scene_activation: torch.Tensor,
        question: str,
    ) -> str:
        """Answer a question in natural language."""
        # Tokenize question
        question_tokens = tokenize(question)
        question_tensor = torch.tensor(question_tokens, device=DEVICE)
        
        # Get answer logits
        with torch.no_grad():
            logits = self.forward(scene_activation, question_tensor)
            answer_idx = torch.argmax(logits).item()
        
        return IDX_TO_WORD.get(answer_idx, '<unk>')


class LanguageBrain(CausalBrain):
    """
    Brain with language generation and QA capabilities.
    
    Extends CausalBrain with GenerativeLanguageCortex and QuestionAnswerer.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        n_concepts: int = 200,
        visual_input_size: int = 56,
    ):
        super().__init__(feature_dim, n_concepts, visual_input_size)
        
        # Add generative language components
        self.generator = GenerativeLanguageCortex(
            vocab_size=len(VOCAB),
            hidden_dim=256,
            n_concepts=n_concepts,
        )
        
        self.qa = QuestionAnswerer(
            vocab_size=len(VOCAB),
            hidden_dim=256,
            n_concepts=n_concepts,
        )
    
    def describe_scene(self, image: torch.Tensor) -> str:
        """Generate natural language description of scene."""
        activation = self.get_scene_activation(image)
        return self.generator.describe(activation)
    
    def answer_question(self, image: torch.Tensor, question: str) -> str:
        """Answer a question about the scene."""
        activation = self.get_scene_activation(image)
        return self.qa.answer(activation, question)
    
    def explain_causality(
        self,
        image_before: torch.Tensor,
        image_after: torch.Tensor,
    ) -> str:
        """Explain what caused the change between scenes."""
        causal_result = self.infer_causality_from_images(image_before, image_after)
        
        # Get interaction type
        interaction_probs = causal_result['interaction_probs']
        interaction_idx = torch.argmax(interaction_probs).item()
        
        interaction_names = ['push', 'block', 'independent', 'chain']
        interaction = interaction_names[interaction_idx] if interaction_idx < len(interaction_names) else 'unknown'
        
        # Generate explanation
        if interaction == 'push':
            return "One object pushed another, causing it to move."
        elif interaction == 'block':
            return "One object blocked another's motion."
        elif interaction == 'independent':
            return "The objects moved independently, no interaction."
        elif interaction == 'chain':
            return "A chain reaction: one object pushed another, which pushed a third."
        else:
            return "The causal relationship is unclear."


def tokenize(text: str) -> List[int]:
    """Convert text to token indices."""
    words = text.lower().replace('?', '').replace('.', '').split()
    tokens = [VOCAB.get(w, VOCAB['<unk>']) for w in words]
    return tokens


def detokenize(tokens: List[int]) -> str:
    """Convert token indices to text."""
    words = [IDX_TO_WORD.get(t, '<unk>') for t in tokens]
    return ' '.join(words)


def generate_scene_description(
    shapes: List[str],
    colors: List[str],
    relations: Optional[List[str]] = None,
) -> Tuple[str, List[int]]:
    """
    Generate a scene description and its tokenization.
    
    Example: "a red circle and a blue square"
    """
    parts = []
    
    for i, (shape, color) in enumerate(zip(shapes, colors)):
        if i == 0:
            parts.append(f"a {color} {shape}")
        else:
            parts.append(f"and a {color} {shape}")
    
    description = ' '.join(parts)
    tokens = [VOCAB['<start>']] + tokenize(description) + [VOCAB['<end>']]
    
    return description, tokens


def generate_qa_pair(
    shapes: List[str],
    colors: List[str],
) -> QAPair:
    """Generate a question-answer pair for training."""
    import random
    
    # Pick a random question type
    question_type = random.choice(list(QuestionType))
    
    if len(shapes) == 0:
        # Empty scene
        return QAPair(
            scene_description="",
            question="Is there anything?",
            answer="no",
            question_type=QuestionType.EXISTS,
        )
    
    idx = random.randint(0, len(shapes) - 1)
    shape = shapes[idx]
    color = colors[idx]
    
    if question_type == QuestionType.COLOR:
        question = f"what color is the {shape}"
        answer = color
    elif question_type == QuestionType.SHAPE:
        question = f"what shape is {color}"
        answer = shape
    elif question_type == QuestionType.EXISTS:
        question = f"is there a {color} {shape}"
        answer = "yes"
    else:
        # Default
        question = f"what is {color}"
        answer = shape
    
    description, _ = generate_scene_description(shapes, colors)
    
    return QAPair(
        scene_description=description,
        question=question,
        answer=answer,
        question_type=question_type,
    )


def generate_qa_dataset(n_samples: int = 500) -> List[QAPair]:
    """Generate QA dataset for training."""
    import random
    
    shapes = ['circle', 'square', 'triangle', 'star']
    colors = ['red', 'blue', 'green', 'yellow']
    
    dataset = []
    
    for _ in range(n_samples):
        n_objects = random.randint(1, 3)
        scene_shapes = [random.choice(shapes) for _ in range(n_objects)]
        scene_colors = [random.choice(colors) for _ in range(n_objects)]
        
        qa_pair = generate_qa_pair(scene_shapes, scene_colors)
        dataset.append(qa_pair)
    
    return dataset


if __name__ == "__main__":
    print("Testing Language ATL...")
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {len(VOCAB)}")
    
    # Test generative language cortex
    generator = GenerativeLanguageCortex()
    print(f"\nGenerativeLanguageCortex parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test generation
    test_activation = torch.randn(200, device=DEVICE)
    description = generator.describe(test_activation)
    print(f"Sample description: '{description}'")
    
    # Test QA
    qa = QuestionAnswerer()
    print(f"\nQuestionAnswerer parameters: {sum(p.numel() for p in qa.parameters()):,}")
    
    # Test answering
    question = "what color is the circle"
    answer = qa.answer(test_activation, question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    
    # Test QA dataset generation
    qa_dataset = generate_qa_dataset(n_samples=10)
    print(f"\nGenerated {len(qa_dataset)} QA pairs")
    for i, qa_pair in enumerate(qa_dataset[:3]):
        print(f"  {i+1}. Scene: {qa_pair.scene_description}")
        print(f"     Q: {qa_pair.question}")
        print(f"     A: {qa_pair.answer}")
    
    print("\n✓ Language ATL module working!")
