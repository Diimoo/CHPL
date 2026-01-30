#!/usr/bin/env python3
"""
Phase 4: Abstraction & Creativity

HierarchicalATL extends LanguageATL with:
- Hierarchical concept abstraction (concrete → abstract)
- Analogical reasoning (A:B :: C:?)
- Creative generation (novel scene synthesis)
- Few-shot concept learning

This is the final phase of cognitive development.
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

from language_atl import LanguageBrain, VOCAB, IDX_TO_WORD, tokenize, DEVICE
from causal_atl import CausalATL


class HierarchicalATL(CausalATL):
    """
    Hierarchical ATL with multiple abstraction levels.
    
    Level 0: Concrete features (red, circle, small)
    Level 1: Attribute categories (color, shape, size)
    Level 2: Object categories (object, relation)
    Level 3: Abstract concepts (thing, property)
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_concepts: int = 200,
        temperature: float = 0.2,
        n_levels: int = 4,
    ):
        super().__init__(feature_dim, n_concepts, temperature)
        
        self.n_levels = n_levels
        
        # Hierarchical prototypes for each level
        self.level_prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(n_concepts, feature_dim) * 0.1)
            for _ in range(n_levels)
        ])
        
        # Level transition networks (bottom-up abstraction)
        self.abstractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_concepts, n_concepts),
                nn.ReLU(),
                nn.Linear(n_concepts, n_concepts),
            )
            for _ in range(n_levels - 1)
        ])
        
        # Analogy solver
        self.analogy_transform = nn.Sequential(
            nn.Linear(n_concepts * 2, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, n_concepts),
        )
        
        # Creative generator
        self.creative_decoder = nn.Sequential(
            nn.Linear(n_concepts, n_concepts * 2),
            nn.ReLU(),
            nn.Linear(n_concepts * 2, feature_dim),
        )
        
        # Few-shot learning module
        self.prototype_updater = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
        # Learned concept names (for few-shot learning)
        self.concept_names: Dict[str, torch.Tensor] = {}
        
        self.to(DEVICE)
    
    def activate_hierarchical(
        self,
        features: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Activate at multiple abstraction levels.
        
        Args:
            features: [feature_dim] visual/language features
        
        Returns:
            List of [n_concepts] activations, one per level
        """
        features = features.to(DEVICE)
        
        activations = []
        
        # Level 0: concrete activation
        level_0, _ = self.get_activation_pattern(features, 'visual')
        activations.append(level_0)
        
        # Higher levels: progressive abstraction
        current = level_0
        for level in range(1, self.n_levels):
            # Abstract from previous level
            abstracted = self.abstractors[level - 1](current)
            abstracted = F.softmax(abstracted / self.temperature, dim=-1)
            activations.append(abstracted)
            current = abstracted
        
        return activations
    
    def solve_analogy(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve analogy: A is to B as C is to ?
        
        Example:
        "red circle" : "blue circle" :: "red square" : "blue square"
        
        Args:
            A, B, C: [n_concepts] activation patterns
        
        Returns:
            D: [n_concepts] answer activation
        """
        A = A.to(DEVICE)
        B = B.to(DEVICE)
        C = C.to(DEVICE)
        
        # Method 1: Vector arithmetic
        # D = C + (B - A)
        transform = B - A
        D_arithmetic = C + transform
        
        # Method 2: Learned transformation
        combined = torch.cat([A, B])
        learned_transform = self.analogy_transform(combined)
        D_learned = C + learned_transform
        
        # Combine both methods
        D = 0.5 * D_arithmetic + 0.5 * D_learned
        
        # Normalize to valid distribution
        D = F.softmax(D / self.temperature, dim=-1)
        
        return D
    
    def generate_creative(
        self,
        seed: Optional[torch.Tensor] = None,
        constraints: Optional[List[torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate novel activation pattern.
        
        Args:
            seed: optional starting activation
            constraints: optional constraint activations to move toward
            temperature: sampling temperature
        
        Returns:
            activation: [n_concepts] novel pattern
            features: [feature_dim] decoded features
        """
        if seed is None:
            # Start with random activation
            activation = torch.randn(self.n_concepts, device=DEVICE) * temperature
        else:
            activation = seed.clone().to(DEVICE)
        
        # Apply constraints
        if constraints is not None:
            for constraint in constraints:
                constraint = constraint.to(DEVICE)
                # Move toward constraint
                similarity = F.cosine_similarity(
                    activation.unsqueeze(0),
                    constraint.unsqueeze(0)
                )
                # Gradient-like update
                direction = constraint - activation
                activation = activation + 0.3 * direction
        
        # Normalize
        activation = F.softmax(activation / self.temperature, dim=-1)
        
        # Decode to features
        features = self.creative_decoder(activation)
        
        return activation, features
    
    def learn_concept_few_shot(
        self,
        examples: List[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """
        Learn new concept from few examples.
        
        Args:
            examples: list of [feature_dim] example features
            name: name for the new concept
        
        Returns:
            prototype: [feature_dim] learned prototype
        """
        # Get activations for all examples
        example_acts = [self.activate_hierarchical(ex) for ex in examples]
        
        # Find common pattern at each level
        common_patterns = []
        for level in range(self.n_levels):
            level_acts = torch.stack([acts[level] for acts in example_acts])
            common = level_acts.mean(dim=0)
            common_patterns.append(common)
        
        # Create prototype from most informative level
        # Use level with highest variance (most discriminative)
        variances = [p.var().item() for p in common_patterns]
        best_level = np.argmax(variances)
        
        prototype = common_patterns[best_level]
        
        # Store with name
        self.concept_names[name] = prototype
        
        return prototype
    
    def recognize_concept(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> Optional[str]:
        """
        Recognize if features match any learned concept.
        
        Returns name of best matching concept or None.
        """
        if not self.concept_names:
            return None
        
        activations = self.activate_hierarchical(features)
        
        best_name = None
        best_sim = threshold
        
        for name, prototype in self.concept_names.items():
            for level_act in activations:
                sim = F.cosine_similarity(
                    level_act.unsqueeze(0),
                    prototype.unsqueeze(0)
                ).item()
                
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
        
        return best_name


class AbstractBrain(LanguageBrain):
    """
    Brain with abstraction and creativity capabilities.
    
    Extends LanguageBrain with HierarchicalATL.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        n_concepts: int = 200,
        visual_input_size: int = 56,
        n_levels: int = 4,
    ):
        # Initialize parent but replace ATL
        super().__init__(feature_dim, n_concepts, visual_input_size)
        
        # Replace ATL with hierarchical version
        self.atl = HierarchicalATL(
            feature_dim=feature_dim,
            n_concepts=n_concepts,
            temperature=0.2,
            n_levels=n_levels,
        )
        
        self.n_levels = n_levels
    
    def get_hierarchical_activation(
        self,
        image: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Get multi-level activation for image."""
        image = image.to(DEVICE)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.visual(image)
            if features.dim() == 2:
                features = features.squeeze(0)
            activations = self.atl.activate_hierarchical(features)
        
        return activations
    
    def solve_visual_analogy(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        image_C: torch.Tensor,
    ) -> Tuple[torch.Tensor, str]:
        """
        Solve visual analogy: A:B :: C:?
        
        Returns activation for answer and description.
        """
        act_A = self.get_scene_activation(image_A)
        act_B = self.get_scene_activation(image_B)
        act_C = self.get_scene_activation(image_C)
        
        act_D = self.atl.solve_analogy(act_A, act_B, act_C)
        
        # Generate description
        description = self.generator.describe(act_D)
        
        return act_D, description
    
    def generate_creative_scene(
        self,
        constraint_images: Optional[List[torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate novel scene activation with optional constraints.
        
        Returns activation and description.
        """
        constraints = None
        if constraint_images:
            constraints = [self.get_scene_activation(img) for img in constraint_images]
        
        activation, features = self.atl.generate_creative(
            constraints=constraints,
            temperature=temperature,
        )
        
        description = self.generator.describe(activation)
        
        return activation, description
    
    def learn_new_concept(
        self,
        example_images: List[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """
        Learn new concept from example images.
        """
        features = []
        for img in example_images:
            img = img.to(DEVICE)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            with torch.no_grad():
                feat = self.visual(img)
                if feat.dim() == 2:
                    feat = feat.squeeze(0)
                features.append(feat)
        
        prototype = self.atl.learn_concept_few_shot(features, name)
        return prototype
    
    def recognize_in_image(
        self,
        image: torch.Tensor,
    ) -> Optional[str]:
        """
        Recognize learned concepts in image.
        """
        image = image.to(DEVICE)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.visual(image)
            if features.dim() == 2:
                features = features.squeeze(0)
        
        return self.atl.recognize_concept(features)


def create_analogy_dataset(n_samples: int = 200) -> List[Dict]:
    """
    Generate analogy dataset.
    
    Format: A:B :: C:D where transformation is consistent.
    """
    from synthetic_environment import create_stimulus_on_canvas
    
    dataset = []
    
    shapes = ['circle', 'square', 'triangle', 'star']
    colors = ['red', 'blue', 'green', 'yellow']
    
    for _ in range(n_samples):
        # Pick transformation type
        transform_type = np.random.choice(['color', 'shape'])
        
        if transform_type == 'color':
            # A and C have same shape, B and D have same shape
            # A and B differ in color, C and D differ in same way
            shape1 = np.random.choice(shapes)
            shape2 = np.random.choice(shapes)
            color_A = np.random.choice(colors)
            remaining = [c for c in colors if c != color_A]
            color_B = np.random.choice(remaining)
            
            A = create_stimulus_on_canvas(shape1, color_A, 'small', 56, (28, 28))
            B = create_stimulus_on_canvas(shape1, color_B, 'small', 56, (28, 28))
            C = create_stimulus_on_canvas(shape2, color_A, 'small', 56, (28, 28))
            D = create_stimulus_on_canvas(shape2, color_B, 'small', 56, (28, 28))
            
            answer = f"{color_B} {shape2}"
        
        else:  # shape transformation
            color1 = np.random.choice(colors)
            color2 = np.random.choice(colors)
            shape_A = np.random.choice(shapes)
            remaining = [s for s in shapes if s != shape_A]
            shape_B = np.random.choice(remaining)
            
            A = create_stimulus_on_canvas(shape_A, color1, 'small', 56, (28, 28))
            B = create_stimulus_on_canvas(shape_B, color1, 'small', 56, (28, 28))
            C = create_stimulus_on_canvas(shape_A, color2, 'small', 56, (28, 28))
            D = create_stimulus_on_canvas(shape_B, color2, 'small', 56, (28, 28))
            
            answer = f"{color2} {shape_B}"
        
        dataset.append({
            'A': A, 'B': B, 'C': C, 'D': D,
            'answer': answer,
            'transform_type': transform_type,
        })
    
    return dataset


def create_few_shot_dataset(n_concepts: int = 5, examples_per: int = 3) -> List[Dict]:
    """
    Generate few-shot learning dataset.
    
    Each concept is defined by a combination of properties.
    """
    from synthetic_environment import create_stimulus_on_canvas
    
    dataset = []
    
    # Define novel "concepts" as combinations
    concepts = [
        {'name': 'warm_shape', 'colors': ['red', 'yellow', 'orange'], 'shapes': ['circle', 'star']},
        {'name': 'cool_shape', 'colors': ['blue', 'green'], 'shapes': ['square', 'triangle']},
        {'name': 'round_thing', 'colors': ['red', 'blue', 'green', 'yellow'], 'shapes': ['circle']},
        {'name': 'pointy_thing', 'colors': ['red', 'blue', 'green', 'yellow'], 'shapes': ['triangle', 'star']},
        {'name': 'boxy_thing', 'colors': ['red', 'blue', 'green', 'yellow'], 'shapes': ['square']},
    ]
    
    for concept in concepts[:n_concepts]:
        examples = []
        for _ in range(examples_per):
            color = np.random.choice(concept['colors'])
            shape = np.random.choice(concept['shapes'])
            img = create_stimulus_on_canvas(shape, color, 'small', 56, (28, 28))
            examples.append(img)
        
        # Test examples (same concept)
        test_positive = []
        for _ in range(2):
            color = np.random.choice(concept['colors'])
            shape = np.random.choice(concept['shapes'])
            img = create_stimulus_on_canvas(shape, color, 'small', 56, (28, 28))
            test_positive.append(img)
        
        # Negative examples (different concept)
        test_negative = []
        other_shapes = [s for s in ['circle', 'square', 'triangle', 'star'] if s not in concept['shapes']]
        other_colors = [c for c in ['red', 'blue', 'green', 'yellow'] if c not in concept['colors']]
        
        for _ in range(2):
            if other_shapes and np.random.rand() > 0.5:
                shape = np.random.choice(other_shapes)
                color = np.random.choice(['red', 'blue', 'green', 'yellow'])
            elif other_colors:
                shape = np.random.choice(['circle', 'square', 'triangle', 'star'])
                color = np.random.choice(other_colors)
            else:
                shape = np.random.choice(other_shapes) if other_shapes else 'circle'
                color = 'purple'
            
            img = create_stimulus_on_canvas(shape, color, 'small', 56, (28, 28))
            test_negative.append(img)
        
        dataset.append({
            'name': concept['name'],
            'examples': examples,
            'test_positive': test_positive,
            'test_negative': test_negative,
        })
    
    return dataset


if __name__ == "__main__":
    print("Testing HierarchicalATL...")
    print(f"Device: {DEVICE}")
    
    # Initialize
    h_atl = HierarchicalATL(feature_dim=64, n_concepts=200, n_levels=4)
    print(f"\nHierarchicalATL parameters:")
    total = sum(p.numel() for p in h_atl.parameters())
    print(f"  Total: {total:,}")
    
    # Test hierarchical activation
    test_features = torch.randn(64, device=DEVICE)
    activations = h_atl.activate_hierarchical(test_features)
    print(f"\nHierarchical activations:")
    for i, act in enumerate(activations):
        print(f"  Level {i}: shape={act.shape}, sum={act.sum():.3f}")
    
    # Test analogy
    A = torch.randn(200, device=DEVICE)
    B = torch.randn(200, device=DEVICE)
    C = torch.randn(200, device=DEVICE)
    D = h_atl.solve_analogy(A, B, C)
    print(f"\nAnalogy test:")
    print(f"  D shape: {D.shape}")
    print(f"  D sum: {D.sum():.3f}")
    
    # Test creative generation
    activation, features = h_atl.generate_creative(temperature=1.0)
    print(f"\nCreative generation:")
    print(f"  Activation shape: {activation.shape}")
    print(f"  Features shape: {features.shape}")
    
    # Test few-shot learning
    examples = [torch.randn(64, device=DEVICE) for _ in range(3)]
    prototype = h_atl.learn_concept_few_shot(examples, "test_concept")
    print(f"\nFew-shot learning:")
    print(f"  Prototype shape: {prototype.shape}")
    print(f"  Learned concepts: {list(h_atl.concept_names.keys())}")
    
    # Test analogy dataset
    analogy_data = create_analogy_dataset(n_samples=5)
    print(f"\nAnalogy dataset: {len(analogy_data)} samples")
    for item in analogy_data[:2]:
        print(f"  Transform: {item['transform_type']}, Answer: {item['answer']}")
    
    # Test few-shot dataset
    few_shot_data = create_few_shot_dataset(n_concepts=3, examples_per=3)
    print(f"\nFew-shot dataset: {len(few_shot_data)} concepts")
    for item in few_shot_data:
        print(f"  Concept: {item['name']}, Examples: {len(item['examples'])}")
    
    print("\n✓ HierarchicalATL module working!")
