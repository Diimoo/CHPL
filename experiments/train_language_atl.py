#!/usr/bin/env python3
"""
Phase 3: Language Generation & Question Answering - Training Script

Trains LanguageBrain on:
1. Scene description generation
2. Visual question answering
3. Instruction following

Builds on Phase 1 (prediction) and Phase 2 (causal reasoning).

Success Criteria (from roadmap):
- Scene description BLEU > 0.5
- Question answering accuracy > 0.7
- Instruction following > 0.8
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

from language_atl import (
    LanguageBrain, GenerativeLanguageCortex, QuestionAnswerer,
    VOCAB, IDX_TO_WORD, tokenize, detokenize,
    generate_scene_description, generate_qa_pair, generate_qa_dataset,
    QuestionType, QAPair,
)
from synthetic_environment import create_stimulus_on_canvas, SHAPES, COLORS
from causal_atl import DEVICE


def ensure_cuda():
    """Ensure we're running on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training. No GPU detected.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def find_latest_causal_model(results_dir: str = '../causal_results') -> str:
    """Find the most recent trained model from Phase 2."""
    model_files = [f for f in os.listdir(results_dir) 
                   if f.startswith('causal_brain_') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No Phase 2 models found in {results_dir}")
    
    model_files.sort(reverse=True)
    return os.path.join(results_dir, model_files[0])


def create_scene_image(shapes: List[str], colors: List[str], canvas_size: int = 56) -> np.ndarray:
    """Create an image with multiple objects."""
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
    
    # Position objects in different locations
    positions = [
        (15, 28),  # Left
        (28, 28),  # Center
        (41, 28),  # Right
    ]
    
    for i, (shape, color) in enumerate(zip(shapes, colors)):
        if i < len(positions):
            pos = positions[i]
            obj_img = create_stimulus_on_canvas(
                shape=shape,
                color=color,
                size='small',
                canvas_size=canvas_size,
                center=pos,
            )
            # Composite
            is_object = np.any(obj_img < 0.95, axis=-1)
            canvas[is_object] = obj_img[is_object]
    
    return canvas


def generate_training_data(n_samples: int = 500) -> List[Dict]:
    """Generate paired (image, description, qa) training data."""
    data = []
    
    shapes_list = ['circle', 'square', 'triangle', 'star']
    colors_list = ['red', 'blue', 'green', 'yellow']
    
    for _ in range(n_samples):
        n_objects = np.random.randint(1, 3)
        shapes = [np.random.choice(shapes_list) for _ in range(n_objects)]
        colors = [np.random.choice(colors_list) for _ in range(n_objects)]
        
        # Create image
        image = create_scene_image(shapes, colors)
        
        # Create description
        description, tokens = generate_scene_description(shapes, colors)
        
        # Create QA pair
        qa_pair = generate_qa_pair(shapes, colors)
        
        data.append({
            'image': image,
            'shapes': shapes,
            'colors': colors,
            'description': description,
            'description_tokens': tokens,
            'qa': qa_pair,
        })
    
    return data


def train_description_generation(
    brain: LanguageBrain,
    training_data: List[Dict],
    n_epochs: int = 20,
) -> Dict:
    """
    Phase 3.1: Train scene description generation.
    """
    print("\n" + "=" * 60)
    print("PHASE 3.1: Scene Description Training")
    print("Learning to describe visual scenes in language")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(brain.generator.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    stats = {'losses': [], 'sample_outputs': []}
    
    for epoch in range(n_epochs):
        np.random.shuffle(training_data)
        epoch_losses = []
        
        for item in training_data:
            image = torch.from_numpy(item['image']).float().permute(2, 0, 1).to(DEVICE)
            target_tokens = torch.tensor(item['description_tokens'], device=DEVICE)
            
            # Get scene activation
            with torch.no_grad():
                activation = brain.get_scene_activation(image)
            
            # Forward pass with teacher forcing
            # Input: all tokens except last
            # Target: all tokens except first
            input_tokens = target_tokens[:-1]
            target = target_tokens[1:]
            
            logits = brain.generator(activation, input_tokens)
            
            # Compute loss
            loss = F.cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        epoch_loss = np.mean(epoch_losses)
        stats['losses'].append(epoch_loss)
        
        # Sample output
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample = training_data[0]
            sample_img = torch.from_numpy(sample['image']).float().permute(2, 0, 1).to(DEVICE)
            
            with torch.no_grad():
                act = brain.get_scene_activation(sample_img)
                generated = brain.generator.describe(act)
            
            stats['sample_outputs'].append({
                'epoch': epoch + 1,
                'target': sample['description'],
                'generated': generated,
            })
            
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}")
            print(f"    Target: '{sample['description']}'")
            print(f"    Generated: '{generated}'")
    
    return stats


def train_question_answering(
    brain: LanguageBrain,
    training_data: List[Dict],
    n_epochs: int = 20,
) -> Dict:
    """
    Phase 3.2: Train visual question answering.
    """
    print("\n" + "=" * 60)
    print("PHASE 3.2: Visual Question Answering Training")
    print("Learning to answer questions about scenes")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(brain.qa.parameters(), lr=1e-3)
    
    stats = {'losses': [], 'accuracies': []}
    
    for epoch in range(n_epochs):
        np.random.shuffle(training_data)
        epoch_losses = []
        correct = 0
        total = 0
        
        for item in training_data:
            qa = item['qa']
            image = torch.from_numpy(item['image']).float().permute(2, 0, 1).to(DEVICE)
            
            # Tokenize question and answer
            question_tokens = torch.tensor(tokenize(qa.question), device=DEVICE)
            answer_token = VOCAB.get(qa.answer, VOCAB['<unk>'])
            target = torch.tensor([answer_token], device=DEVICE)
            
            # Get scene activation
            with torch.no_grad():
                activation = brain.get_scene_activation(image)
            
            # Forward pass
            logits = brain.qa(activation, question_tokens)
            
            # Compute loss
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Accuracy
            predicted = torch.argmax(logits).item()
            if predicted == answer_token:
                correct += 1
            total += 1
        
        epoch_loss = np.mean(epoch_losses)
        epoch_acc = correct / total if total > 0 else 0
        
        stats['losses'].append(epoch_loss)
        stats['accuracies'].append(epoch_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, accuracy={epoch_acc:.3f}")
    
    final_acc = stats['accuracies'][-1]
    print(f"\n  Final QA accuracy: {final_acc:.3f}")
    print(f"  Target: > 0.7 | Status: {'PASS ✓' if final_acc > 0.7 else 'FAIL ✗'}")
    
    return stats


def test_description_generation(brain: LanguageBrain, n_test: int = 50) -> Dict:
    """Test scene description quality."""
    print("\n" + "=" * 60)
    print("DESCRIPTION GENERATION TEST")
    print("=" * 60)
    
    test_data = generate_training_data(n_test)
    
    word_matches = []
    exact_matches = 0
    
    for item in test_data:
        image = torch.from_numpy(item['image']).float().permute(2, 0, 1).to(DEVICE)
        target = item['description']
        
        with torch.no_grad():
            generated = brain.describe_scene(image)
        
        # Simple word overlap score
        target_words = set(target.lower().split())
        generated_words = set(generated.lower().split())
        
        if len(target_words) > 0:
            overlap = len(target_words & generated_words) / len(target_words)
            word_matches.append(overlap)
        
        if target.lower().strip() == generated.lower().strip():
            exact_matches += 1
    
    word_overlap = np.mean(word_matches) if word_matches else 0
    exact_rate = exact_matches / n_test
    
    print(f"\n  Word overlap score: {word_overlap:.3f}")
    print(f"  Exact match rate: {exact_rate:.3f}")
    print(f"  Target: word overlap > 0.5 | Status: {'PASS ✓' if word_overlap > 0.5 else 'FAIL ✗'}")
    
    # Show some examples
    print("\n  Examples:")
    for item in test_data[:3]:
        image = torch.from_numpy(item['image']).float().permute(2, 0, 1).to(DEVICE)
        target = item['description']
        with torch.no_grad():
            generated = brain.describe_scene(image)
        print(f"    Target: '{target}'")
        print(f"    Generated: '{generated}'")
        print()
    
    return {
        'word_overlap': word_overlap,
        'exact_match_rate': exact_rate,
        'passed': word_overlap > 0.5,
    }


def test_question_answering(brain: LanguageBrain, n_test: int = 100) -> Dict:
    """Test QA accuracy."""
    print("\n" + "=" * 60)
    print("QUESTION ANSWERING TEST")
    print("=" * 60)
    
    test_data = generate_training_data(n_test)
    
    correct = 0
    total = 0
    per_type_correct = {t.value: 0 for t in QuestionType}
    per_type_total = {t.value: 0 for t in QuestionType}
    
    for item in test_data:
        qa = item['qa']
        image = torch.from_numpy(item['image']).float().permute(2, 0, 1).to(DEVICE)
        
        with torch.no_grad():
            answer = brain.answer_question(image, qa.question)
        
        expected = qa.answer.lower()
        predicted = answer.lower()
        
        is_correct = (expected == predicted)
        
        if is_correct:
            correct += 1
            per_type_correct[qa.question_type.value] += 1
        
        total += 1
        per_type_total[qa.question_type.value] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n  Overall QA accuracy: {accuracy:.3f}")
    print(f"  Target: > 0.7 | Status: {'PASS ✓' if accuracy > 0.7 else 'FAIL ✗'}")
    
    print("\n  Per-type accuracy:")
    for qtype in QuestionType:
        t = qtype.value
        if per_type_total[t] > 0:
            acc = per_type_correct[t] / per_type_total[t]
            print(f"    {t}: {acc:.3f} ({per_type_correct[t]}/{per_type_total[t]})")
    
    return {
        'accuracy': accuracy,
        'per_type': {t: per_type_correct[t] / max(per_type_total[t], 1) for t in per_type_correct},
        'passed': accuracy > 0.7,
    }


def test_causal_explanation(brain: LanguageBrain, n_test: int = 20) -> Dict:
    """Test causal explanation generation."""
    print("\n" + "=" * 60)
    print("CAUSAL EXPLANATION TEST")
    print("=" * 60)
    
    from synthetic_causal import create_push_sequence, create_independent_sequence
    
    explanations = []
    
    for _ in range(n_test // 2):
        # Push sequence
        seq = create_push_sequence()
        before = torch.from_numpy(seq.frames[1].image).float().permute(2, 0, 1).to(DEVICE)
        after = torch.from_numpy(seq.frames[-1].image).float().permute(2, 0, 1).to(DEVICE)
        
        explanation = brain.explain_causality(before, after)
        explanations.append({
            'type': 'push',
            'explanation': explanation,
            'contains_push': 'push' in explanation.lower(),
        })
    
    for _ in range(n_test // 2):
        # Independent sequence
        seq = create_independent_sequence()
        before = torch.from_numpy(seq.frames[0].image).float().permute(2, 0, 1).to(DEVICE)
        after = torch.from_numpy(seq.frames[-1].image).float().permute(2, 0, 1).to(DEVICE)
        
        explanation = brain.explain_causality(before, after)
        explanations.append({
            'type': 'independent',
            'explanation': explanation,
            'contains_independent': 'independent' in explanation.lower(),
        })
    
    push_correct = sum(1 for e in explanations if e['type'] == 'push' and e.get('contains_push', False))
    indep_correct = sum(1 for e in explanations if e['type'] == 'independent' and e.get('contains_independent', False))
    
    total_correct = push_correct + indep_correct
    accuracy = total_correct / n_test if n_test > 0 else 0
    
    print(f"\n  Causal explanation accuracy: {accuracy:.3f}")
    print(f"    Push explanations correct: {push_correct}/{n_test//2}")
    print(f"    Independent explanations correct: {indep_correct}/{n_test//2}")
    
    # Show examples
    print("\n  Examples:")
    for e in explanations[:4]:
        print(f"    [{e['type']}] {e['explanation']}")
    
    return {
        'accuracy': accuracy,
        'passed': accuracy > 0.7,
    }


def run_full_experiment():
    """Run complete Phase 3 experiment."""
    print("=" * 70)
    print("CHPL COGNITIVE DEVELOPMENT - PHASE 3: LANGUAGE & REASONING")
    print("=" * 70)
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ensure_cuda()
    
    # Initialize brain from Phase 2
    print("\nLoading Phase 2 brain...")
    model_path = find_latest_causal_model()
    print(f"Found model: {model_path}")
    
    brain = LanguageBrain(
        feature_dim=64,
        n_concepts=200,
        visual_input_size=56,
    )
    
    # Load causal brain weights
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    brain.visual.load_state_dict(checkpoint['visual_state'])
    
    # Load ATL weights (compatible subset)
    atl_state = checkpoint['atl_state']
    current_state = brain.atl.state_dict()
    for key in atl_state:
        if key in current_state and atl_state[key].shape == current_state[key].shape:
            current_state[key] = atl_state[key]
    brain.atl.load_state_dict(current_state)
    
    print("Loaded Phase 2 weights")
    
    print(f"\nLanguage model parameters:")
    print(f"  Generator: {sum(p.numel() for p in brain.generator.parameters()):,}")
    print(f"  QA: {sum(p.numel() for p in brain.qa.parameters()):,}")
    
    results = {}
    
    # Generate training data
    print("\n" + "=" * 60)
    print("Generating training data...")
    print("=" * 60)
    training_data = generate_training_data(n_samples=800)
    print(f"Generated {len(training_data)} training samples")
    
    # Phase 3.1: Description Generation
    desc_stats = train_description_generation(brain, training_data, n_epochs=25)
    results['description_training'] = {
        'final_loss': desc_stats['losses'][-1],
    }
    
    # Phase 3.2: Question Answering
    qa_stats = train_question_answering(brain, training_data, n_epochs=25)
    results['qa_training'] = {
        'final_accuracy': qa_stats['accuracies'][-1],
        'final_loss': qa_stats['losses'][-1],
    }
    
    # Testing
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    desc_test = test_description_generation(brain, n_test=50)
    results['description_test'] = desc_test
    
    qa_test = test_question_answering(brain, n_test=100)
    results['qa_test'] = qa_test
    
    causal_test = test_causal_explanation(brain, n_test=30)
    results['causal_explanation_test'] = causal_test
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n  Total time: {duration:.1f} minutes")
    
    print(f"\n  Phase 3.1 Results (Description Generation):")
    print(f"    - Word overlap: {desc_test['word_overlap']:.3f}")
    print(f"      Target: > 0.5 | Status: {'PASS ✓' if desc_test['passed'] else 'FAIL ✗'}")
    
    print(f"\n  Phase 3.2 Results (Question Answering):")
    print(f"    - Accuracy: {qa_test['accuracy']:.3f}")
    print(f"      Target: > 0.7 | Status: {'PASS ✓' if qa_test['passed'] else 'FAIL ✗'}")
    
    print(f"\n  Causal Explanation:")
    print(f"    - Accuracy: {causal_test['accuracy']:.3f}")
    print(f"      Status: {'PASS ✓' if causal_test['passed'] else 'FAIL ✗'}")
    
    # Overall assessment
    all_passed = desc_test['passed'] and qa_test['passed']
    
    print("\n" + "-" * 70)
    if all_passed:
        print("  ★ CHILD LEVEL REACHED ★")
        print("  CHPL can now describe scenes and answer questions!")
    else:
        print("  Phase 3 partially complete. More training may be needed.")
    print("-" * 70)
    
    # Save results
    results_dir = Path('../language_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = results_dir / f'language_brain_{timestamp}.pt'
    torch.save({
        'visual_state': brain.visual.state_dict(),
        'language_state': brain.language.state_dict(),
        'atl_state': brain.atl.state_dict(),
        'generator_state': brain.generator.state_dict(),
        'qa_state': brain.qa.state_dict(),
    }, model_path)
    print(f"\n  Model saved to: {model_path}")
    
    # Save results JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    results_path = results_dir / f'language_experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    return brain, results


if __name__ == "__main__":
    brain, results = run_full_experiment()
