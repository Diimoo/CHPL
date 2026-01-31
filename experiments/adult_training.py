#!/usr/bin/env python3
"""
Master Training Script for Adult-Level CHPL

Orchestrates all training for:
1. Distributional Semantics (50k+ words from Wikipedia)
2. Knowledge Graph (3k+ patterns from video)
3. Dialogue System integration
4. Continuous Observation (background)

Run: python adult_training.py --step 1  # for distributional semantics
     python adult_training.py --step 2  # for knowledge graph
     python adult_training.py --all     # for everything
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import argparse
from datetime import datetime
from pathlib import Path


def step1_distributional_semantics():
    """Train Word2Vec on Wikipedia for 50k+ words."""
    print("=" * 70)
    print("STEP 1: DISTRIBUTIONAL SEMANTICS")
    print("Training Word2Vec on Simple Wikipedia")
    print("=" * 70)
    
    from distributional_language import run_distributional_experiment
    
    # Handle paths relative to script location
    script_dir = Path(__file__).parent.parent
    corpus_path = script_dir / 'data/wikipedia/simplewiki_text.txt'
    output_dir = script_dir / 'language_model'
    
    if not Path(corpus_path).exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run Wikipedia extraction first!")
        return None
    
    model, results = run_distributional_experiment(
        corpus_path=corpus_path,
        output_dir=output_dir,
        use_gensim=True,
    )
    
    return results


def step2_knowledge_graph():
    """Build knowledge graph from video patterns."""
    print("=" * 70)
    print("STEP 2: KNOWLEDGE GRAPH")
    print("Building hierarchical pattern storage")
    print("=" * 70)
    
    from knowledge_graph import KnowledgeGraph, ScalableVideoLearner
    from hierarchical_atl import AbstractBrain, DEVICE
    
    # Load brain
    brain = None
    for results_dir in ['../abstraction_results', '../language_results']:
        if Path(results_dir).exists():
            model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]
            if model_files:
                model_files.sort(reverse=True)
                model_path = os.path.join(results_dir, model_files[0])
                print(f"Loading brain from: {model_path}")
                
                import torch
                brain = AbstractBrain(
                    feature_dim=64,
                    n_concepts=200,
                    visual_input_size=56,
                )
                
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
                brain.visual.load_state_dict(checkpoint['visual_state'])
                break
    
    if brain is None:
        print("Creating new brain...")
        brain = AbstractBrain(feature_dim=64, n_concepts=200, visual_input_size=56)
    
    # Create knowledge graph
    learner = ScalableVideoLearner(brain, '../knowledge_graph.db')
    
    # Process synthetic videos for each domain
    domains = ['physics', 'biology', 'chemistry']
    
    for domain in domains:
        print(f"\nProcessing {domain} videos...")
        video_dir = f'../data/videos/{domain}'
        
        stats = learner.process_curriculum(video_dir, domain)
        print(f"  Videos: {stats['videos']}, Patterns: {stats['patterns']}")
    
    # Summary
    kg_stats = learner.knowledge.get_stats()
    print(f"\nKnowledge Graph Stats:")
    print(f"  Total patterns: {kg_stats['total_patterns']}")
    print(f"  By domain: {kg_stats['by_domain']}")
    
    # Save
    learner.knowledge.save('../knowledge_graph.pkl')
    
    return kg_stats


def step3_dialogue_integration():
    """Integrate dialogue with knowledge graph."""
    print("=" * 70)
    print("STEP 3: DIALOGUE INTEGRATION")
    print("Connecting dialogue system to knowledge")
    print("=" * 70)
    
    from dialogue_system import DialogueManager
    from knowledge_graph import KnowledgeGraph
    
    # Load knowledge graph
    kg = KnowledgeGraph('../knowledge_graph.db')
    
    if Path('../knowledge_graph.pkl').exists():
        kg.load('../knowledge_graph.pkl')
    
    # Create dialogue manager with knowledge
    dm = DialogueManager(knowledge_graph=kg)
    
    # Test conversation
    print("\nTest conversation:")
    
    greeting = dm.start_conversation()
    print(f"CHPL: {greeting}")
    
    test_inputs = [
        "What do you know about gravity?",
        "Can you explain why objects fall?",
        "Thanks!",
    ]
    
    for user_input in test_inputs:
        print(f"User: {user_input}")
        response = dm.respond(user_input)
        print(f"CHPL: {response}")
    
    summary = dm.get_conversation_summary()
    print(f"\nConversation: {summary['turns']} turns, topics: {summary['topics']}")
    
    return summary


def step4_background_observation():
    """Start background observation pipeline."""
    print("=" * 70)
    print("STEP 4: BACKGROUND OBSERVATION")
    print("Starting continuous observation (runs in background)")
    print("=" * 70)
    
    from continuous_observer import ContinuousObservationPipeline, StreamConfig
    
    # Create pipeline
    pipeline = ContinuousObservationPipeline('../observations.db')
    
    # Add synthetic streams
    streams = [
        StreamConfig('traffic_1', 'synthetic', '', 1.0, 'Traffic camera 1'),
        StreamConfig('nature_1', 'synthetic', '', 2.0, 'Nature camera 1'),
        StreamConfig('lobby_1', 'synthetic', '', 1.0, 'Lobby camera 1'),
    ]
    
    for config in streams:
        pipeline.add_stream(config)
    
    # Start observation (30 minutes for demo)
    print("Starting observation for 30 minutes...")
    pipeline.start(duration_seconds=1800)
    
    print("Observation running in background.")
    print("Check progress with: python continuous_observer.py --check --db ../observations.db")
    
    return {'streams': len(streams), 'status': 'running'}


def run_all():
    """Run all training steps."""
    print("=" * 70)
    print("ADULT-LEVEL CHPL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    
    # Step 1: Distributional Semantics
    print("\n" + "=" * 70)
    results['step1'] = step1_distributional_semantics()
    
    # Step 2: Knowledge Graph
    print("\n" + "=" * 70)
    results['step2'] = step2_knowledge_graph()
    
    # Step 3: Dialogue Integration
    print("\n" + "=" * 70)
    results['step3'] = step3_dialogue_integration()
    
    # Step 4: Background Observation (optional, runs async)
    # results['step4'] = step4_background_observation()
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    print(f"\nResults:")
    if results.get('step1'):
        print(f"  Vocabulary: {results['step1'].get('vocab_size', 'N/A')} words")
    if results.get('step2'):
        print(f"  Patterns: {results['step2'].get('total_patterns', 'N/A')}")
    if results.get('step3'):
        print(f"  Dialogue: {results['step3'].get('turns', 'N/A')} turns tested")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'../adult_training_results_{timestamp}.json'
    
    # Convert non-serializable items
    serializable_results = {}
    for k, v in results.items():
        if v is not None:
            serializable_results[k] = {
                kk: vv for kk, vv in v.items() 
                if isinstance(vv, (int, float, str, list, dict, bool))
            }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                       help='Run specific step')
    parser.add_argument('--all', action='store_true',
                       help='Run all steps')
    
    args = parser.parse_args()
    
    if args.step == 1:
        step1_distributional_semantics()
    elif args.step == 2:
        step2_knowledge_graph()
    elif args.step == 3:
        step3_dialogue_integration()
    elif args.step == 4:
        step4_background_observation()
    elif args.all:
        run_all()
    else:
        print("Usage:")
        print("  python adult_training.py --step 1  # Distributional semantics")
        print("  python adult_training.py --step 2  # Knowledge graph")
        print("  python adult_training.py --step 3  # Dialogue integration")
        print("  python adult_training.py --step 4  # Background observation")
        print("  python adult_training.py --all     # All steps")
