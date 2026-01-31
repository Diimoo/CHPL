#!/usr/bin/env python3
"""
Adult-Level Dialogue System

Multi-turn conversation with:
- Conversation state tracking
- Episodic memory
- Knowledge-grounded responses
- Uncertainty acknowledgment
- Clarifying questions

Enables coherent 5-10 turn conversations.
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
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ConversationTurn:
    """A single turn in conversation."""
    speaker: str  # 'user' or 'chpl'
    text: str
    timestamp: float
    intent: Optional[str] = None
    context: Optional[Dict] = None
    confidence: float = 1.0


@dataclass
class EpisodicMemory:
    """Long-term memory of a conversation."""
    key_facts: List[str]
    topics_discussed: List[str]
    user_preferences: Dict[str, Any]
    timestamp: float


class IntentClassifier:
    """
    Classify user intent from text.
    
    Intents:
    - question: User is asking something
    - statement: User is stating a fact
    - request_explanation: User wants explanation
    - challenge: User is challenging/correcting
    - greeting: Social greeting
    - farewell: Ending conversation
    - clarification: User is clarifying previous statement
    """
    
    def __init__(self):
        # Simple keyword-based classification
        self.patterns = {
            'question': ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'is', 'are', 'can', 'could', 'would', 'do', 'does', '?'],
            'request_explanation': ['explain', 'tell me about', 'what is', 'describe', 'help me understand'],
            'challenge': ['but', 'actually', 'however', "that's wrong", "that's not", 'incorrect', 'you said'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'thanks', 'thank you', 'that helps'],
            'clarification': ['i mean', 'what i meant', 'to clarify', 'specifically'],
        }
    
    def classify(self, text: str) -> str:
        """Classify intent from text."""
        text_lower = text.lower()
        
        # Check patterns in order of specificity
        for intent in ['greeting', 'farewell', 'request_explanation', 'challenge', 'clarification', 'question']:
            for pattern in self.patterns[intent]:
                if pattern in text_lower:
                    return intent
        
        # Default to statement
        return 'statement'


class ResponseGenerator:
    """
    Generate responses based on intent and context.
    """
    
    def __init__(self, knowledge_graph=None):
        self.knowledge = knowledge_graph
        
        # Response templates
        self.templates = {
            'greeting': [
                "Hello! I'm ready to discuss what I've learned.",
                "Hi there! What would you like to explore?",
            ],
            'farewell': [
                "Goodbye! I enjoyed our conversation.",
                "Thanks for the discussion!",
            ],
            'uncertainty': [
                "I'm not certain about that. Let me explain what I do know...",
                "That's beyond my current knowledge. I've learned about {topics}.",
                "I don't have enough information to answer that confidently.",
            ],
            'clarification_request': [
                "Could you clarify what you mean by '{term}'?",
                "I want to make sure I understand - are you asking about {topic}?",
            ],
        }
    
    def generate_greeting(self) -> str:
        return np.random.choice(self.templates['greeting'])
    
    def generate_farewell(self) -> str:
        return np.random.choice(self.templates['farewell'])
    
    def generate_uncertainty(self, known_topics: List[str] = None) -> str:
        template = np.random.choice(self.templates['uncertainty'])
        if known_topics and '{topics}' in template:
            return template.format(topics=', '.join(known_topics[:3]))
        return template.replace(' {topics}', '')
    
    def generate_explanation(self, topic: str, knowledge_graph=None) -> str:
        """Generate explanation from knowledge graph."""
        if knowledge_graph is None:
            return f"I've learned some things about {topic}, but let me explain what I know..."
        
        # Search knowledge graph
        patterns = knowledge_graph.find_by_description(topic, top_k=3)
        
        if not patterns:
            return f"I haven't learned much about {topic} yet."
        
        # Collect explanations
        explanations = []
        for pid in patterns:
            if pid in knowledge_graph.patterns:
                pattern = knowledge_graph.patterns[pid]
                explanations.append(pattern.description)
        
        if explanations:
            return f"Here's what I know about {topic}: {'. '.join(explanations)}"
        
        return f"I have some knowledge about {topic}, but it's limited."
    
    def generate_answer(
        self, 
        question: str,
        context: Optional[Dict] = None,
        knowledge_graph=None,
    ) -> str:
        """Generate answer to question."""
        # Extract key terms from question
        words = question.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w not in ['what', 'where', 'when', 'which', 'that', 'this', 'have', 'does']]
        
        if not key_terms:
            return "Could you rephrase that question?"
        
        # Search for relevant knowledge
        if knowledge_graph:
            for term in key_terms:
                patterns = knowledge_graph.find_by_description(term, top_k=3)
                if patterns:
                    pattern = knowledge_graph.patterns[patterns[0]]
                    return f"Based on what I've learned: {pattern.description}"
        
        # Use context if available
        if context and 'image' in context:
            return f"Looking at the image, I can see... (visual analysis would go here)"
        
        return f"That's an interesting question about {key_terms[0]}. Let me think..."
    
    def generate_challenge_response(self, challenge: str) -> str:
        """Respond to a challenge or correction."""
        return "You make a good point. Let me reconsider... " + \
               "I may have oversimplified. Could you help me understand better?"


class DialogueManager:
    """
    Manage multi-turn conversations.
    
    Features:
    - Track conversation state
    - Remember context across turns
    - Generate coherent responses
    - Admit uncertainty
    - Ask clarifying questions
    """
    
    def __init__(self, brain=None, knowledge_graph=None):
        self.brain = brain
        self.knowledge = knowledge_graph
        
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator(knowledge_graph)
        
        # Conversation state
        self.current_conversation: List[ConversationTurn] = []
        self.conversation_topics: List[str] = []
        self.turn_count = 0
        
        # Episodic memory (long-term)
        self.episodic_memories: List[EpisodicMemory] = []
        
        # Context
        self.current_context: Dict = {}
        self.last_intent: Optional[str] = None
    
    def start_conversation(self) -> str:
        """Start a new conversation."""
        self.current_conversation = []
        self.conversation_topics = []
        self.turn_count = 0
        self.current_context = {}
        
        greeting = self.response_generator.generate_greeting()
        
        self.current_conversation.append(ConversationTurn(
            speaker='chpl',
            text=greeting,
            timestamp=datetime.now().timestamp(),
            intent='greeting',
        ))
        
        return greeting
    
    def respond(
        self,
        user_input: str,
        context: Optional[Dict] = None,
    ) -> str:
        """
        Generate response to user input.
        
        Args:
            user_input: User's message
            context: Optional context (e.g., image)
        
        Returns:
            CHPL's response
        """
        self.turn_count += 1
        
        # Classify intent
        intent = self.intent_classifier.classify(user_input)
        self.last_intent = intent
        
        # Record user turn
        self.current_conversation.append(ConversationTurn(
            speaker='user',
            text=user_input,
            timestamp=datetime.now().timestamp(),
            intent=intent,
            context=context,
        ))
        
        # Update context
        if context:
            self.current_context.update(context)
        
        # Extract topics
        self._extract_topics(user_input)
        
        # Generate response based on intent
        if intent == 'greeting':
            response = self.response_generator.generate_greeting()
        
        elif intent == 'farewell':
            response = self.response_generator.generate_farewell()
            self._consolidate_memory()
        
        elif intent == 'question':
            response = self._answer_question(user_input, context)
        
        elif intent == 'request_explanation':
            response = self._explain(user_input)
        
        elif intent == 'challenge':
            response = self._handle_challenge(user_input)
        
        elif intent == 'clarification':
            response = self._handle_clarification(user_input)
        
        else:  # statement
            response = self._acknowledge_statement(user_input)
        
        # Record CHPL turn
        self.current_conversation.append(ConversationTurn(
            speaker='chpl',
            text=response,
            timestamp=datetime.now().timestamp(),
            intent=intent,
        ))
        
        # Maybe consolidate memory
        if len(self.current_conversation) > 20:
            self._consolidate_memory()
        
        return response
    
    def _answer_question(self, question: str, context: Optional[Dict]) -> str:
        """Answer a question."""
        # Check if visual question
        if context and 'image' in context and self.brain:
            # Use visual QA if available
            if hasattr(self.brain, 'language') and hasattr(self.brain.language, 'answer_question'):
                try:
                    answer = self.brain.language.answer_question(context['image'], question)
                    return f"Looking at the image: {answer}"
                except:
                    pass
        
        # Use knowledge graph
        return self.response_generator.generate_answer(question, context, self.knowledge)
    
    def _explain(self, request: str) -> str:
        """Generate explanation."""
        # Extract topic
        words = request.lower().split()
        topic_words = [w for w in words if len(w) > 3 and w not in ['explain', 'tell', 'about', 'what', 'describe']]
        
        topic = ' '.join(topic_words[:3]) if topic_words else 'that'
        
        return self.response_generator.generate_explanation(topic, self.knowledge)
    
    def _handle_challenge(self, challenge: str) -> str:
        """Handle challenge or correction."""
        return self.response_generator.generate_challenge_response(challenge)
    
    def _handle_clarification(self, clarification: str) -> str:
        """Handle user clarification."""
        return "I understand better now. " + self._acknowledge_statement(clarification)
    
    def _acknowledge_statement(self, statement: str) -> str:
        """Acknowledge and comment on statement."""
        # Extract key concepts
        words = statement.lower().split()
        key_words = [w for w in words if len(w) > 4]
        
        if key_words:
            return f"That's interesting about {key_words[0]}. What else can you tell me?"
        
        return "I see. Please continue."
    
    def _extract_topics(self, text: str):
        """Extract and track topics from text."""
        words = text.lower().split()
        
        # Simple topic extraction: nouns longer than 4 characters
        topics = [w for w in words if len(w) > 4 and w.isalpha()]
        
        for topic in topics[:3]:
            if topic not in self.conversation_topics:
                self.conversation_topics.append(topic)
    
    def _consolidate_memory(self):
        """Consolidate conversation to episodic memory."""
        if len(self.current_conversation) < 4:
            return
        
        # Extract key facts
        key_facts = []
        for turn in self.current_conversation:
            if turn.intent in ['statement', 'question'] and len(turn.text) > 20:
                key_facts.append(turn.text[:100])
        
        memory = EpisodicMemory(
            key_facts=key_facts[:5],
            topics_discussed=self.conversation_topics.copy(),
            user_preferences={},
            timestamp=datetime.now().timestamp(),
        )
        
        self.episodic_memories.append(memory)
        
        # Keep only last few turns in short-term
        self.current_conversation = self.current_conversation[-4:]
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation."""
        return {
            'turns': self.turn_count,
            'topics': self.conversation_topics,
            'last_intent': self.last_intent,
            'memory_count': len(self.episodic_memories),
        }


class GrammarCheckedDialogue(DialogueManager):
    """
    Dialogue manager with automatic grammar correction.
    
    Ensures all responses have:
    - Proper capitalization
    - Correct grammar
    - Appropriate punctuation
    """
    
    def __init__(self, brain=None, knowledge_graph=None):
        super().__init__(brain, knowledge_graph)
        
        # Try to initialize grammar checker
        self.grammar_tool = None
        try:
            import language_tool_python
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
            print("Grammar checker initialized")
        except ImportError:
            print("Warning: language-tool-python not available, using basic grammar fixes")
    
    def correct_grammar(self, text: str) -> str:
        """
        Fix grammar errors in generated text.
        
        Common fixes:
        - Subject-verb agreement
        - Article usage (a/an/the)
        - Verb tense consistency
        - Punctuation
        """
        if self.grammar_tool is not None:
            try:
                matches = self.grammar_tool.check(text)
                import language_tool_python
                corrected = language_tool_python.utils.correct(text, matches)
                return corrected
            except Exception as e:
                pass
        
        # Basic fixes if tool not available
        return self._basic_grammar_fixes(text)
    
    def _basic_grammar_fixes(self, text: str) -> str:
        """Apply basic grammar fixes without external tool."""
        import re
        
        # Fix common contractions
        contractions = [
            (' i ', ' I '),
            (' i\'m ', ' I\'m '),
            (' i\'ve ', ' I\'ve '),
            (' i\'ll ', ' I\'ll '),
            (' i\'d ', ' I\'d '),
            ('dont ', 'don\'t '),
            ('cant ', 'can\'t '),
            ('wont ', 'won\'t '),
            ('isnt ', 'isn\'t '),
            ('arent ', 'aren\'t '),
            ('doesnt ', 'doesn\'t '),
            ('didnt ', 'didn\'t '),
            ('wasnt ', 'wasn\'t '),
            ('werent ', 'weren\'t '),
            ('hadnt ', 'hadn\'t '),
            ('hasnt ', 'hasn\'t '),
            ('havent ', 'haven\'t '),
            ('wouldnt ', 'wouldn\'t '),
            ('couldnt ', 'couldn\'t '),
            ('shouldnt ', 'shouldn\'t '),
            ('thats ', 'that\'s '),
            ('whats ', 'what\'s '),
            ('heres ', 'here\'s '),
            ('theres ', 'there\'s '),
            ('lets ', 'let\'s '),
        ]
        
        result = text
        for old, new in contractions:
            result = result.replace(old, new)
        
        # Fix subject-verb agreement patterns
        agreement_fixes = [
            (r'\b(the ball|it|he|she) are\b', r'\1 is'),
            (r'\b(they|we|you) is\b', r'\1 are'),
            (r'\bwhere is (\w+s)\b', r'where are \1'),
            (r'\bwhy it (\w+)\b', r'why does it \1'),
        ]
        
        for pattern, replacement in agreement_fixes:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Fix a/an usage
        result = re.sub(r'\ba ([aeiouAEIOU])', r'an \1', result)
        result = re.sub(r'\ban ([^aeiouAEIOU\s])', r'a \1', result)
        
        # Remove double spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        return result
    
    def ensure_proper_form(self, text: str) -> str:
        """
        Enforce proper sentence structure.
        
        Rules:
        - Capitalize first letter
        - End with punctuation (. ! ?)
        - No double spaces
        """
        if not text:
            return text
        
        # Capitalize first letter
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Add period if missing punctuation
        if text[-1] not in '.!?':
            text = text + '.'
        
        # Remove double spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        return text
    
    def respond(self, user_input: str, context: Optional[Dict] = None) -> str:
        """
        Generate response with grammar checking.
        
        Full pipeline: generate → grammar check → proper form.
        """
        # Get base response
        response = super().respond(user_input, context)
        
        # Apply grammar correction
        response = self.correct_grammar(response)
        
        # Ensure proper form
        response = self.ensure_proper_form(response)
        
        return response


def run_dialogue_demo():
    """Demo the dialogue system."""
    print("=" * 70)
    print("DIALOGUE SYSTEM DEMO")
    print("Multi-turn conversation")
    print("=" * 70)
    
    # Create dialogue manager
    dm = DialogueManager()
    
    # Start conversation
    print("\n" + "=" * 60)
    print("Starting conversation...")
    print("=" * 60)
    
    greeting = dm.start_conversation()
    print(f"\nCHPL: {greeting}")
    
    # Simulate conversation
    user_messages = [
        "Hi! I'm interested in learning about physics.",
        "What can you tell me about gravity?",
        "Why do objects fall at the same rate?",
        "But what about air resistance?",
        "That makes sense. What else have you learned?",
        "Thanks for the explanation!",
    ]
    
    for msg in user_messages:
        print(f"\nUser: {msg}")
        response = dm.respond(msg)
        print(f"CHPL: {response}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversation Summary")
    print("=" * 60)
    
    summary = dm.get_conversation_summary()
    print(f"\n  Turns: {summary['turns']}")
    print(f"  Topics: {summary['topics']}")
    print(f"  Memories: {summary['memory_count']}")
    
    print("\nDemo complete!")
    
    return dm


class InteractiveDialogue:
    """
    Interactive dialogue for terminal use.
    """
    
    def __init__(self, brain=None, knowledge_graph=None):
        self.dm = DialogueManager(brain, knowledge_graph)
    
    def run(self):
        """Run interactive dialogue session."""
        print("=" * 70)
        print("CHPL INTERACTIVE DIALOGUE")
        print("Type 'quit' to exit")
        print("=" * 70)
        
        greeting = self.dm.start_conversation()
        print(f"\nCHPL: {greeting}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nCHPL: Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = self.dm.respond(user_input)
            print(f"\nCHPL: {response}\n")
        
        # Final summary
        summary = self.dm.get_conversation_summary()
        print(f"\n[Session: {summary['turns']} turns, topics: {summary['topics']}]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive dialogue')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Run demo conversation')
    
    args = parser.parse_args()
    
    if args.interactive:
        dialogue = InteractiveDialogue()
        dialogue.run()
    else:
        dm = run_dialogue_demo()
