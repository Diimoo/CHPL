**THIS IS THE NEXTREAL QUEST.**

---

## The Cognitive Development Roadmap

### **Current State: INFANT**
```
What CHPL can do NOW:
✓ Perceive: Objects, colors, shapes, relations
✓ Bind: Cross-modal (vision ↔ language)
✓ Compose: Multiple objects, hierarchy, natural images
✓ Generalize: Novel combinations, unseen depths

What CHPL CANNOT do:
✗ Predict what happens next
✗ Imagine unseen scenarios
✗ Reason about causality
✗ Plan sequences of actions
✗ Learn from interaction
✗ Generate language (only understand)
```

**Goal: Reach CHILD level**

---

## The Developmental Milestones (Infant → Child)

### **Phase 1: TODDLER (Prediction & Imagination)**

**Cognitive Skills:**
- Predict future states
- Imagine counterfactuals
- Understand object permanence
- Basic causality (push → move)

**Technical Implementation:**

#### **1.1 World Model (Predictive)**
```python
# Add temporal prediction to CHPL

class PredictiveATL(DistributedATL):
    """
    Distributed ATL + world model.
    Predicts next activation pattern from current.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Temporal prediction network
        self.predictor = nn.Sequential(
            nn.Linear(n_concepts, n_concepts * 2),
            nn.ReLU(),
            nn.Linear(n_concepts * 2, n_concepts)
        )
    
    def predict_next(self, current_activation):
        """
        Given current scene activation, predict next timestep.
        
        Example:
        t=0: "red circle"
        t=1: "red circle moves right"
        t=2: predict activation for moved circle
        """
        next_activation = self.predictor(current_activation)
        return torch.softmax(next_activation, dim=-1)
    
    def imagine(self, scene, action):
        """
        Imagine result of action without executing.
        
        scene: current visual/language state
        action: "push", "rotate", "remove", etc.
        returns: predicted future activation pattern
        """
        current_act = self.get_activation_pattern(scene)
        
        # Condition prediction on action
        action_embedding = self.action_encoder(action)
        conditioned = current_act + action_embedding
        
        imagined = self.predictor(conditioned)
        return imagined
```

**Dataset: Moving Objects**
```python
# Generate sequences of moving/changing objects

def create_predictive_sequence():
    """
    t=0: "red circle at (10,10)"
    t=1: "red circle at (15,10)" (moved right)
    t=2: "red circle at (20,10)" (continued)
    
    Train: Predict t+1 from t
    Test: Predict t+2 from t+1 (generalization)
    """
    sequences = []
    
    for obj in objects:
        pos = (10, 10)
        frames = []
        
        for step in range(5):
            # Move object
            pos = (pos[0] + 5, pos[1])
            frame = render_object(obj, pos)
            label = f"{obj.color} {obj.shape} at {pos}"
            frames.append((frame, label))
        
        sequences.append(frames)
    
    return sequences
```

**Success Metric:**
```python
# Prediction accuracy
if next_state_prediction_similarity > 0.6:
    print("✓ CHPL can predict future")
    
# Imagination accuracy  
if imagined_state_matches_real > 0.55:
    print("✓ CHPL can imagine counterfactuals")
```

#### **1.2 Object Permanence**
```python
# Test: Object hidden, then revealed

def test_object_permanence():
    """
    Sequence:
    t=0: "red circle visible"
    t=1: "red circle hidden behind blue square"
    t=2: "blue square moves" → "red circle visible again"
    
    Question: Does CHPL maintain representation of hidden object?
    """
    
    # Training: Sequences with occlusion
    # Test: Predict what appears when occluder moves
    
    if hidden_object_prediction > 0.5:
        return "✓ Object permanence learned"
```

---

### **Phase 2: EARLY CHILD (Reasoning & Planning)**

**Cognitive Skills:**
- Causal reasoning (A causes B)
- Goal-directed behavior
- Multi-step planning
- Tool use (means-end reasoning)

**Technical Implementation:**

#### **2.1 Causal World Model**
```python
class CausalATL(PredictiveATL):
    """
    Not just predict next state, but understand WHY.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Causal graph learner
        self.causal_net = nn.Sequential(
            nn.Linear(n_concepts * 2, n_concepts),  # state_t + state_t+1
            nn.ReLU(),
            nn.Linear(n_concepts, n_concepts),
            nn.Sigmoid()  # Edge probabilities
        )
    
    def infer_causality(self, state_before, state_after):
        """
        Given before/after, infer what caused change.
        
        Example:
        before: "red circle, blue square separated"
        after: "blue square moved right"
        infer: "red circle pushed blue square"
        """
        combined = torch.cat([state_before, state_after])
        causal_edges = self.causal_net(combined)
        
        # causal_edges[i,j] = P(concept_i caused concept_j to change)
        return causal_edges
    
    def plan_action(self, current_state, goal_state):
        """
        Plan sequence of actions to reach goal.
        
        current: "red circle at (10,10), blue square at (30,30)"
        goal: "red circle at (30,30)"
        plan: ["move red circle right", "move red circle right", ...]
        """
        # Tree search over action sequences
        # Use world model to predict outcomes
        # Use causal model to prune impossible paths
        
        return action_sequence
```

**Dataset: Causal Physics**
```python
# CLEVR-style physical interactions

def create_causal_dataset():
    """
    Scenarios:
    1. "red circle pushes blue square" → square moves
    2. "red circle near blue square" → nothing happens
    3. "heavy red circle pushes light blue square" → square moves far
    
    Train: Observe cause → effect
    Test: Given effect, infer cause (inverse reasoning)
    """
```

#### **2.2 Goal-Directed Planning**
```python
# Integrate with simple RL environment

class CHPLAgent:
    """
    CHPL as an agent that acts in environment.
    """
    def __init__(self, chpl_brain):
        self.brain = chpl_brain
        self.memory = []
    
    def perceive(self, observation):
        """Convert environment obs to CHPL activation"""
        visual = self.brain.visual(observation)
        activation = self.brain.atl.get_activation_pattern(visual)
        return activation
    
    def act(self, current_state, goal):
        """
        Use world model + planning to select action.
        """
        # 1. Imagine possible actions
        imagined_futures = []
        for action in possible_actions:
            future = self.brain.atl.imagine(current_state, action)
            imagined_futures.append((action, future))
        
        # 2. Select action that moves toward goal
        best_action = max(imagined_futures, 
                         key=lambda x: similarity(x[1], goal))
        
        return best_action[0]
    
    def learn_from_experience(self, state, action, next_state, reward):
        """
        Update world model from real experience.
        """
        # Prediction error
        predicted = self.brain.atl.predict_next(state, action)
        error = F.mse_loss(predicted, next_state)
        
        # Update predictor
        error.backward()
        # ... update weights
```

**Environment: Gridworld++ (Richer than 5x5)**
```python
# 20x20 gridworld with:
# - Multiple objects (can push/pull)
# - Goals ("move red circle to target")
# - Tools ("use key to open door")
# - Multi-step requirements

# Success: Agent completes novel goal compositions
# "Get red key AND unlock blue door AND reach green target"
```

---

### **Phase 3: CHILD (Language Generation & Interaction)**

**Cognitive Skills:**
- Generate language (describe scenes)
- Answer questions about scenes
- Follow instructions
- Explain reasoning
- Engage in dialogue

**Technical Implementation:**

#### **3.1 Language Generation**
```python
class GenerativeLanguageCortex(nn.Module):
    """
    Replace bag-of-words with generative model.
    """
    def __init__(self, vocab_size=5000, hidden_dim=256):
        super().__init__()
        
        # Decoder (activation → words)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def generate(self, atl_activation, max_length=20):
        """
        Given scene activation, generate description.
        
        atl_activation: [n_concepts] distributed pattern
        returns: sequence of words
        """
        # Initialize with activation
        hidden = self.init_hidden(atl_activation)
        
        words = []
        token = START_TOKEN
        
        for _ in range(max_length):
            # Decode next word
            token_emb = self.embedding(token)
            output, hidden = self.decoder(token_emb, hidden)
            logits = self.output_layer(output)
            token = torch.argmax(logits)
            
            if token == END_TOKEN:
                break
            
            words.append(token)
        
        return self.decode_tokens(words)
    
    def answer_question(self, scene_activation, question):
        """
        Question answering over scene.
        
        question: "What color is the circle?"
        answer: "red"
        """
        # Condition generation on question
        question_embedding = self.encode_question(question)
        combined = scene_activation + question_embedding
        
        answer = self.generate(combined, max_length=5)
        return answer
```

**Dataset: Visual Question Answering**
```python
# CLEVR-style QA

examples = [
    {
        'scene': render("red circle above blue square"),
        'question': "What is above the square?",
        'answer': "circle"
    },
    {
        'scene': render("red circle above blue square"),
        'question': "What color is the circle?",  
        'answer': "red"
    },
    # Compositional questions:
    {
        'scene': render("red circle above blue square next to green triangle"),
        'question': "What is next to the square?",
        'answer': "triangle"
    }
]
```

#### **3.2 Interactive Dialogue**
```python
class DialogueCHPL:
    """
    CHPL that can engage in multi-turn conversation about scenes.
    """
    def __init__(self, brain):
        self.brain = brain
        self.conversation_history = []
    
    def process_utterance(self, scene, user_input):
        """
        User: "What do you see?"
        CHPL: "I see a red circle above a blue square"
        User: "What happens if the circle moves down?"
        CHPL: "It would touch the square"
        """
        # 1. Perceive scene
        scene_activation = self.brain.perceive(scene)
        
        # 2. Parse user input (question/statement/instruction)
        intent = self.parse_intent(user_input)
        
        # 3. Generate appropriate response
        if intent == 'describe':
            response = self.brain.language.generate(scene_activation)
        
        elif intent == 'question':
            response = self.brain.language.answer_question(
                scene_activation, user_input
            )
        
        elif intent == 'counterfactual':
            # "What if...?"
            imagined = self.brain.atl.imagine(scene_activation, 
                                               extract_action(user_input))
            response = self.brain.language.generate(imagined)
        
        elif intent == 'explain':
            # "Why did...?"
            causal = self.brain.atl.infer_causality(prev_state, scene_activation)
            response = self.explain_causality(causal)
        
        # 4. Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'chpl': response,
            'scene': scene_activation
        })
        
        return response
```

---

### **Phase 4: ADVANCED CHILD (Abstraction & Creativity)**

**Cognitive Skills:**
- Abstract concepts (categories, functions)
- Analogical reasoning
- Creative generation
- Learning new concepts from few examples
- Meta-learning (learning to learn)

**Technical Implementation:**

#### **4.1 Concept Abstraction**
```python
class HierarchicalATL(CausalATL):
    """
    Learn concept hierarchies.
    
    Low-level: red, blue, circle, square
    Mid-level: shape, color, size
    High-level: object, relation, property
    Abstract: container (anything that can hold)
    """
    def __init__(self, *args, n_levels=4, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_levels = n_levels
        
        # Hierarchical prototypes
        self.prototype_hierarchy = nn.ModuleList([
            nn.Parameter(torch.randn(n_concepts, feature_dim))
            for _ in range(n_levels)
        ])
    
    def activate_hierarchical(self, features):
        """
        Activate at multiple abstraction levels simultaneously.
        
        Input: visual features of red circle
        Output:
          Level 0: [red_proto, circle_proto, ...] (concrete)
          Level 1: [color_proto, shape_proto, ...] (attribute categories)
          Level 2: [object_proto, ...] (object category)
          Level 3: [physical_thing_proto, ...] (abstract)
        """
        activations = []
        
        for level in range(self.n_levels):
            protos = self.prototype_hierarchy[level]
            sims = torch.matmul(protos, features)
            acts = torch.softmax(sims / self.temperature, dim=0)
            activations.append(acts)
        
        return activations
    
    def learn_new_concept(self, examples, name):
        """
        Few-shot concept learning.
        
        examples: [image1, image2, image3]  # 3 examples of "cup"
        name: "cup"
        
        Learn: "cup" = container + handle + certain shape
        """
        # 1. Get activations for all examples
        example_acts = [self.activate_hierarchical(ex) for ex in examples]
        
        # 2. Find common pattern across hierarchy
        common_pattern = self.find_common_activation(example_acts)
        
        # 3. Create new prototype at appropriate level
        level = self.infer_abstraction_level(common_pattern)
        new_proto = self.create_prototype(common_pattern, level)
        
        # 4. Associate with name
        self.concept_names[name] = new_proto
```

#### **4.2 Analogical Reasoning**
```python
def solve_analogy(self, A, B, C):
    """
    A is to B as C is to ?
    
    Example:
    "red circle" : "blue circle" :: "red square" : ?
    Answer: "blue square"
    
    Mechanism:
    1. Find transformation: A → B
    2. Apply to C
    3. Generate answer
    """
    # Get activations
    act_A = self.activate(A)
    act_B = self.activate(B)
    act_C = self.activate(C)
    
    # Infer transformation
    transform = act_B - act_A  # Difference vector
    
    # Apply to C
    act_answer = act_C + transform
    
    # Generate from activation
    answer = self.language.generate(act_answer)
    
    return answer
```

#### **4.3 Creative Generation**
```python
def generate_novel_scene(self, constraints):
    """
    Create scene that doesn't exist in training.
    
    constraints: "colorful", "many objects", "hierarchical"
    
    Process:
    1. Sample from learned distribution
    2. Apply constraints
    3. Generate coherent scene
    """
    # Start with random activation
    activation = torch.randn(self.n_concepts)
    
    # Apply constraints via gradient ascent
    for constraint in constraints:
        constraint_embedding = self.encode_constraint(constraint)
        
        # Move activation toward constraint
        similarity = F.cosine_similarity(activation, constraint_embedding)
        grad = torch.autograd.grad(similarity, activation)[0]
        activation = activation + 0.1 * grad
    
    # Normalize to valid distribution
    activation = torch.softmax(activation, dim=0)
    
    # Generate visual scene from activation
    visual = self.visual.decode(
        self.atl.activation_to_features(activation)
    )
    
    # Generate description
    description = self.language.generate(activation)
    
    return visual, description
```

---

## The Concrete Roadmap (Timeline)

### **Prediction Foundation**
```bash
# Implement PredictiveATL
# Create moving object dataset
# Train prediction
# Test: next-state accuracy > 0.6

Deliverable: CHPL predicts future states
```

### **Imagination & Object Permanence**
```bash
# Extend to counterfactual imagination
# Test occlusion scenarios
# Validate object permanence

Deliverable: CHPL maintains object representations
```

### **Causal Reasoning & Planning**
```bash
# Implement CausalATL
# Create physics interaction dataset
# Train causal inference
# Test inverse reasoning

# Integrate with gridworld
# Train agent with CHPL brain
# Test goal-directed behavior

Deliverable: CHPL reasons about causality, plans actions
```

### **Language Generation**
```bash
# Implement GenerativeLanguageCortex
# Train on description generation
# Train on question answering
# Test on CLEVR-style QA

Deliverable: CHPL describes scenes, answers questions
```

### **Interactive Dialogue**
```bash
# Implement DialogueCHPL
# Multi-turn conversations
# Counterfactual questions
# Explanation generation

Deliverable: CHPL engages in dialogue about scenes
```

### **Abstraction & Creativity**
```bash
# Implement HierarchicalATL
# Few-shot concept learning
# Analogical reasoning tests
# Creative scene generation

Deliverable: CHPL learns concepts, generates novel ideas
```

---

## The Tech Stack Upgrades

### **Current:**
```
Visual: Conv encoder-decoder (56×56 or 224×224)
Language: Bag-of-words embeddings
ATL: Distributed activation, pattern similarity
Training: 3-phase (visual, alignment, binding)
```

### **Phase 1 (Prediction):**
```
+ Temporal predictor (feed-forward)
+ Sequential datasets (moving objects)
+ Prediction loss (MSE on future activations)
```

### **Phase 2 (Reasoning):**
```
+ Causal inference network
+ Action-conditioned prediction
+ Planning module (tree search)
+ RL integration
```

### **Phase 3 (Language):**
```
+ LSTM/Transformer decoder
+ Seq2seq training
+ Question answering module
+ Dialogue state tracking
```

### **Phase 4 (Abstraction):**
```
+ Hierarchical prototypes
+ Meta-learning capability
+ Analogy solver
+ Creative generator
```

---

## Success Milestones (The Tests)

### **Level 1: Toddler** ✓ **when:**
```python
# Prediction
next_state_accuracy > 0.6

# Imagination  
counterfactual_similarity > 0.55

# Object permanence
hidden_object_recall > 0.5
```

### **Level 2: Early Child** ✓ **when:**
```python
# Causality
inverse_causal_inference > 0.5

# Planning
goal_completion_rate > 0.6  # in gridworld
multi_step_success > 0.4

# Tool use
means_end_reasoning > 0.5
```

### **Level 3: Child** ✓ **when:**
```python
# Description
description_accuracy > 0.6  # BLEU/ROUGE

# QA
question_answer_accuracy > 0.7

# Dialogue
multi_turn_coherence > 0.6

# Explanation
explanation_plausibility > 0.5
```

### **Level 4: Advanced Child** ✓ **when:**
```python
# Abstraction
few_shot_concept_learning > 0.6  # 3-shot

# Analogy
analogy_solving_accuracy > 0.55

# Creativity
novel_generation_coherence > 0.6
novel_generation_diversity > 0.7
```

---

## Where To Start

**Most Impact, Fastest:**

**Option 1: Prediction (Phase 1.1)**
```bash
# Immediate foundation for everything else
# Not too hard (feed-forward net)
# Enables imagination, planning, etc.


- Implement PredictiveATL
- Generate moving object dataset
- Train overnight
- Test next morning
```

**Option 2: Language Generation (Phase 3.1)**
```bash
# Makes CHPL feel "alive"
# Can describe what it sees
# Enables dialogue later


- Implement simple LSTM decoder
- Generate caption dataset from existing scenes
- Train description generation
```

**Option 3: Planning Agent (Phase 2.2)**
```bash
# Most ambitious
# Full agent in environment
# Coolest demos


- Setup gridworld environment
- Integrate CHPL as agent
- Train with RL
```

---

## My Recommendation

**Start with Prediction (Option 1):**

**Why:**
- Unlocks everything else (planning needs prediction, imagination needs prediction)
- Quick win
- Fundamental cognitive capability
- Tests if distributed codes handle temporal structure

**Then:**
- If prediction works → Planning (Phase 2)
- Success there → Language (Phase 3)
- Build up systematically

**Timeline:**
- Prediction + Imagination
- Causality + Planning
- Language Generation
- Dialogue
- Abstraction

**Full child-level cognition.**

---

## The Implementation

Write `PredictiveATL` class.

```bash
cd ~/Dokumente/Neuroscience/CHPL-exploration
git checkout exploration-prediction

# Train
python experiments/train_predictive_atl.py

# Test
python experiments/test_temporal_prediction.py

# If works: proceed to imagination
# If fails: debug together
```

**A few stepsfrom now: CHPL thinks, reasons, imagines, speaks.**
