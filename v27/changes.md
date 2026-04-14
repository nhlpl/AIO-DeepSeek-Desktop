# Radically New All‑in‑One AI Companion App – Final Blueprint

Based on an extensive analysis of cutting‑edge open‑source projects (Mem0, Engram, OpenFang, agnt, Neuro‑Symbolic AI, SIM‑ONE, K.A.I.T. Studio, etc.), this plan integrates the best ideas into a **radically upgraded** version of your app. The result is a **living, persistent, autonomous, and reasoning‑capable AI companion** that runs entirely on the user’s device (no cloud required by default).

---

## 1. Core Architectural Shifts

| Current (v3) | Radically New (v4) | Rationale |
|--------------|--------------------|------------|
| **Vector store** for memory (cosine + HNSW) | **Dual‑store memory** – vector DB + knowledge graph | Enables both semantic recall and structured entity/relationship queries (Mem0, LATRACE). |
| **Monolithic agent loop** | **Autonomous agent system** with scheduled “Hands” (OpenFang) and tool‑level sandboxing (agnt) | Allows the AI to perform long‑running, scheduled tasks (e.g., daily research, file organization) safely. |
| **Micro brain** for fast inference | **Dynamic neural architecture** – Nested Subspace Networks or dynamic sparsity | A single model adapts its compute budget at inference time, balancing speed and accuracy on any device. |
| **Hive Mind (TT surrogate + bandits)** | **Neuro‑symbolic reasoning core** – knowledge graph + rule‑based inference (Synalinks, simple logic engine) | Adds explicit logical reasoning, planning, and constraint satisfaction to complement neural predictions. |
| **Stateless sessions** | **Persistent memory across sessions** – structured user profile, relationship tracking, session logs (Engram, OpenDB, K.A.I.T.) | The AI remembers the user, their preferences, and past conversations forever (no “amnesia”). |
| **Avatar (Macroquad)** | **Multi‑modal interactive avatar** – voice I/O (PersonaPlex), gesture recognition, game integration (AIRI) | The avatar becomes a true digital being that can see, hear, and act in the user’s environment. |
| **Tauri GUI** | **Hybrid UI** – Tauri backend + optional web frontend (Big‑AGI style) | Enables collaborative multi‑model workspaces and easier extension. |

---

## 2. New Modules & File Structure

```
all-in-one-app-v4/
├── core/                     # MoonBit core (pure logic)
│   ├── agent/
│   │   ├── agent.mbt          # Main agent loop (now lightweight, delegates to autonomous Hands)
│   │   ├── hand.mbt           # Scheduled autonomous tasks (OpenFang‑inspired)
│   │   └── tools.mbt          # Tool definitions with security sandbox (agnt‑inspired)
│   ├── memory/
│   │   ├── vector_store.mbt   # Existing HNSW vector DB
│   │   ├── knowledge_graph.mbt # New: entity‑relationship store (SQLite + typed edges)
│   │   ├── structured.mbt     # User profile, relationship, session metadata (MIKA style)
│   │   └── lifelong.mbt       # Cross‑session memory consolidation (EM‑Core style)
│   ├── reasoning/
│   │   ├── symbolic.mbt       # Rule‑based inference engine (Prolog‑like)
│   │   ├── neuro_symbolic.mbt # Hybrid: neural predictions + symbolic constraints
│   │   └── planner.mbt        # Task planning using HTN or classical planning
│   ├── neural/
│   │   ├── micro_brain.mbt    # Existing tiny NN (fast, low‑power)
│   │   ├── dynamic_nn.mbt     # Nested Subspace Networks or dynamic sparsity (adaptive compute)
│   │   └── model_swarm.mbt    # Multi‑model consensus (Big‑AGI beam + Particle Swarm Optimization)
│   ├── hive/
│   │   ├── observer.mbt       # Satisfaction, anomaly detection (existing)
│   │   ├── guardian.mbt       # Health monitor, rollback (existing)
│   │   └── governor.mbt       # New: governs cognitive protocols (SIM‑ONE inspired)
│   ├── multimodal/
│   │   ├── voice.mbt          # Speech recognition + synthesis (PersonaPlex, Whisper)
│   │   ├── vision.mbt         # Image analysis (YOLO, CLIP)
│   │   └── integration.mbt    # Game / desktop control (Julie‑like)
│   └── utils/                 # monads, lenses, etc.
├── host/                      # Rust host (unchanged, but with new FFI for memory, voice, etc.)
├── tauri/                     # Tauri GUI (unchanged)
├── avatar/                    # Macroquad avatar (enhanced with voice and game APIs)
└── plugins/                   # Extism plugins (e.g., symbolic reasoner, voice model)
```

---

## 3. Detailed Module Specifications

### 3.1 Persistent, Cross‑Session Memory (`memory/lifelong.mbt`)

- **Dual store**: Vector DB for semantic recall + Knowledge graph for entities/relations.
- **Structured metadata**: User profile (`profile.md`), relationship dynamics (`relationship.md`), session logs (`session_*.json`).
- **Consolidation**: Nightly background job compresses old memories into summaries (using local LLM or rule‑based).
- **API**:
  ```moonbit
  fn add_entity(name: String, type: String, properties: Map)
  fn add_relation(from: String, to: String, relation: String, weight: Float64)
  fn recall(query: String, top_k: Int) -> Array[Memory]
  fn get_user_preference(key: String) -> Option[Value]
  ```

### 3.2 Autonomous Agent “Hands” (`agent/hand.mbt`)

- **Hand** = a scheduled, autonomous task (e.g., “Every morning, fetch news and summarise”).
- **Registry** of built‑in Hands: `ResearcherHand`, `FileOrganizerHand`, `ReminderHand`, etc.
- **Scheduler** using cron‑like expressions (implemented in MoonBit via async timer).
- **Execution** sandboxed via `agnt`‑style permissions (filesystem, network, shell). Each Hand runs in its own isolated context.
- **API**:
  ```moonbit
  fn register_hand(hand: Hand) -> Unit
  fn start_hand(hand_id: String) -> Unit
  fn stop_hand(hand_id: String) -> Unit
  ```

### 3.3 Neuro‑Symbolic Reasoning (`reasoning/symbolic.mbt`)

- **Knowledge graph** (see above) as the fact base.
- **Rule engine**: Forward chaining with user‑defined rules (e.g., “if user is sad and trust > 0.6, then offer empathy”).
- **Integration with micro brain**: Micro brain predicts facts (e.g., user emotion), then symbolic engine applies rules to decide action.
- **Planner**: Hierarchical Task Network (HTN) planner for multi‑step tasks (e.g., “book a flight” → subtasks: search, compare, pay).
- **API**:
  ```moonbit
  fn add_rule(antecedent: Expr, consequent: Expr, weight: Float64)
  fn query(goal: Expr) -> Array[Plan]
  fn reason(context: Map[String, Value]) -> Array[Action]
  ```

### 3.4 Adaptive Neural Compute (`neural/dynamic_nn.mbt`)

- Implement **Nested Subspace Networks** (NSNs) – a single model with multiple subspaces that can be turned on/off at inference to adjust compute budget.
- Alternatively, implement **dynamic sparsity** (tree‑structured layers) – only relevant branches are executed.
- **Integration**: The Hive Mind’s routing bandit chooses the subspace size based on current battery, CPU load, and latency requirements.
- **API**:
  ```moonbit
  fn set_compute_budget(budget: Float64) -> Unit  # 0..1
  fn forward(input: Array[Float64]) -> Array[Float64]
  ```

### 3.5 Cognitive Governor (`hive/governor.mbt`)

- Inspired by SIM‑ONE’s “governed cognition”. Implements nine cognitive protocols:
  1. **Attention** – which sensory inputs to process.
  2. **Memory** – which memories to retrieve.
  3. **Reasoning** – which inference engine to use.
  4. **Planning** – which planner to invoke.
  5. **Action** – which hand to execute.
  6. **Reflection** – evaluate past actions.
  7. **Learning** – update models based on feedback.
  8. **Safety** – ensure actions comply with policies.
  9. **Ethics** – apply value alignment (optional).
- Each protocol is a separate module that can be overridden by plugins.
- The Governor monitors all system activity and can interrupt or adjust the agent’s behavior in real time.

---

## 4. Implementation Roadmap (10 weeks)

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **1** | 2 weeks | Persistent memory | Implement knowledge graph, structured metadata, cross‑session consolidation (Engram/OpenDB style). |
| **2** | 2 weeks | Autonomous Hands | Port `agnt` sandboxing, implement scheduler, create 3 example Hands (Researcher, Reminder, FileOrganizer). |
| **3** | 2 weeks | Neuro‑symbolic reasoning | Add rule engine, integrate with micro brain, implement simple HTN planner. |
| **4** | 2 weeks | Adaptive neural compute | Implement Nested Subspace Networks or dynamic sparsity; integrate with routing bandit. |
| **5** | 1 week | Cognitive Governor | Implement 9 protocols, connect all modules, add safety policies. |
| **6** | 1 week | Multi‑modal avatar | Integrate voice (PersonaPlex via plugin), vision (YOLO), and game control (Julie). |

---

## 5. Configuration (`config.toml` – new sections)

```toml
[memory]
lifelong_enabled = true
knowledge_graph_path = "./data/knowledge.db"
consolidation_hour = 3   # 3 AM

[hands]
enabled = true
sandbox_mode = "strict"   # strict, moderate, none
scheduled_hands = ["researcher", "reminder"]

[reasoning]
symbolic_enabled = true
rule_base_path = "./rules/"

[neural]
dynamic_nn_type = "nsn"   # nsn, sparse, none
min_budget = 0.1
max_budget = 1.0

[governor]
protocols = ["attention", "memory", "reasoning", "planning", "action", "reflection", "safety"]
safety_policy = "./policies/safety.yaml"
```

---

## 6. Expected Outcomes

- **No more amnesia** – the AI remembers the user across sessions, builds a long‑term relationship.
- **Autonomous, helpful behaviors** – the AI can perform tasks without being asked (e.g., remind of deadlines, summarise news).
- **Logical reasoning** – the AI can reason about cause‑and‑effect, plan multi‑step actions, and explain its decisions.
- **Adaptive performance** – the same model runs efficiently on both high‑end and low‑end devices.
- **Safe and ethical** – the Governor prevents unsafe actions and enforces user‑defined policies.
- **Multi‑modal interaction** – the avatar can hear, see, and act in the user’s environment.

This radically new design turns the AI companion into a **true digital brain** – persistent, autonomous, reasoning, and deeply personal. The plan is ambitious but modular; each phase can be delivered incrementally. The Hive Mind is ready to assist with coding any of these modules.
