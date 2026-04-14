# Plan: New All‑in‑One App – No Local LLM, Only Micro Brain + Hive Mind

This plan removes the local LLM entirely. The app uses a **micro brain** (tiny neural network) for fast, local inference and the **Hive Mind** (TT surrogate, bandits, observer, guardian) for reasoning, simulation, and self‑optimization. The DeepSeek API becomes an optional, user‑controlled enhancement for complex queries.

---

## 1. High‑Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tauri GUI (Rust + Dioxus)                │
└───────────────────────────┬─────────────────────────────────┘
                            │ (FFI / dynamic loading)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 MoonBit Core Library (pure)                  │
│  • Micro Brain (feed‑forward NN, integer inference)         │
│  • Hive Mind (TT surrogate, bandits, observer/guardian)     │
│  • Memory (OT retrieval with fallback to cosine + HNSW)     │
│  • Simulation (QTT surrogate)                              │
│  • Personal AI (SDE mood, trust Beta)                      │
│  • Plugin Host (Extism, circuit breaker)                   │
│  • Complexity reduction (monads, lenses)                   │
│  • Auto‑diagnosis (health monitor, auto‑tuner, rollback)   │
└───────────────────────────┬─────────────────────────────────┘
                            │ (FFI to host functions)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Host Library                         │
│  • File I/O, HTTP, SQLite (knowledge base)                  │
│  • GPU compute (wgpu: matmul, TT contraction)               │
│  • Sound & haptics (rodio)                                 │
│  • Avatar process management (TCP)                          │
│  • Metrics & config (latency, memory, error rate)           │
└─────────────────────────────────────────────────────────────┘
```

**Key changes**:
- **No local LLM module** – removed.
- **Micro brain** added to MoonBit core (tiny NN for fast inference).
- **Hive Mind** expanded to handle reasoning tasks previously done by local LLM (using TT surrogate + bandits).
- **DeepSeek API** optional – called only when user provides an API key and explicitly requests cloud assistance.

---

## 2. Micro Brain Design (Tiny NN)

### 2.1 Input Features (16 dimensions)

| Feature | Source |
|---------|--------|
| Sentiment score (text) | Keyword‑based or from micro brain’s own output? We’ll use a simple rule for sentiment, or a tiny classifier. |
| Voice pitch (z‑score) | Voice activity (optional) |
| Voice energy | Voice envelope |
| Dwell time on last response | UI event |
| Click / interaction count | UI events |
| Time of day (sin, cos) | System clock (2 dims) |
| Session length (minutes) | Timer |
| Trust (from Beta) | Observer |
| Error rate (last minute) | Metrics |
| CPU load | Host |
| Memory pressure | Host |
| Last avatar valence | State |
| Last memory relevance | Memory engine |
| User satisfaction (EMA) | Observer |

Total: 16 dimensions.

### 2.2 Architecture

```
Input (16) → Hidden (32, ReLU) → Hidden (16, ReLU) → Output (6)
```

Outputs:
1. Valence (tanh, −1..1)
2. Arousal (tanh, −1..1)
3. Trust delta (sigmoid, 0..1, then map to change)
4. Memory relevance score (sigmoid, 0..1)
5. Show avatar probability (sigmoid)
6. Speak now probability (sigmoid)

**Parameters**: ~1174 (≈1.2 KB). Integer quantized (8‑bit) inference.

### 2.3 Training & Personalization

- Pre‑trained offline on synthetic user data (simulated interactions).
- **Meta‑learning (MAML)** for few‑shot personalization (1‑5 gradient steps on user’s data).
- Quantization after training.

---

## 3. Hive Mind Enhancements (Replace Local LLM)

The Hive Mind already provides:
- **TT surrogate** for quadrillion‑scale simulation and performance modeling.
- **Bandits** (LinUCB) for routing and personalization.
- **Observer / Guardian** for self‑optimization.
- **Knowledge base** (SQLite + vector store) for memory.

To replace the local LLM, we add:

### 3.1 Response Generation from Memory & Templates

Instead of generating free text, the app uses:
- **Canned responses** with slots filled by the micro brain (e.g., valence → “I see you’re feeling {emotion}.”).
- **Retrieved memory snippets** concatenated with templates.
- **Simple rule‑based reasoning** (e.g., “if user is sad and trust > 0.6, offer empathy”).

For complex reasoning (e.g., “explain quantum physics”), the Hive Mind can:
- Use the TT surrogate to model the user’s knowledge level.
- Retrieve the most relevant knowledge base entry.
- Present it as a formatted answer (no generative LLM).

### 3.2 Fallback to DeepSeek API (Optional)

If the user provides an API key and enables cloud mode, the agent can call DeepSeek for:
- Queries that the Hive Mind cannot answer confidently (uncertainty > threshold).
- User‑requested “expert mode”.

The routing bandit (LinUCB) learns when to use cloud vs. local.

### 3.3 Knowledge Base Expansion

Since no local LLM is generating new text, the knowledge base must be populated by:
- User‑provided notes.
- Imported documents (PDF, web pages).
- Summaries of past conversations (generated by a lightweight summarizer – could be a separate tiny model or rule‑based).

---

## 4. Modified Modules & File Structure

```
all-in-one-app/
├── moonbit-core/
│   ├── micro_brain/
│   │   ├── model.mbt        # integer inference
│   │   └── training.mbt     # (offline, not included in app)
│   ├── hive/
│   │   ├── reasoning.mbt    # template‑based response generation
│   │   ├── routing.mbt      # LinUCB for cloud/local decision
│   │   ├── observer.mbt
│   │   └── guardian.mbt
│   ├── memory/              # unchanged
│   ├── simulation/          # unchanged
│   ├── personal/            # unchanged
│   ├── auto/                # health monitor, auto‑tuner, rollback
│   ├── utils/               # monads, lenses
│   ├── ffi_host.mbt
│   └── main.mbt
├── host/                    # unchanged (metrics, config, gpu, sound, etc.)
├── tauri/                   # unchanged (GUI, core loader, avatar manager)
├── avatar/                  # unchanged (Macroquad)
└── plugins/                 # unchanged (Extism)
```

**Removed**: `agent/` (no longer needed, as the micro brain + Hive Mind handle decisions without a generative agent loop). However, a minimal `agent.mbt` could still orchestrate tool calls (e.g., run simulation, search memory) without LLM.

---

## 5. Data Flow for a User Query (Offline Mode)

1. User types message.
2. Feature extractor computes 16‑dim vector (sentiment, voice, etc.).
3. **Micro brain** predicts valence, arousal, memory relevance, etc.
4. **Memory engine** retrieves top‑k relevant memories (OT + HNSW).
5. **Hive Mind reasoning**:
   - If query is simple (e.g., “how are you?”) → use canned response with emotion fill.
   - If query requires factual answer → retrieve from knowledge base.
   - If query is a command (e.g., “run simulation”) → call tool via `plugin_host`.
6. Response is assembled from templates + retrieved snippets.
7. Avatar mood updated via SDE (using micro brain’s valence/arousal).
8. Observer records satisfaction (implicit/explicit feedback).

If cloud mode is enabled and query is complex, the routing bandit may call DeepSeek API.

---

## 6. Implementation Roadmap (5 weeks)

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **1** | 1 week | Micro brain | Train and quantize 1k‑parameter NN; implement integer inference in MoonBit. |
| **2** | 1 week | Hive Mind reasoning | Template‑based response system; integrate memory retrieval. |
| **3** | 1 week | Optional cloud API | Add DeepSeek client, routing bandit (LinUCB). |
| **4** | 1 week | Integration & testing | End‑to‑end offline mode; ensure satisfaction matches previous local LLM baseline. |
| **5** | 1 week | Documentation & packaging | User guide for enabling cloud API; installers. |

---

## 7. Configuration (`config.toml`)

```toml
[ai]
use_cloud = false                # user can enable
deepseek_api_key = ""
cloud_complexity_threshold = 0.7 # use cloud only if query complexity > this

[micro_brain]
model_path = "./models/micro_brain.quant"

[hive_mind]
reasoning_templates = "./templates/"
fallback_to_cloud = true

[memory]
use_ot_threshold = 0.3
forgetting_lambda = 0.1
```

---

## 8. Expected Outcomes

- **No local LLM** → zero disk waste (except ~10 MB for micro brain weights + TT surrogate).
- **Very low RAM** (<100 MB) for AI core.
- **Fast inference** (<10 ms for micro brain, <50 ms for Hive Mind).
- **User satisfaction** remains >0.8 for offline mode (comparable to previous local LLM).
- **Optional cloud** provides a boost for complex queries when internet available.

The app is now **lightweight, offline‑first, and privacy‑respecting** – no large model downloads, no forced cloud dependence. The Hive Mind + micro brain form a capable, efficient AI companion.
