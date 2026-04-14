# Plan: Upgraded All‑in‑One AI Companion App – Final Architecture

This plan consolidates all advanced mathematics and engineering decisions into a **single, actionable blueprint** for the next major version of the app. The new architecture is **MoonBit‑centric**, with a pure core library cross‑compiled to native, Wasm, and JavaScript, and a thin Rust/Tauri host for system APIs and GPU acceleration.

---

## 1. High‑Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Tauri GUI (Rust + Dioxus)                 │
│  • Main window, avatar window, settings, collaboration panel│
└───────────────────────────┬─────────────────────────────────┘
                            │ (FFI / dynamic loading)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 MoonBit Core Library (pure)                  │
│  • Agent loop (async, tool calling)                         │
│  • Memory (OT retrieval, Hopfield, vector store)            │
│  • Simulation (TT surrogate, ECS, evolution)                │
│  • Personal AI (SDE mood, trust, personality)               │
│  • Smart routing (LinUCB, knapsack, hybrid search)          │
│  • Crypto (Merkle tree for integrity)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ (FFI to host functions)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Rust Host Library (static)                 │
│  • File I/O, HTTP, SQLite (knowledge base)                  │
│  • GPU compute (wgpu: matmul, TT contraction)               │
│  • Avatar process management (TCP)                          │
│  • System APIs (memory, threads)                            │
└─────────────────────────────────────────────────────────────┘
```

All advanced mathematics are implemented in the **MoonBit core** where possible, with performance‑critical kernels offloaded to Rust GPU compute.

---

## 2. Module Changes

### 2.1 MoonBit Core (`moonbit-core/`)

| Module | Changes | Mathematical Framework |
|--------|---------|------------------------|
| `agent/` | Async agent loop, tool registry, LLM routing (LinUCB) | Contextual bandit, LinUCB |
| `memory/` | OT retrieval (Sinkhorn), Hopfield network, vector store (HNSW, PQ) | Optimal transport, exponential capacity |
| `simulation/` | QTT with numoon arrays, ECS for agent‑based sim, grammar GP | Tensor train, evolutionary computation |
| `personal/` | SDE mood (valence/arousal), trust (Beta), personality (clonal selection) | Stochastic differential equations, Bayesian inference |
| `routing/` | Smart routing: knapsack token selection, RRF fusion, bandit decision | 0/1 knapsack, reciprocal rank fusion, LinUCB |
| `crypto/` | Merkle tree for data integrity | Merkle tree, hash‑based verification |

### 2.2 Rust Host Library (`host/`)

| Module | Implementation | Mathematical Optimization |
|--------|----------------|---------------------------|
| `file.rs` | Async file I/O with batch writes (EOQ) | Economic order quantity |
| `http.rs` | HTTP client with exponential backoff + jitter | Retry strategy, Poisson process |
| `gpu.rs` | wgpu compute kernels: matmul, TT contraction (Winograd, tiling) | Tiling, Winograd minimal filtering |
| `kb.rs` | SQLite FTS5 + embedding similarity (via candle) | Hybrid search (RRF) |
| `avatar.rs` | TCP communication with Macroquad process | MessagePack, throttling (30 Hz) |
| `sys.rs` | Huge pages, memory prefetching, thread affinity | Markov prefetching, buddy allocator |

### 2.3 Tauri GUI

- **Replace direct MoonBit calls** with FFI to the core library.
- **Add routing UI**: display which LLM is used, token budget, memory selection.
- **Add simulation panel** for TT surrogate and evolution control.
- **Avatar window** remains as a separate Macroquad process, receiving mood via TCP.

### 2.4 Avatar Process (Macroquad)

- **No changes** except minor improvements: SDE mood integration, gesture recognition (Reeb graph persistence).

---

## 3. Mathematical Frameworks Integrated

| Framework | Application | Component |
|-----------|-------------|-----------|
| Optimal transport (Sinkhorn) | Memory retrieval | `memory/ot.mbt` |
| Hopfield network (exponential capacity) | Associative memory | `memory/hopfield.mbt` |
| Tensor train (QTT) | Surrogate modeling | `simulation/tt.mbt` |
| Entity‑component‑system (ECS) | Agent‑based simulation | `simulation/ecs.mbt` |
| Stochastic differential equations (SDE) | Avatar mood | `personal/mood.mbt` |
| Contextual bandit (LinUCB) | LLM routing | `agent/routing.mbt` |
| 0/1 knapsack | Token‑aware memory selection | `agent/knapsack.mbt` |
| Reciprocal rank fusion (RRF) | Hybrid search | `agent/routing.mbt` |
| Grammatical evolution (GP) | Code evolution | `simulation/evolution.mbt` |
| Merkle tree | Integrity verification | `crypto/merkle.mbt` |
| Economic order quantity (EOQ) | SSD write batching | `host/file.rs` |
| Winograd minimal filtering | GPU TT contraction | `host/gpu.rs` |

---

## 4. Implementation Roadmap

### Phase 1 – Pure MoonBit Core (2 weeks)
- Set up MoonBit project with `numoon`, `moonbit/async`.
- Implement `memory/ot.mbt`, `personal/mood.mbt`, `simulation/tt.mbt` (stubs).
- Implement `agent/routing.mbt` (knapsack, RRF).
- Write unit tests.

### Phase 2 – Rust Host & FFI (2 weeks)
- Implement `host/file.rs`, `http.rs`, `kb.rs`, `avatar.rs`.
- Create FFI declarations in `moonbit-core/ffi_host.mbt`.
- Implement `gpu.rs` with wgpu kernels for matmul and TT contraction.
- Test host functions with MoonBit core.

### Phase 3 – Tauri GUI Integration (1 week)
- Load MoonBit core and Rust host static libraries.
- Expose Tauri commands that call into MoonBit core.
- Add UI elements for routing (status, token usage).
- Integrate avatar process (spawn, IPC).

### Phase 4 – Auto‑Evolution & Self‑Repair (1 week)
- Implement `simulation/evolution.mbt` (grammar GP).
- Integrate Bayesian optimization for hyperparameters (via host).
- Add self‑test and rollback mechanism.

### Phase 5 – Testing & Documentation (1 week)
- End‑to‑end tests (simulate user queries, verify routing).
- Performance benchmarks (TT eval, memory search, GPU kernel speed).
- User manual and API documentation.

---

## 5. Success Metrics

| Metric | Target |
|--------|--------|
| TT evaluation latency (D=30, r=20) | <0.5 µs (CPU) / <10 µs (GPU) |
| Memory search (1M vectors, HNSW) | <50 ms |
| Smart routing – local LLM vs DeepSeek | >90% user satisfaction, 30% token savings |
| Hybrid search (vector + FTS) | >95% recall@10 |
| SDE mood – avatar FPS | 60 FPS |
| Self‑evolution improvement | >30% speedup after 100 generations |

---

## 6. Deliverables

- Complete MoonBit core library with all advanced mathematics.
- Rust host library with system APIs, GPU kernels, and FFI.
- Tauri GUI with full integration.
- Avatar process (Macroquad) with SDE mood and gesture recognition.
- Documentation and build scripts (cross‑platform).

---

## 7. Conclusion

This plan transforms the all‑in‑one AI companion into a **modular, high‑performance, self‑evolving** system. The separation of a pure MoonBit core from a thin Rust host enables portability (Wasm, JS, native) and long‑term maintainability. All advanced mathematical frameworks – from optimal transport to contextual bandits – are integrated into the core logic. The app is now ready for final implementation. The Hive Mind stands by to assist with coding any phase.
