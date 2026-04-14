# Plan: Code Changes for the All‑in‑One AI Companion App

This plan integrates all advanced mathematics and custom implementations into the existing codebase. The changes are organized by module, with clear file paths, dependencies, and implementation steps.

---

## 1. Core MoonBit Modules (`core/`)

### 1.1 Tensor Train & Simulations

- **File:** `core/tt.mbt`
- **Changes:**
  - Implement `QTT` struct with `eval`, `mean`, `gradient`.
  - Add `error_bound` using simulated André–Quillen cohomology.
  - Add Winograd‑accelerated TT contraction kernel (GPU fallback to CPU).
- **Dependencies:** `moonbitlang/x/ndarray`, `moonbitlang/async`

### 1.2 Memory & Optimal Transport

- **File:** `core/memory/ot.mbt`
- **Changes:**
  - Implement Sinkhorn algorithm for optimal transport.
  - Add `retrieve_ot(query_emb, memories, top_k)`.
  - Integrate with Hopfield memory layer.
- **Dependencies:** `moonbitlang/x/ndarray`, `moonbitlang/rand`

### 1.3 Auto‑Layers (Performance, Error, Evolution, Adaptation)

- **Files:** `auto/performance/bayesian_opt.mbt`, `auto/error/isolation_forest.mbt`, `auto/evolution/grammar_gp.mbt`, `auto/adaptation/contextual_bandit.mbt`
- **Changes:**
  - Bayesian optimization: GP surrogate, expected improvement.
  - Isolation Forest: anomaly detection on metrics.
  - Grammar GP: CFG for MoonBit expressions, evolve memory retrieval.
  - LinUCB: contextual bandit for personalization.
- **Dependencies:** `moonbitlang/x/ndarray`, `moonbitlang/rand`

### 1.4 Knowledge Base & Semantic Search (FFI to Rust)

- **File:** `core/knowledge.mbt`
- **Changes:** FFI declarations to call Rust `KnowledgeBase` methods.
- **Dependencies:** `moonbitlang/ffi`

---

## 2. Rust Backend Modules (`src/`)

### 2.1 Knowledge Base

- **File:** `src/knowledge.rs`
- **Changes:**
  - SQLite FTS5 for full‑text search.
  - SentenceTransformer (`all-MiniLM-L6-v2`) via `candle`.
  - Hybrid fusion using optimal transport (Sinkhorn barycenter).
  - Embedding caching (bfloat16) + Markov‑chain prefetching.
- **Dependencies:** `rusqlite`, `candle-core`, `candle-transformers`, `serde_json`, `rand`

### 2.2 GPU Compute Kernels

- **Files:** `src/gpu/kernels.rs`, `src/gpu/matmul.wgsl`, `src/gpu/tt.wgsl`
- **Changes:**
  - Compute kernels for matrix multiplication, TT contraction (Winograd), batch evaluation.
  - Shared memory fusion for batch evaluation.
- **Dependencies:** `wgpu`, `bytemuck`, `encase`

### 2.3 Physics Integration

- **File:** `src/physics/engine.rs`
- **Changes:**
  - Rigid body physics with `rapier3d`.
  - Fluid simulation with `salva2d` (optional feature).
  - Deformable bodies with `del_fem` (optional feature).
- **Dependencies:** `rapier3d`, `salva2d`, `del_fem`

### 2.4 Avatar Manager

- **File:** `src/avatar_manager.rs`
- **Changes:**
  - Spawn/terminate avatar process.
  - TCP communication (MessagePack) with handshake `"READY"`.
  - Throttle state updates to 30 Hz.

### 2.5 UI & Layout

- **File:** `src/ui/layout.rs`
- **Changes:**
  - Cassowary constraint solver (`kasuari` crate).
  - Optimal transport repositioning (Hungarian algorithm).
- **Dependencies:** `kasuari`, `lap`

### 2.6 Data Visualization

- **File:** `src/ui/plots.rs`
- **Changes:**
  - Use `plotters` for charts (fitness evolution, memory landscape, TT heatmaps).
- **Dependencies:** `plotters`

### 2.7 Tauri Commands

- **File:** `src/main.rs`
- **Changes:**
  - Expose new commands: `run_evolution`, `semantic_search`, `optimize_resources`, `update_avatar`.

---

## 3. Avatar Standalone (Macroquad)

### 3.1 Mood SDE

- **File:** `avatar/src/mood_sde.rs`
- **Changes:** Euler–Maruyama integration of valence/arousal SDE.

### 3.2 Gesture Recognition

- **File:** `avatar/src/gesture_reeb.rs`
- **Changes:** Reeb graph persistence (simplified) for gesture classification.

### 3.3 Fractal Tree Renderer

- **File:** `avatar/src/fractal_tree.rs`
- **Changes:** L‑system generator, color mapping from mood, analytic antialiasing.

---

## 4. Configuration & Build

### 4.1 `Cargo.toml` (Workspace)

- Add dependencies: `rusqlite`, `candle-core`, `candle-transformers`, `wgpu`, `rapier3d`, `salva2d`, `del_fem`, `kasuari`, `lap`, `plotters`, `serde_json`, `rand`, `bytemuck`, `encase`.

### 4.2 `moon.mod.json`

- Add dependencies for MoonBit modules: `moonbitlang/x`, `moonbitlang/async`, `moonbitlang/rand`.

### 4.3 Build Script (`build.rs`)

- Compile MoonBit core to native library (`libcore.a`).
- Set `RUSTFLAGS` to link against MoonBit library.

### 4.4 Configuration File (`~/.bit/config.toml`)

- Sections: `[auto]`, `[resource]`, `[monitoring]`, `[collaboration]`, `[multi_lingual]`, `[federated]`.

---

## 5. Implementation Order (Phases)

| Phase | Focus | Files | Duration |
|-------|-------|-------|----------|
| **1** | Core MoonBit modules | `tt.mbt`, `ot.mbt`, `bayesian_opt.mbt` | 3 days |
| **2** | Knowledge base & GPU kernels | `knowledge.rs`, `gpu/kernels.rs` | 4 days |
| **3** | Auto‑layers & evolution | `isolation_forest.mbt`, `grammar_gp.mbt`, `contextual_bandit.mbt` | 3 days |
| **4** | Avatar & physics | `avatar/` (mood, gesture, tree), `physics/engine.rs` | 4 days |
| **5** | UI & visualization | `ui/layout.rs`, `ui/plots.rs`, Tauri commands | 3 days |
| **6** | Integration & testing | All modules, end‑to‑end tests | 3 days |

---

## 6. Testing Plan

- **Unit tests:** For TT evaluation, OT retrieval, GP evolution.
- **Integration tests:** Simulate API calls, avatar communication, GPU kernels.
- **Property‑based tests:** Using `proptest` for invariants (e.g., TT symmetry).
- **Performance benchmarks:** `criterion` for TT eval, search latency.

---

## 7. Documentation

- **API docs:** `cargo doc` for Rust, `moon doc` for MoonBit.
- **User manual:** Markdown files in `docs/`.
- **Developer guide:** How to add new plugins, auto‑layers, or physics components.

---

## 8. Deliverables

- Complete MoonBit core with QTT, OT, auto‑layers.
- Rust backend with knowledge base, GPU kernels, physics, avatar manager.
- Avatar standalone with SDE mood, gesture recognition, fractal tree.
- Tauri GUI with responsive layout, plots, and all commands.
- Configurable auto‑optimization, self‑healing, and federated learning.

This plan transforms the prototype into a **production‑ready, mathematically advanced, and highly performant** AI companion. The Hive Mind is ready to assist with coding any specific module.
