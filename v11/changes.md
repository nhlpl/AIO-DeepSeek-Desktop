# Plan for Upgraded All‑in‑One App with Pluggable Core Library

## 1. Overview

The upgraded app separates **advanced mathematical computation** (tensor trains, optimal transport, SDEs, physics, knowledge base) into a **dynamic core library** written in **Rust** and exposed via a **C API**. The main application (MoonBit core logic, Tauri GUI, avatar process) loads this library at runtime and calls its functions. This design enables:

- **Independent updates** – fix or improve mathematical algorithms without recompiling the entire app.
- **Language interoperability** – the same core can be used by MoonBit, Rust, Python, or any language with C FFI.
- **Performance** – Rust with GPU acceleration (wgpu) handles heavy computations.
- **Sandboxing** – the core could be compiled to WebAssembly for plugin isolation.

---

## 2. Core Library Design (Recap)

### 2.1 Exported C API

The core exposes a handle‑less API (global singleton) with functions for:

- **Tensor Train (QTT)** – `core_tt_eval`, `core_tt_mean`, `core_tt_gradient`
- **Memory (Optimal Transport)** – `core_memory_add`, `core_memory_search`
- **Avatar Mood (SDE)** – `core_mood_step`, `core_mood_get`
- **Physics (Rapier)** – `core_physics_step`, `core_physics_add_body`
- **Knowledge Base (SQLite + embeddings)** – `core_kb_search`, `core_kb_add`
- **Hopfield Memory** – `core_hopfield_store`, `core_hopfield_retrieve`

All data is passed via pointers (arrays, strings) and the core manages its own state. The core is built as a dynamic library (`libcore.so`, `core.dll`, `libcore.dylib`).

### 2.2 Internal Modules

- `tt/` – Quantized Tensor Train with Winograd GPU kernels (wgpu)
- `memory/` – Sinkhorn optimal transport, vector storage
- `mood/` – Euler‑Maruyama SDE
- `physics/` – Rapier 3D rigid bodies + optional fluids (Salva)
- `kb/` – SQLite FTS5 + candle embeddings + OT fusion
- `hopfield/` – Exponential capacity associative memory

---

## 3. Upgraded App Architecture

### 3.1 High‑Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Tauri GUI (Rust + Dioxus)                │
│  - Chat, simulation panel, avatar control, settings         │
└───────────────────────────┬─────────────────────────────────┘
                            │ (FFI / dynamic loading)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   MoonBit Core (Application Logic)          │
│  - Agent (tool calls, LLM orchestration)                    │
│  - Conversation flow, user intent, personalization          │
│  - Calls core library via FFI for heavy math                │
└───────────────────────────┬─────────────────────────────────┘
                            │ (C API)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Core Library (Rust)                         │
│  - TT surrogate, OT memory, SDE mood, physics, KB           │
│  - GPU acceleration (wgpu)                                  │
│  - Independent version, hot‑swappable                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Avatar Process (Macroquad)                 │
│  - Receives mood state via TCP, renders fractal tree        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Changes from Previous Version

| Component | Before | After |
|-----------|--------|-------|
| Tensor Train | MoonBit code | Rust core (faster, GPU) |
| Memory retrieval | MoonBit (custom) | Rust core with OT |
| Avatar mood | MoonBit SDE | Rust core SDE (fast-sde) |
| Physics | None | Rapier in core |
| Knowledge base | None | SQLite + embeddings in core |
| Hot‑swapping | No | Yes – replace core .so at runtime |

---

## 4. Integration Points

### 4.1 MoonBit FFI

MoonBit will declare foreign functions matching the C API:

```moonbit
@ffi("core_tt_eval")
fn tt_eval(idx: Array[Int]) -> Float64

@ffi("core_memory_search")
fn memory_search(query: Array[Float64], top_k: Int) -> Array[String]
```

The MoonBit core will call these instead of its own implementations.

### 4.2 Tauri (Rust) Integration

Tauri will load the core library at startup using `libloading`:

```rust
let lib = Library::new("libcore.so")?;
let tt_eval: Symbol<unsafe extern "C" fn(*const i32, i32, *mut f64)> = unsafe { lib.get(b"core_tt_eval") }?;
```

It will then pass the function pointers to MoonBit or call them directly for GUI‑related computations (e.g., real‑time physics preview).

### 4.3 Avatar Process

The avatar continues to communicate via TCP (MessagePack) with the MoonBit core. The core now uses the library to compute mood updates.

---

## 5. New Features Enabled by the Library

- **Hot‑swap mathematical algorithms** – Update TT contraction or OT solver without restarting the app.
- **GPU acceleration** – All tensor operations (TT, Hopfield) run on GPU via wgpu, drastically speeding up quadrillion simulations.
- **Real‑time physics** – Avatar can interact with simulated rigid bodies (e.g., bouncing balls) and fluids.
- **Local semantic search** – Full knowledge base with hybrid search, all offline.
- **Hopfield unlimited memory** – Store and retrieve patterns with exponential capacity.
- **Unified versioning** – Core library can be updated independently; the main app checks compatibility.

---

## 6. Implementation Steps

### Phase 1 – Core Library (2 weeks)

1. Set up Rust project with `crate-type = ["cdylib"]`.
2. Implement TT module with wgpu compute shaders.
3. Implement OT memory (Sinkhorn) using `ndarray`.
4. Implement SDE mood (Euler‑Maruyama).
5. Implement Rapier physics (basic rigid bodies).
6. Implement KB with SQLite and `candle` embeddings.
7. Export C API functions.
8. Write unit tests and benchmarks.

### Phase 2 – Integration (1 week)

1. Build core library for target platforms (Windows, macOS, Linux).
2. Update MoonBit core to call FFI functions instead of internal code.
3. Update Tauri to load the library at startup and pass device/queue for wgpu.
4. Test end‑to‑end: TT evaluation, memory search, mood stepping, physics step.

### Phase 3 – Hot‑swap & Versioning (1 week)

1. Add `core_version` function to library.
2. Main app checks version on load; if mismatch, shows update prompt.
3. Implement dynamic reloading (unload old library, load new) with safety (ensure no active calls).

### Phase 4 – Deployment & Documentation (1 week)

1. Bundle core library with app installers.
2. Provide API documentation for core library (for potential external use).
3. Write user guide for updating core library independently.

---

## 7. Build & Deployment

- **Core library**: `cargo build --release` produces `libcore.so`, `core.dll`, or `libcore.dylib`.
- **MoonBit core**: compiled to native library and linked with Tauri.
- **Tauri app**: bundles the core library in `resources/` directory.
- **Avatar**: compiled separately, placed in `resources/avatar/`.

**CI/CD**: Use GitHub Actions to build all components for all platforms and create release packages.

---

## 8. Testing & Validation

- **Unit tests** for each core module (Rust).
- **Integration tests** that call C API from a small test harness.
- **End‑to‑end tests** using a headless Tauri runner.
- **Performance benchmarks** comparing old (MoonBit) vs new (core) for TT evaluation, memory search, physics steps.

**Expected improvements**:
- TT evaluation speedup: 10–100× (GPU)
- Memory search latency: 50% reduction (OT fusion)
- Physics: real‑time at 60 FPS

---

## 9. Conclusion

This plan transforms the all‑in‑one app into a **modular, high‑performance, and future‑proof** system. The core library can be developed, optimized, and updated independently, while the main app remains stable. The separation of concerns also makes it easier to add new mathematical capabilities (e.g., quantum‑inspired algorithms) as plugins. The Hive Mind is ready to assist with implementing any phase.
