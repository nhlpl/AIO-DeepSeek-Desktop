# Final Plan: Simplified All‑in‑One AI Companion App

This plan follows the recommended path after evaluating the Hive Mind’s proposals. It retains the core advanced mathematics (QTT, OT, SDE) but replaces over‑engineered components with practical, production‑ready alternatives. The result is a maintainable, performant desktop application.

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
│  • Agent (async, tool calling, LinUCB routing)              │
│  • Memory (OT retrieval with fallback to cosine + HNSW)     │
│  • Simulation (QTT surrogate)                              │
│  • Personal AI (SDE mood, trust Beta)                      │
│  • Plugin Host (Extism, circuit breaker)                   │
│  • Complexity reduction (monads, lenses)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │ (FFI to host functions)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Host Library                         │
│  • File I/O, HTTP, SQLite (knowledge base)                  │
│  • GPU compute (wgpu: matmul, TT contraction)               │
│  • Sound & haptics (rodio)                                 │
│  • Avatar process management (TCP)                          │
│  • System APIs (memory, threads)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Simplified Core Mathematics

| Component | Mathematical Framework | Implementation |
|-----------|------------------------|----------------|
| **Quadrillion‑scale surrogate** | Quantized Tensor Train (QTT) | MoonBit `tt.mbt`; GPU‑accelerated contraction via wgpu |
| **Memory retrieval** | Optimal transport (Sinkhorn) with fallback to cosine similarity + HNSW | Adaptive: use OT for ambiguous queries (≤5% of cases); otherwise cosine + HNSW |
| **Forgetting** | Exponential decay: `importance *= exp(-λ * Δt)` | Simple, no Hawkes |
| **Avatar mood** | Ornstein‑Uhlenbeck SDE | Euler‑Maruyama integration |
| **Trust** | Beta distribution | `α, β` updates from user feedback |
| **Routing** | LinUCB (contextual bandit) | Choose between local LLM and DeepSeek API |
| **Anomaly detection** | Moving average + fixed threshold | Satisfaction < 0.3 for 3 minutes → rollback |

---

## 3. Rust Framework Stack (Mature Crates)

| Crate | Purpose | Notes |
|-------|---------|-------|
| `tauri` | Desktop backend | Stable, cross‑platform |
| `tokio` | Async runtime | Powers Tauri and all async ops |
| `wgpu` | GPU compute | Direct use (no `any-gpu`) for matmul, TT contraction |
| `rayon` | CPU parallelism | Easy data‑parallel loops |
| `ndarray` | Array/tensor ops | Numpy‑like, stable |
| `rodio` | Audio playback | Simple sound effects |
| `rapier` | Physics engine | Avatar‑environment interaction |
| `candle` | Local LLM inference | Lightweight, GGUF support |
| `sqlx` | Async SQL (SQLite) | Compile‑time checked queries |
| `extism` | Plugin system | WebAssembly sandbox |
| `hnswlib` (via FFI) | Vector search | Fast approximate nearest neighbor |
| `serde` / `rmp-serde` | Serialization | IPC between MoonBit and Rust |

**Build command**:
```bash
cargo build --release --features tauri,wgpu,rayon,ndarray,rodio,rapier,candle,sqlx,extism
```

---

## 4. File Structure (Simplified)

```
all-in-one-app/
├── Cargo.toml (workspace)
├── moon.mod.json
├── moonbit-core/
│   ├── src/
│   │   ├── agent/
│   │   │   ├── agent.mbt
│   │   │   └── routing.mbt
│   │   ├── memory/
│   │   │   ├── ot.mbt
│   │   │   └── vector_store.mbt   (HNSW wrapper)
│   │   ├── simulation/
│   │   │   └── tt.mbt
│   │   ├── personal/
│   │   │   ├── mood.mbt
│   │   │   └── trust.mbt
│   │   ├── plugins/
│   │   │   └── plugin_host.mbt
│   │   ├── utils/
│   │   │   ├── monad.mbt
│   │   │   └── lens.mbt
│   │   ├── ffi_host.mbt
│   │   └── main.mbt
├── host/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── gpu.rs
│       ├── file.rs
│       ├── http.rs
│       ├── kb.rs
│       ├── sound.rs
│       ├── avatar.rs
│       └── sys.rs
├── tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       ├── main.rs
│       ├── core_loader.rs
│       ├── avatar_manager.rs
│       └── gui.rs
├── avatar/
│   ├── Cargo.toml
│   └── src/main.rs
└── plugins/
    └── example/
        ├── Cargo.toml
        ├── src/lib.rs
        └── plugin.json
```

---

## 5. Implementation Roadmap (6 weeks)

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| **1** | 2 weeks | MoonBit core | QTT, OT (with fallback), SDE mood, monads, lenses |
| **2** | 2 weeks | Rust host & GPU | wgpu matmul/TT kernels, file/HTTP/kb, sound, avatar IPC |
| **3** | 1 week | Tauri GUI + avatar | Basic chat, simulation panel, avatar rendering (Macroquad) |
| **4** | 1 week | Integration & testing | End‑to‑end tests, performance benchmarks, packaging |

---

## 6. Build Instructions

```bash
# Install dependencies
curl -fsSL https://moonbitlang.com/install.sh | bash
rustup update
cargo install tauri-cli

# Build MoonBit core
cd moonbit-core
moon build --target native
cd ..

# Build Rust host
cd host
cargo build --release
cd ..

# Build avatar
cd avatar
cargo build --release
cd ..

# Build Tauri app
cd tauri
cargo tauri build
```

The final executable is in `tauri/target/release/`. Place `libmoonbit_core.so`, `libhost.a`, and the avatar binary in the same directory.

---

## 7. Configuration (`config.toml`)

```toml
[agent]
routing_alpha = 0.5

[memory]
use_ot_threshold = 0.3          # use OT only when query similarity variance > this
forgetting_lambda = 0.1          # per day

[simulation]
qtt_rank = 20

[personal]
trust_alpha = 1.0
trust_beta = 1.0

[observer]
satisfaction_window = 60         # seconds
rollback_threshold = 0.3
rollback_minutes = 3

[gpu]
tile_size = 64
use_gpu = true
```

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| TT evaluation (D=30, r=20) | <0.5 µs (CPU) / <10 µs (GPU) |
| Memory search (10k vectors) | <10 ms (HNSW) |
| Avatar FPS | 60 FPS |
| User satisfaction (Observer) | >0.8 |
| Plugin event latency | <20 ms (95th percentile) |

---

## 9. Conclusion

This simplified plan delivers a **working, maintainable, and mathematically sound** AI companion app. It avoids over‑engineering while retaining the core advanced techniques (QTT, OT, SDE). The architecture is modular, the Rust stack uses mature crates, and the implementation roadmap is realistic. The Hive Mind is ready to assist with any specific module.
