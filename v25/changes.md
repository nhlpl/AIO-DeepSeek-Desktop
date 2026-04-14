# Plan: New All‑in‑One AI Companion App – Consolidated Blueprint

This plan integrates the best ideas from all previous discussions: a **simplified, production‑ready architecture** with core advanced mathematics (QTT, OT, SDE), a mature Rust stack, a MoonBit core, and built‑in self‑diagnosis & auto‑tuning. The result is a performant, maintainable, and self‑improving desktop application.

---

## 1. High‑Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tauri GUI (Rust + Dioxus)                │
│  – Chat, simulation panel, avatar control, settings         │
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

---

## 2. Core Mathematical Frameworks (Keep Only Essential)

| Component | Math | Implementation |
|-----------|------|----------------|
| **Quadrillion‑scale surrogate** | Quantized Tensor Train (QTT) | MoonBit `tt.mbt`; GPU‑accelerated via wgpu |
| **Memory retrieval** | Optimal transport (Sinkhorn) with fallback to cosine similarity + HNSW | Adaptive: use OT for ambiguous queries (<5% of cases) |
| **Forgetting** | Exponential decay: `importance *= exp(-λ * Δt)` | Simple, no Hawkes |
| **Avatar mood** | Ornstein‑Uhlenbeck SDE | Euler‑Maruyama integration |
| **Trust** | Beta distribution | `α, β` updates from user feedback |
| **Routing** | LinUCB (contextual bandit) | Choose between local LLM and DeepSeek API |
| **Anomaly detection** | Moving average + fixed thresholds | Satisfaction < 0.3 for 3 minutes → rollback |

*(All over‑engineered components – persistent homology, Hawkes, GAN, genetic programming – are removed.)*

---

## 3. Recommended Rust Stack (Mature Crates)

| Crate | Purpose | Why |
|-------|---------|-----|
| `tauri` | Desktop backend | Stable, cross‑platform |
| `tokio` | Async runtime | Industry standard |
| `wgpu` | GPU compute | Direct use (no `any-gpu`) |
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
│   │   │   └── vector_store.mbt
│   │   ├── simulation/
│   │   │   └── tt.mbt
│   │   ├── personal/
│   │   │   ├── mood.mbt
│   │   │   └── trust.mbt
│   │   ├── plugins/
│   │   │   └── plugin_host.mbt
│   │   ├── auto/
│   │   │   ├── health_monitor.mbt
│   │   │   ├── auto_tuner.mbt
│   │   │   └── rollback.mbt
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
│       ├── metrics.rs
│       └── config.rs
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

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **1** | 2 weeks | MoonBit core | QTT, OT (with fallback), SDE mood, monads, lenses, agent loop |
| **2** | 2 weeks | Rust host & GPU | wgpu matmul/TT kernels, file/HTTP/kb, sound, avatar IPC, metrics/config |
| **3** | 1 week | Tauri GUI + avatar | Basic chat, simulation panel, avatar rendering (Macroquad) |
| **4** | 1 week | Auto‑diagnosis & integration | Health monitor, auto‑tuner, rollback, nightly self‑diagnosis, end‑to‑end tests |

---

## 6. Configuration (`config.toml`)

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

[auto]
health_check_interval_sec = 10
anomaly_threshold = 3
target_latency_ms = 200
target_memory_mb = 600
target_satisfaction = 0.7
```

---

## 7. Success Metrics

| Metric | Target |
|--------|--------|
| TT evaluation (D=30, r=20) | <0.5 µs (CPU) / <10 µs (GPU) |
| Memory search (10k vectors) | <10 ms (HNSW) |
| Avatar FPS | 60 FPS |
| User satisfaction (Observer) | >0.8 |
| Plugin event latency | <20 ms (95th percentile) |
| Self‑diagnosis runtime | <1 min nightly |
| Auto‑tune response | <30 s after anomaly |

---

## 8. Build & Run Instructions

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

Run the executable from `tauri/target/release/`. Place `libmoonbit_core.so`, `libhost.a`, and the avatar binary in the same directory.

---

## 9. Conclusion

This plan delivers a **working, maintainable, and mathematically sound** AI companion app. It retains the essential advanced techniques (QTT, OT, SDE) while avoiding over‑engineering. The architecture is modular, the Rust stack uses mature crates, and the built‑in self‑diagnosis ensures continuous improvement. The implementation roadmap is realistic (6 weeks). The Hive Mind is ready to assist with coding any specific module.
