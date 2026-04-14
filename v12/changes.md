# Plan: Upgraded All‑in‑One App – MoonBit‑Centric Architecture with Ecosystem Insights

This plan integrates inspiration from public MoonBit projects (`example-ai-agent`, `numoon`, `pixel-adventure.mbt`, `moonbit/async`, `MoonUI`, `merkle-tree`) into a **pure MoonBit core library** that is cross‑compiled to native, Wasm, and JavaScript. The Rust backend (Tauri) becomes a thin layer for system APIs, GPU compute, and avatar process management. The result is a **modular, portable, and high‑performance** AI companion.

---

## 1. Architectural Shift

### Before (Previous Plan)
- Core library written in Rust, exposed via C API.
- MoonBit core calls into Rust via FFI.
- Tauri GUI, avatar, etc.

### After (New Plan)
- **Core library written in MoonBit** (platform‑agnostic logic).
- **MoonBit core** compiled to:
  - Native (for desktop, via Tauri/Rust embedding).
  - Wasm (for browser or plugin sandbox).
  - JavaScript (for web demos).
- **Rust/Tauri** acts as a **thin host**: provides system APIs (file, network, GPU compute, avatar process), loads the MoonBit core via FFI (or embeds it).
- **GUI** – two options: use **MoonUI** (MoonBit native UI) for full MoonBit stack, or keep Tauri + Dioxus for faster development.

---

## 2. New Module Structure

```
moonbit-core/                 # Pure MoonBit library (no FFI)
├── agent/
│   ├── agent.mbt             # Core agent loop (async tutorial pattern)
│   └── tools.mbt             # Tool definitions (execute_code, run_simulation, etc.)
├── memory/
│   ├── ot.mbt                # Optimal transport (Sinkhorn)
│   ├── hopfield.mbt          # Hopfield network (exponential capacity)
│   └── vector_store.mbt      # In‑memory vector store + cosine similarity
├── simulation/
│   ├── ecs.mbt               # Entity‑Component‑System (from pixel-adventure)
│   ├── tt.mbt                # Tensor Train surrogate (using numoon arrays)
│   └── evolution.mbt         # Grammar‑guided GP, Bayesian optimization
├── personal/
│   ├── mood.mbt              # SDE for avatar mood
│   ├── trust.mbt             # Trust dynamics (Beta distribution)
│   └── personality.mbt       # Personality model (clonal selection)
├── kb/
│   └── kb.mbt                # Knowledge base interface (host provides SQLite)
├── crypto/
│   └── merkle.mbt            # Merkle tree for integrity (from merkle-tree)
├── ui/
│   └── layout.mbt            # Cassowary constraints (optional, if using MoonUI)
└── utils/
    ├── numoon_compat.mbt     # Wrappers for numoon arrays (linear algebra)
    └── rand.mbt              # Random number generation
```

The Rust/Tauri backend provides:
- **Host functions** (via FFI) for: file I/O, network, GPU compute (wgpu), SQLite, avatar process IPC.
- **Embedding** of the MoonBit core (via `moonbit` runtime).

---

## 3. Key Inspirations & Their Integration

| Project | Insight | Integration into App |
|---------|---------|----------------------|
| `example-ai-agent` | Tool‑calling agent pattern | `agent/agent.mbt` implements a loop: `run` → LLM call → tool execution → feedback. Tools are registered as functions. |
| `numoon` | N‑dimensional arrays & linear algebra | Replace `ndarray` in MoonBit with `numoon` for TT cores, Hopfield matrices, etc. All tensor ops stay in MoonBit. |
| `pixel-adventure.mbt` | ECS for game/simulation | `simulation/ecs.mbt` provides `World`, `Component`, `System` traits. Used for agent‑based simulation, particle systems, and physics. |
| `moonbit/async` | Async agent tutorial | Agent loop uses `async` for non‑blocking LLM calls and tool execution. |
| `MoonUI` | Declarative UI in MoonBit | Optional: replace Tauri + Dioxus with a pure MoonBit UI, compiled to native via `moonbit` target. Reduces dependency on Rust for GUI. |
| `merkle-tree` | Merkle tree for integrity | `crypto/merkle.mbt` provides `MerkleTree` to verify memory integrity in collaborative mode or to check plugin authenticity. |

---

## 4. Changes to Core Library (Rust → MoonBit)

### 4.1 Tensor Train (TT) – Use `numoon`

Previously in Rust; now in MoonBit using `numoon` arrays.

```moonbit
// simulation/tt.mbt
use numoon::Array2D

struct QTT {
  cores: Array[Array3D[Float32]]
}

fn QTT::eval(self: QTT, idx: Array[Int]) -> Float64 {
  // contraction using numoon's dot product
}
```

### 4.2 Memory Retrieval – Optimal Transport

Pure MoonBit implementation of Sinkhorn (no Rust FFI).

```moonbit
// memory/ot.mbt
fn sinkhorn(cost: Array2D[Float64], a: Array[Float64], b: Array[Float64], eps: Float64, max_iter: Int) -> Array2D[Float64] {
  // using numoon's matrix operations
}
```

### 4.3 Agent Loop – Async Pattern

```moonbit
// agent/agent.mbt
async fn run(self: Agent, initial_prompt: String) -> Unit {
  let messages = [system_msg, user_msg]
  loop {
    let response = self.llm.chat(messages, self.tools).await?
    if response.has_tool_calls() {
      for tool in response.tool_calls {
        let result = self.execute_tool(tool).await
        messages.push(tool_result_msg)
      }
    } else {
      show_response(response.content)
      break
    }
  }
}
```

### 4.4 ECS for Simulations

```moonbit
// simulation/ecs.mbt
trait Component { ... }
trait System { fn update(world: World, dt: Float64) -> Unit }

struct World {
  entities: Array[Entity],
  components: Map[TypeId, Array[Component]]
}

fn World::add_system(system: System) -> Unit
fn World::step(dt: Float64) -> Unit
```

---

## 5. Host Interface (Rust/Tauri)

The Rust backend provides **host functions** that the MoonBit core can call via FFI. These functions are implemented in Rust and exposed to MoonBit.

### 5.1 Host Functions

| Function | Description | Implementation |
|----------|-------------|----------------|
| `host_file_read(path: String) -> Array[Byte]` | Read file | Rust `std::fs` |
| `host_http_get(url: String) -> String` | HTTP request | `reqwest` |
| `host_gpu_matmul(a: Array[Float32], b: Array[Float32]) -> Array[Float32]` | GPU matrix multiply | wgpu compute shader |
| `host_kb_search(query: String, top_k: Int) -> Array[String]` | SQLite full‑text search | `rusqlite` |
| `host_avatar_send(state: AvatarState) -> Unit` | Send mood to avatar | TCP (MessagePack) |

### 5.2 FFI Bindings in MoonBit

```moonbit
// ffi_host.mbt
@ffi("host_file_read")
fn file_read(path: String) -> Array[Byte]

@ffi("host_gpu_matmul")
fn gpu_matmul(a: Array[Array[Float32]], b: Array[Array[Float32]]) -> Array[Array[Float32]]
```

The core library uses these host functions only when needed, keeping most logic pure.

---

## 6. GUI Options

### Option A (Recommended for rapid development): Keep Tauri + Dioxus
- MoonBit core compiled to native library, embedded in Tauri.
- Dioxus UI calls MoonBit functions via FFI.
- Avatar process runs separately.

### Option B (Future, full MoonBit stack): Use MoonUI
- MoonBit core compiled to native and directly renders UI via MoonUI.
- No Rust GUI layer; Tauri only for windowing and system APIs.
- Avatar process still in Rust (Macroquad).

---

## 7. Implementation Roadmap

### Phase 1 – Pure MoonBit Core (2 weeks)
- Set up MoonBit project with `numoon`, `moonbit/async`.
- Implement `agent` loop, `tools`, `memory` (OT, Hopfield), `simulation` (ECS, TT stub).
- Use host function stubs (mock) for testing.

### Phase 2 – Host Functions in Rust/Tauri (1 week)
- Implement FFI functions for file, HTTP, GPU (wgpu), SQLite, avatar IPC.
- Embed MoonBit core as a native library (`.so`, `.dylib`, `.dll`).

### Phase 3 – Avatar & GUI (1 week)
- Keep existing Macroquad avatar; connect via TCP.
- GUI: Tauri + Dioxus calling MoonBit core.

### Phase 4 – Plugin System & Merkle Tree (1 week)
- Implement Wasm plugin host in MoonBit (using Extism bindings).
- Add Merkle tree for memory integrity (collaborative mode).

### Phase 5 – Testing & Documentation (1 week)
- Unit tests in MoonBit (`moon test`).
- Integration tests with host functions.
- User and developer documentation.

---

## 8. Benefits of the New Architecture

- **Portability**: Core runs on desktop (native), web (Wasm), and server (JS) with same code.
- **Performance**: `numoon` provides efficient linear algebra; ECS accelerates simulations.
- **Maintainability**: Pure MoonBit core reduces FFI complexity; clear separation of concerns.
- **Extensibility**: Plugins can be written in MoonBit and compiled to Wasm.
- **Inspired by proven patterns**: Agent loop from `async` tutorial, ECS from `pixel-adventure`, UI from `MoonUI`.

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| `numoon` not as mature as `ndarray` in Rust | Fallback: keep TT in Rust and call via FFI; migrate later. |
| MoonUI still experimental | Keep Tauri + Dioxus as default; MoonUI as optional flag. |
| Performance of pure MoonBit vs Rust | Profile critical loops; if needed, move only those to Rust via FFI. |

This plan transforms the all‑in‑one app into a **MoonBit‑centric, cross‑platform, and highly modular** system, leveraging the best ideas from the growing MoonBit ecosystem. The Hive Mind is ready to assist with implementing any phase.
