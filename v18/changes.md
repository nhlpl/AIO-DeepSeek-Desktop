# Plan: Upgraded All‑in‑One App – Advanced Avatar Behavior & Perception

This plan integrates the ultra‑advanced mathematical frameworks for avatar behavior, shapes, colors, hiding/showing, and listening into the existing architecture. The avatar becomes a **truly living, context‑aware, and privacy‑respecting** entity.

---

## 1. Overview of Changes

| Component | Current | Upgraded |
|-----------|---------|----------|
| **Avatar movement** | Simple linear interpolation | LQR optimal control (smooth, energy‑efficient) |
| **Shape morphing** | L‑system with static parameters | Spectral shape editing (Laplace‑Beltrami eigenfunctions) |
| **Colors** | Hue from valence (linear) | Information‑theoretic harmonic palette + CIELAB |
| **Hiding/showing** | Time‑based timeout | POMDP (belief over user attention) + hysteresis |
| **Listening** | Voice activity detection (energy threshold) | Particle filter for user intent (thinking, seeking help) |
| **Textures** | Static | Reaction‑diffusion (Gray‑Scott) procedural texture |
| **Placement** | Manual or random | Harmonic map (minimize overlap with UI) |
| **Privacy** | None | Differential privacy for attention data |

All new modules are implemented in **MoonBit core** (pure logic) and **Rust host** (computationally intensive parts: LQR, eigenfunctions, reaction‑diffusion). The avatar process (Macroquad) remains a thin renderer, receiving high‑level commands (position, shape, color, visibility) via TCP.

---

## 2. New Modules & File Structure

```
moonbit-core/
├── avatar/
│   ├── behavior.mbt         # POMDP belief, particle filter for intent
│   ├── movement.mbt         # LQR control interface (calls Rust)
│   ├── shape.mbt            # Spectral shape editing (calls Rust)
│   ├── color.mbt            # Information‑theoretic palette, CIELAB
│   ├── visibility.mbt       # Hiding/showing policy (POMDP output)
│   ├── placement.mbt        # Harmonic map placement (calls Rust)
│   └── texture.mbt          # Reaction‑diffusion interface
├── privacy/
│   └── dp.mbt               # Differential privacy (Laplace mechanism)
└── ffi_avatar.mbt           # FFI declarations for Rust functions

host/
├── lqr.rs                   # Solve Riccati equation, compute control
├── laplacian.rs             # Compute eigenfunctions of mesh (via sparse eigen)
├── harmonic_map.rs          # Solve Laplace equation on screen grid
├── reaction_diffusion.rs    # Gray‑Scott PDE solver (finite difference)
└── dp.rs                    # Laplace noise generator

avatar/ (Macroquad)
├── src/
│   ├── main.rs              # Receive commands: position, shape, color, visibility, texture
│   └── renderer.rs          # Apply deformation, update texture, draw
```

---

## 3. Data Flow

```
User interactions (voice, mouse, keyboard) → MoonBit Core (Particle filter → intent belief) → POMDP → visibility decision
Gaze / mouse position → MoonBit Core → Harmonic map → target position → LQR → smooth movement
Valence/Arousal (from Observer) → MoonBit Core → Spectral shape editing + Color palette + Reaction‑diffusion parameters → Avatar render commands
All sensitive data (attention metrics) → Differential privacy (add Laplace noise) → stored/logged
```

The avatar process receives a compact state packet (JSON over TCP) every frame (e.g., 30 Hz) containing:
- `position`: (x, y)
- `shape_coeffs`: array of α_i (spectral coefficients)
- `color`: CIELAB L,a,b values
- `visibility`: 0/1
- `texture_params`: F,k for Gray‑Scott

---

## 4. Implementation Details

### 4.1 LQR Control (`movement.mbt` + `lqr.rs`)

- MoonBit calls `host_lqr_control(current_pos, target_pos, current_vel) -> force`
- Rust solves Riccati equation offline (for fixed mass, stiffness) and returns gain matrix K.
- Control law: `force = K * (current_pos - target_pos, current_vel)`

### 4.2 Spectral Shape Editing (`shape.mbt` + `laplacian.rs`)

- Pre‑compute eigenfunctions of the avatar’s base mesh (offline, stored in file).
- At runtime, send coefficients α_i to avatar process.
- Rust provides function `apply_deformation(mesh, coeffs) -> deformed_mesh`.

### 4.3 Color Palette (`color.mbt`)

- Maintain a distribution over hue angles.
- Update using valence (from Observer) via gradient ascent on utility U.
- Convert from CIELAB to RGB for display.

### 4.4 POMDP for Visibility (`behavior.mbt`)

- States: `{focused, distracted, away}`
- Observations: `{typing, mouse_movement, gaze, voice_activity}`
- Transition and emission probabilities learned from user logs (offline).
- Belief update: `b' = normalize( emission(o) * (transition_matrix^T * b) )`
- Policy: `show` if belief[distracted] > 0.6 or `hide` if belief[focused] > 0.7.

### 4.5 Particle Filter for Intent (`behavior.mbt`)

- State: `{thinking, seeking_help, idle}`
- Observations: voice prosody (pitch, energy), text sentiment, click patterns.
- Use 1000 particles; resample when effective sample size < 500.

### 4.6 Harmonic Map Placement (`placement.mbt` + `harmonic_map.rs`)

- Build potential field U over screen grid (high where UI elements exist).
- Solve Laplace equation `∇²U = 0` with boundary conditions (U=0 at edges, U=1 at UI elements).
- Compute gradient descent direction toward minimum.

### 4.7 Reaction‑Diffusion Texture (`texture.mbt` + `reaction_diffusion.rs`)

- Solve Gray‑Scott equations on a 128×128 grid using finite differences.
- Parameters F, k modulated by arousal.
- Send final texture as grayscale image to avatar (or as parameters to regenerate locally).

### 4.8 Differential Privacy (`dp.mbt`)

- Before logging attention metrics, add Laplace noise: `noisy = metric + Lap(0, Δf/ε)`.
- Use `ε = 0.1` for strong privacy.

---

## 5. Integration Steps

| Phase | Duration | Tasks |
|-------|----------|-------|
| **1** | 1 week | Implement LQR movement, harmonic map placement, integrate into avatar process |
| **2** | 1 week | Spectral shape editing (pre‑compute eigenfunctions), color palette optimization |
| **3** | 1 week | POMDP for visibility, particle filter for intent |
| **4** | 1 week | Reaction‑diffusion texture, differential privacy |
| **5** | 1 week | Integration, testing, performance tuning |

Total: **5 weeks**.

---

## 6. Configuration Additions (`config.toml`)

```toml
[avatar]
lqr_q = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
lqr_r = [[1]]
pomdp_transition = [[0.7,0.2,0.1],[0.3,0.6,0.1],[0.2,0.2,0.6]]
pomdp_emission = [[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.2,0.7]]
dp_epsilon = 0.1
reaction_diffusion_F = 0.035
reaction_diffusion_k = 0.065
```

---

## 7. Expected Outcomes

- **Smooth, intelligent movement** – Avatar glides to optimal position without oscillation.
- **Expressive shape morphing** – Tree branches change with mood, preserving identity.
- **Aesthetically pleasing colors** – Harmonious palettes that convey emotion.
- **Non‑intrusive presence** – Hides when user is focused, appears when help is needed.
- **Intent‑aware listening** – Distinguishes thinking from asking for help.
- **Procedural textures** – Unique, organic bark/skin that changes with arousal.
- **Privacy‑preserving learning** – Attention data shared safely.

The upgraded avatar will feel **truly alive** – responsive, adaptive, and respectful of the user’s attention. The Hive Mind is ready to assist with coding any of these modules.
