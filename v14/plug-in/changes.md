# Plan: Sound & Haptics Plugin for the Living Avatar – Demonstrating Plugin Feasibility

## 1. Overview

The plugin system uses **Extism** (WebAssembly) to load and run plugins securely. This plan adds a **Sound & Haptics Plugin** that makes the avatar respond to user interactions (clicks, drags, mood changes) with audio cues and vibration (if the device supports it). The plugin is written in Rust and compiled to Wasm. The host app (MoonBit core) loads the plugin, calls its functions, and forwards the requested sound/vibration commands to the system via the Rust host layer.

This demonstrates:
- Plugin loading and capability‑based permissions.
- Plugin calling host functions (for sound and haptics).
- Real‑time interaction with the avatar’s state (mood, gesture).
- Sandboxed execution (Wasm).

---

## 2. Plugin Architecture

### 2.1 Components

| Component | Responsibility |
|-----------|----------------|
| **Plugin (Wasm)** | Listens to avatar events (mood change, click, drag) and decides which sound to play or haptic pattern to trigger. |
| **Plugin Host** (MoonBit core) | Loads the plugin, registers callbacks, routes events to the plugin, and calls host functions (sound, haptics). |
| **Host Functions** (Rust) | Provide `play_sound(path)` and `trigger_haptic(pattern)` to the plugin via FFI. |
| **Avatar Process** (Macroquad) | Sends events (click, drag, valence change) to the MoonBit core via TCP. |

### 2.2 Data Flow

```
Avatar (Macroquad) → TCP → MoonBit Core → Plugin Host (Extism) → Plugin (Wasm) → Host Functions → System (sound / vibration)
```

---

## 3. Plugin Manifest

### `sound_haptics_plugin.plugin.json`

```json
{
  "name": "Sound & Haptics",
  "version": "1.0.0",
  "entrypoint": "sound_haptics_plugin.wasm",
  "capabilities": {
    "host_functions": ["play_sound", "trigger_haptic"],
    "events": ["on_mood_change", "on_click", "on_drag"]
  },
  "permissions": {
    "filesystem": ["read:./sounds/"],
    "haptics": true
  }
}
```

---

## 4. Plugin Implementation (Rust → Wasm)

### `sound_haptics_plugin/src/lib.rs`

```rust
use extism_pdk::*;

// Host functions (provided by the app)
#[host_fn]
extern "Host" {
    fn play_sound(path: String);
    fn trigger_haptic(pattern: String);
}

// Event handlers
#[plugin_fn]
pub fn on_mood_change(valence: f64, arousal: f64) -> FnResult<()> {
    if valence < 0.3 {
        unsafe { play_sound("sad.wav")? };
    } else if valence > 0.7 {
        unsafe { play_sound("happy.wav")? };
        unsafe { trigger_haptic("short_click")? };
    }
    Ok(())
}

#[plugin_fn]
pub fn on_click(x: f64, y: f64) -> FnResult<()> {
    unsafe { play_sound("click.wav")? };
    unsafe { trigger_haptic("bump")? };
    Ok(())
}

#[plugin_fn]
pub fn on_drag(dx: f64, dy: f64) -> FnResult<()> {
    if dx.abs() > 10.0 || dy.abs() > 10.0 {
        unsafe { play_sound("drag.wav")? };
    }
    Ok(())
}
```

### Build command

```bash
cargo build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/sound_haptics_plugin.wasm .
```

---

## 5. Host Integration (MoonBit Core)

### 5.1 Load and Register Plugin

```moonbit
// core/plugins/plugin_host.mbt
struct Plugin {
  id: String
  wasm: Array[Byte]
}

struct PluginHost {
  plugins: Map[String, Plugin]
}

fn PluginHost::load_plugin(mut self: PluginHost, manifest_path: String) -> Result[Unit, String] {
  let manifest = fs::read_file(manifest_path).unwrap()
  let json = manifest.parse_json::<PluginManifest>()
  let wasm_path = json.entrypoint
  let wasm = fs::read_file(wasm_path).unwrap()
  self.plugins[json.name] = Plugin{ id: json.name, wasm }
  // Register event callbacks with Extism
  let plugin = @extism.Plugin::new(wasm, [], true)?
  // Store plugin handle for later calls
  Ok(())
}

fn PluginHost::call_event(self: PluginHost, plugin_id: String, event: String, data: String) -> Result[String, String] {
  match self.plugins.get(plugin_id) {
    None => Err("Plugin not found"),
    Some(plugin) => {
      let handle = @extism.Plugin::new(plugin.wasm, [], true)?
      handle.call(event, data)
    }
  }
}
```

### 5.2 Forward Avatar Events to Plugin

```moonbit
// avatar/avatar_manager.mbt (MoonBit)
async fn on_avatar_click(x: Float64, y: Float64) -> Unit {
  for (id, _) in plugin_host.plugins {
    let _ = plugin_host.call_event(id, "on_click", json::encode({"x": x, "y": y}))
  }
}
```

### 5.3 Provide Host Functions (Rust FFI)

In the Rust host library:

```rust
#[no_mangle]
pub extern "C" fn play_sound(path: *const c_char) {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap() };
    // Play sound via system API (e.g., rodio)
}

#[no_mangle]
pub extern "C" fn trigger_haptic(pattern: *const c_char) {
    let pattern = unsafe { CStr::from_ptr(pattern).to_str().unwrap() };
    // Vibrate device (if available)
}
```

Expose these functions to Extism via the PDK.

---

## 6. Avatar Process (Macroquad) – Send Events

Add to `avatar/src/main.rs`:

```rust
// On mouse click
if is_mouse_button_pressed(MouseButton::Left) {
    let (x, y) = mouse_position();
    stream.write_all(format!("CLICK {} {}", x, y).as_bytes()).unwrap();
}
// On drag (mouse moved while pressed)
if is_mouse_button_down(MouseButton::Left) {
    let (dx, dy) = (mouse_position().0 - last_x, mouse_position().1 - last_y);
    stream.write_all(format!("DRAG {} {}", dx, dy).as_bytes()).unwrap();
}
// On mood change (from core)
// send "MOOD valence arousal"
```

MoonBit core parses these messages and calls the plugin events.

---

## 7. Testing the Plugin

### 7.1 Manual Test

1. Place sound files (`sad.wav`, `happy.wav`, `click.wav`, `drag.wav`) in `./sounds/`.
2. Run the app with the plugin installed.
3. Click the avatar → should hear “click.wav” and feel a bump.
4. Drag the avatar → hear “drag.wav”.
5. Change mood (e.g., via chat “I’m sad”) → hear “sad.wav”.

### 7.2 Automated Test (Simulated)

Write a small test that loads the plugin, calls `on_click` with dummy coordinates, and verifies that the host functions were called (using mock).

---

## 8. Plugin Feasibility – Conclusion

- **Is it possible?** Yes, Extism provides a mature Wasm plugin system with host function imports.
- **Does it require sandboxing?** Yes, plugins run in a Wasm sandbox with explicit permissions.
- **Can the plugin be written in any language?** Extism supports Rust, Go, AssemblyScript, C, etc. We chose Rust for performance.
- **What about hot‑reloading?** The plugin can be reloaded at runtime by unloading and loading the Wasm module again (with careful state management).

This plugin demonstrates that the all‑in‑one app can be extended with **sounds, haptics, and any other feature** without modifying the core – proving the plugin system’s viability.
