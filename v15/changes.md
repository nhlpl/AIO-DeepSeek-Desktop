# Plan: Bug Fixes and Architectural Improvements for the All‑in‑One App

Based on the simulation results, we identified critical issues: host function unavailability, plugin load failures, high event rate latency, intermittent sound failures, and missing haptics. Below is a comprehensive plan to address these bugs and improve resilience.

---

## 1. Summary of Identified Issues

| Issue | Root Cause | Impact | Priority |
|-------|------------|--------|----------|
| **Host function unavailability** | Plugin assumes sound/haptics always work | Errors, no feedback | High |
| **Plugin load failure** | No fallback; app crashes if manifest missing | App fails to start | High |
| **High event rate latency** | Synchronous event handling; blocking host functions | UI stutter | Medium |
| **Intermittent sound failures** | File I/O errors, no retry | Inconsistent experience | Medium |
| **Missing haptics** | App assumes haptics always available | Error spam | Low |
| **No graceful degradation** | Plugin errors stop event processing for others | Loss of features | High |

---

## 2. Architectural Changes

### 2.1 Host Function Availability Check & Graceful Degradation

**Change**: Before calling any host function, check availability via a cached capability flag.

**Implementation** (MoonBit core):
```moonbit
// In ffi_host.mbt, add:
@ffi("host_sound_available")
fn sound_available() -> Bool

@ffi("host_haptic_available")
fn haptic_available() -> Bool

// In plugin_host.mbt, before calling:
if sound_available() {
  host_play_sound(path)
} else {
  log_warning("Sound not available, skipping")
}
```

**Rust host**:
```rust
static SOUND_AVAILABLE: AtomicBool = AtomicBool::new(true);
static HAPTIC_AVAILABLE: AtomicBool = AtomicBool::new(true);

#[no_mangle]
pub extern "C" fn host_sound_available() -> bool {
    SOUND_AVAILABLE.load(Ordering::Relaxed)
}

// On init, probe rodio; if fails, set to false.
```

### 2.2 Asynchronous Event Queue & Non‑blocking Host Calls

**Change**: Move plugin event handling to a separate thread pool; host functions are called asynchronously.

**Implementation**:
- MoonBit core uses a `AsyncQueue` for events.
- A background task (`spawn`) processes events and calls plugins.
- Host functions (`play_sound`, `trigger_haptic`) are called with `spawn` so they don’t block the main loop.

```moonbit
// In avatar_manager.mbt
let event_queue = AsyncQueue::new()

fn on_avatar_event(ev) {
  event_queue.push(ev)
}

async fn event_processor() {
  loop {
    let ev = event_queue.pop().await
    // Call plugins asynchronously
    for plugin in plugin_host.plugins.values() {
      spawn(async {
        let result = plugin.call_event(ev.type, ev.data)
        if let Some(actions) = result {
          for action in actions {
            match action {
              "play_sound" => spawn(async { host_play_sound(action.path) })
              "trigger_haptic" => spawn(async { host_trigger_haptic(action.pattern) })
            }
          }
        }
      })
    }
  }
}
```

### 2.3 Plugin Load Failure Handling

**Change**: Plugins are loaded in background; failures are logged but do not prevent app startup.

**Implementation**:
```moonbit
async fn load_plugins_async(plugin_dir: String) {
  let entries = fs::read_dir(plugin_dir).await
  for entry in entries {
    if entry.ends_with(".plugin.json") {
      spawn(async {
        match plugin_host.load_plugin(entry) {
          Ok(_) => log_info("Plugin loaded: " + entry)
          Err(e) => log_error("Failed to load plugin: " + entry + " error: " + e)
        }
      })
    }
  }
}
```

In `main.mbt`, call `load_plugins_async` without awaiting.

### 2.4 Retry & Fallback for Intermittent Sound Failures

**Change**: Wrap host function calls with retry logic and fallback to default sound.

**Implementation**:
```rust
// In Rust host
fn play_sound_with_retry(path: &str, retries: u32) -> bool {
  for i in 0..retries {
    if let Ok(_) = play_sound_impl(path) { return true; }
    std::thread::sleep(Duration::from_millis(50 * (1 << i)));
  }
  // Fallback: play default beep
  play_sound_impl("beep.wav").ok();
  false
}
```

In MoonBit, after a failure, call `host_play_sound_fallback()`.

### 2.5 Cache Host Function Availability

**Change**: Avoid repeated checks; cache results and listen for device changes (optional).

**Implementation**:
```moonbit
static SOUND_AVAILABLE_CACHE: Cell[Option[Bool]] = Cell::new(None)

fn sound_available_cached() -> Bool {
  match SOUND_AVAILABLE_CACHE.get() {
    Some(v) => v
    None => {
      let v = host_sound_available()
      SOUND_AVAILABLE_CACHE.set(Some(v))
      v
    }
  }
}
```

### 2.6 Circuit Breaker for Plugins

**Change**: If a plugin fails repeatedly, disable it temporarily.

**Implementation**:
```moonbit
struct PluginState {
  consecutive_failures: Int
  disabled_until: Option[Float64]
}

fn PluginHost::call_event_with_circuit(plugin_id, event, data) -> Result {
  let state = self.states.get(plugin_id)
  if let Some(disabled_until) = state.disabled_until {
    if now_secs() < disabled_until { return Err("Plugin temporarily disabled") }
  }
  match self.call_event(plugin_id, event, data) {
    Ok(r) => { state.consecutive_failures = 0; Ok(r) }
    Err(e) => {
      state.consecutive_failures += 1
      if state.consecutive_failures >= 5 {
        state.disabled_until = Some(now_secs() + 60.0)
        log_warning("Plugin disabled for 60s due to repeated failures")
      }
      Err(e)
    }
  }
}
```

### 2.7 Performance: Batch Host Calls

**Change**: Group multiple sound/haptic requests within a short time window (e.g., 10 ms) to avoid flooding.

**Implementation**:
- Maintain a small buffer of pending actions.
- A timer flushes the buffer every 10 ms, sending combined commands (e.g., play only the last sound).

---

## 3. Configuration Additions

Add to `config.toml`:

```toml
[plugins]
load_async = true
circuit_breaker_enabled = true
circuit_breaker_failures = 5
circuit_breaker_timeout_secs = 60

[host_functions]
sound_retries = 3
sound_retry_delay_ms = 50
fallback_sound = "beep.wav"

[performance]
event_queue_max_size = 1000
event_processor_threads = 2
```

---

## 4. Testing & Validation

- **Unit tests** for each improvement (circuit breaker, retry logic, async queue).
- **Integration test** that simulates host function failures and verifies graceful degradation.
- **Load test** with 1000 events/second to ensure no queue overflow.
- **Chaos testing**: randomly disable sound/haptics, corrupt plugin manifests.

---

## 5. Implementation Roadmap

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Add availability checks and graceful degradation | 1 day |
| 2 | Implement async event queue and non‑blocking host calls | 2 days |
| 3 | Add plugin load failure handling | 1 day |
| 4 | Implement retry and fallback for sound failures | 1 day |
| 5 | Add circuit breaker for plugins | 1 day |
| 6 | Performance tuning (batching) | 1 day |
| 7 | Testing and documentation | 2 days |

---

## 6. Expected Outcomes

- **No app crashes** due to missing plugins or host functions.
- **Responsive UI** even under high event rate (latency <5 ms).
- **Reliable sound/haptics** with automatic fallback.
- **Self‑healing** plugins: temporary disable on repeated failures.
- **User‑friendly** error messages (non‑intrusive notifications).

The upgraded app will be robust, resilient, and ready for third‑party plugins.
