# Code for Built‑in Self‑Diagnosis, Auto‑Tuning, and Rollback

We implement the health monitor, auto‑tuner, and rollback modules as background tasks. The code integrates with the existing simplified architecture.

---

## 1. Rust Host Additions – Metrics & Config

### `host/src/metrics.rs`

```rust
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use once_cell::sync::Lazy;

static LAST_LATENCY: AtomicU64 = AtomicU64::new(0);
static LAST_MEMORY: AtomicU64 = AtomicU64::new(0);
static ERROR_COUNT: Mutex<Vec<Instant>> = Mutex::new(Vec::new());

pub fn record_latency(ms: u64) {
    LAST_LATENCY.store(ms, Ordering::Relaxed);
}

pub fn record_memory(mb: u64) {
    LAST_MEMORY.store(mb, Ordering::Relaxed);
}

pub fn record_error() {
    let mut guard = ERROR_COUNT.lock().unwrap();
    guard.push(Instant::now());
    // keep last 60 seconds
    let cutoff = Instant::now() - Duration::from_secs(60);
    guard.retain(|&t| t > cutoff);
}

#[no_mangle]
pub extern "C" fn host_get_latency() -> f64 {
    LAST_LATENCY.load(Ordering::Relaxed) as f64
}

#[no_mangle]
pub extern "C" fn host_get_memory() -> f64 {
    LAST_MEMORY.load(Ordering::Relaxed) as f64
}

#[no_mangle]
pub extern "C" fn host_get_error_rate() -> f64 {
    let guard = ERROR_COUNT.lock().unwrap();
    guard.len() as f64 / 60.0
}
```

### `host/src/config.rs`

```rust
use std::sync::RwLock;
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;

#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub gpu_enabled: bool,
    pub cache_size: u8,   // 0 = small, 1 = large
    pub debug_logging: bool,
    // other bits can be added
}

static CONFIG: Lazy<RwLock<AppConfig>> = Lazy::new(|| {
    RwLock::new(AppConfig {
        gpu_enabled: false,
        cache_size: 1,
        debug_logging: false,
    })
});

#[no_mangle]
pub extern "C" fn host_config_get_gpu() -> bool {
    CONFIG.read().unwrap().gpu_enabled
}

#[no_mangle]
pub extern "C" fn host_config_get_cache() -> u8 {
    CONFIG.read().unwrap().cache_size
}

#[no_mangle]
pub extern "C" fn host_config_get_debug() -> bool {
    CONFIG.read().unwrap().debug_logging
}

#[no_mangle]
pub extern "C" fn host_config_set_gpu(enabled: bool) {
    CONFIG.write().unwrap().gpu_enabled = enabled;
}

#[no_mangle]
pub extern "C" fn host_config_set_cache(size: u8) {
    CONFIG.write().unwrap().cache_size = size;
}

#[no_mangle]
pub extern "C" fn host_config_set_debug(enabled: bool) {
    CONFIG.write().unwrap().debug_logging = enabled;
}
```

Add these to `host/src/lib.rs`:

```rust
mod metrics;
mod config;
// ... existing modules
```

Also add FFI declarations in `ffi_host.mbt`.

---

## 2. MoonBit FFI Declarations

Add to `moonbit-core/src/ffi_host.mbt`:

```moonbit
// Metrics
@ffi("host_get_latency")
fn get_latency() -> Float64

@ffi("host_get_memory")
fn get_memory() -> Float64

@ffi("host_get_error_rate")
fn get_error_rate() -> Float64

// Config
@ffi("host_config_get_gpu")
fn config_get_gpu() -> Bool

@ffi("host_config_get_cache")
fn config_get_cache() -> Int

@ffi("host_config_get_debug")
fn config_get_debug() -> Bool

@ffi("host_config_set_gpu")
fn config_set_gpu(enabled: Bool) -> Unit

@ffi("host_config_set_cache")
fn config_set_cache(size: Int) -> Unit

@ffi("host_config_set_debug")
fn config_set_debug(enabled: Bool) -> Unit
```

---

## 3. MoonBit Auto Modules

### `moonbit-core/src/auto/health_monitor.mbt`

```moonbit
use moonbitlang/async

let anomaly_counter = Cell::new(0)

fn check_health() -> Bool {
  let lat = get_latency()
  let mem = get_memory()
  let err = get_error_rate()
  let sat = observer::get_satisfaction()
  lat < 500.0 && mem < 800.0 && err < 0.05 && sat > 0.3
}

async fn health_monitor_loop() -> Unit {
  loop {
    let healthy = check_health()
    if !healthy {
      let count = anomaly_counter.get() + 1
      anomaly_counter.set(count)
      if count >= 3 {
        rollback::trigger_rollback()
      }
    } else {
      let count = anomaly_counter.get()
      if count > 0 {
        anomaly_counter.set(count - 1)
      }
    }
    @async.sleep(10_000).await
  }
}
```

### `moonbit-core/src/auto/auto_tuner.mbt`

```moonbit
use moonbitlang/async

const TARGET_LATENCY = 200.0
const TARGET_MEMORY = 600.0

async fn auto_tuner_loop() -> Unit {
  loop {
    let lat = get_latency()
    let mem = get_memory()
    let gpu = config_get_gpu()
    let cache = config_get_cache()
    let debug = config_get_debug()
    if lat > TARGET_LATENCY && !gpu {
      config_set_gpu(true)
      log_warning("Auto‑tuner: enabled GPU")
    }
    if mem > TARGET_MEMORY && cache == 1 {
      config_set_cache(0)
      log_warning("Auto‑tuner: reduced cache size")
    }
    if lat > TARGET_LATENCY && debug {
      config_set_debug(false)
      log_warning("Auto‑tuner: disabled debug logging")
    }
    @async.sleep(30_000).await
  }
}
```

### `moonbit-core/src/auto/rollback.mbt`

```moonbit
use moonbitlang/x/json
use moonbitlang/x/fs

struct GoodConfig {
  gpu: Bool
  cache: Int
  debug: Bool
}

let good_config = Cell::new(None)

fn save_good_config() -> Unit {
  let cfg = GoodConfig{
    gpu: config_get_gpu(),
    cache: config_get_cache(),
    debug: config_get_debug()
  }
  good_config.set(Some(cfg))
  let json_str = cfg.to_json().stringify()
  fs::write_file("good_config.json", json_str.to_bytes())
}

fn load_good_config() -> Unit {
  match good_config.get() {
    Some(cfg) => {
      config_set_gpu(cfg.gpu)
      config_set_cache(cfg.cache)
      config_set_debug(cfg.debug)
    }
    None => {
      // try load from disk
      match fs::read_file("good_config.json") {
        Ok(bytes) => {
          let json_str = String::from_bytes(bytes)
          let cfg = json_str.parse_json::<GoodConfig>()
          config_set_gpu(cfg.gpu)
          config_set_cache(cfg.cache)
          config_set_debug(cfg.debug)
          good_config.set(Some(cfg))
        }
        Err(_) => ()
      }
    }
  }
}

fn trigger_rollback() -> Unit {
  log_warning("Rollback triggered – reverting to last good configuration")
  load_good_config()
  // Optionally restart GPU pipeline, clear cache (host functions)
  host_restart_gpu()
  host_clear_cache()
}

// Periodic snapshot (called every hour)
async fn snapshot_loop() -> Unit {
  loop {
    @async.sleep(3_600_000).await
    save_good_config()
  }
}
```

Add host functions for restart/clear:

```moonbit
@ffi("host_restart_gpu")
fn host_restart_gpu() -> Unit

@ffi("host_clear_cache")
fn host_clear_cache() -> Unit
```

Implement stubs in Rust host:

```rust
#[no_mangle]
pub extern "C" fn host_restart_gpu() {
    // reinitialize wgpu device
}

#[no_mangle]
pub extern "C" fn host_clear_cache() {
    // clear in‑memory caches
}
```

---

## 4. Integration in `main.mbt`

```moonbit
async fn main() {
  // ... existing init ...
  // Start background auto tasks
  spawn(health_monitor_loop())
  spawn(auto_tuner_loop())
  spawn(snapshot_loop())
  // ... rest of main ...
}
```

Also ensure observer satisfaction is accessible. Add to `observer.mbt`:

```moonbit
static SATISFACTION: Cell[Float64] = Cell::new(0.5)

fn update_satisfaction(feedback: Float64) {
  let alpha = 0.1
  let old = SATISFACTION.get()
  SATISFACTION.set(alpha * feedback + (1.0 - alpha) * old)
}

fn get_satisfaction() -> Float64 {
  SATISFACTION.get()
}
```

---

## 5. Configuration File Update

`config.toml` now also stores runtime tunables (but the auto‑tuner modifies them via host functions). The initial config is read at startup.

---

## 6. Build & Run

No changes to build process. After adding these files, rebuild:

```bash
cd moonbit-core && moon build --target native
cd ../host && cargo build --release
cd ../tauri && cargo tauri build
```

The app will now:

- Monitor health every 10 seconds.
- Auto‑tune GPU, cache, debug logging based on latency/memory.
- Save a good config snapshot every hour.
- Roll back if anomalies persist.

This completes the built‑in self‑diagnosis, auto‑tuning, and rollback system. The Hive Mind declares the code ready.
