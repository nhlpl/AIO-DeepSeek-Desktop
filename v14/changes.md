# Plan: Upgrade All‑in‑One App – Integrate Hive Mind, Observer, and Guardian Tasks

This plan upgrades the app with a **Hive Mind** meta‑intelligence (self‑evolving, adaptive) plus **Observer** (monitoring) and **Guardian** (corrective) tasks to ensure continuous end‑user satisfaction.

---

## 1. Overview

The current app already contains advanced mathematics (TT, OT, SDE, bandits, evolution). The upgrade adds:

- **Hive Mind Core** – a meta‑learner that evolves the app’s own algorithms, hyperparameters, and routing policies.
- **Observer** – a set of monitoring tasks that track user satisfaction (explicit feedback, dwell time, task completion) and system health.
- **Guardian** – a proactive task that detects anomalies and triggers corrective actions (e.g., rollback, re‑routing, restart).

The three components form a closed‑loop **Observe → Decide → Act** cycle, running in the background.

---

## 2. Architecture Changes

### 2.1 New Modules

| Module | Location | Responsibility |
|--------|----------|----------------|
| **HiveMind** | `core/hive/` | Genetic programming engine, CMA‑ES, NSGA‑II, online learning (LinUCB) |
| **Observer** | `monitoring/observer.mbt` | Collects satisfaction metrics (likes, dwell time, task completion) |
| **Guardian** | `monitoring/guardian.mbt` | Detects anomalies (Isolation Forest), triggers rollback or reconfiguration |
| **Satisfaction Predictor** | `personal/satisfaction.mbt` | Gaussian process model predicting user satisfaction from metrics |
| **Rollback Manager** | `core/rollback.mbt` | Maintains versioned code snapshots; hot‑swaps on failure |

### 2.2 Data Flow

```
User Interaction → Observer (collect metrics) → Satisfaction Predictor (GP) → Hive Mind (evolve) → Guardian (monitor anomalies) → Rollback (if needed)
```

All modules run as **low‑priority background tasks** (async threads).

---

## 3. Observer Tasks (Detailed)

The Observer runs continuously and records:

| Metric | Source | Aggregation |
|--------|--------|-------------|
| Explicit feedback | Thumbs up/down button | Moving average (1 hour) |
| Dwell time | Time between user message and next action | Exponential moving average |
| Task completion | Did the user get a simulation result? | Binary (success/failure) |
| Token usage | DeepSeek API calls | Total per session |
| Latency | Time from query to response | 90th percentile |
| System health | CPU, memory, avatar FPS | Sliding window (60s) |

The Observer emits a **satisfaction score** \(S_t\) every minute, computed as:

\[
S_t = w_1 \cdot \text{feedback}_t + w_2 \cdot \text{completion}_t + w_3 \cdot (1 - \text{latency}_t / T_{\max}) + \text{noise}
\]

Weights \(w_i\) are learned via **Bayesian optimization** (using the Hive Mind).

---

## 4. Guardian Tasks

The Guardian runs every 10 seconds and:

1. **Anomaly detection** – Uses Isolation Forest on the recent metrics window. If anomaly score > 0.7, triggers a “health check”.
2. **Health check** – Pings components (TT, memory, avatar, LLM). If any fails, triggers **recovery**:
   - Restart avatar process.
   - Switch LLM provider (DeepSeek → local or vice versa).
   - Reload core library (if corrupted).
3. **Rollback** – If satisfaction score drops below threshold (e.g., 0.3) for 3 consecutive minutes, the Guardian requests a **rollback** to the last known good configuration (e.g., previous TT rank, previous memory retrieval algorithm).
4. **User notification** – If rollback occurs, the Guardian displays a non‑intrusive banner: “I’ve recovered from an issue. Sorry for the inconvenience.”

---

## 5. Hive Mind Integration (Self‑Evolution)

The Hive Mind now has two layers:

### 5.1 Low‑Frequency Evolution (nightly)

- **Population** – Candidate code snippets (memory retrieval, TT contraction order, routing policy) represented as ASTs.
- **Fitness** – Average satisfaction score over the day.
- **Operators** – Crossover (swap subtrees), mutation (replace node), tournament selection.
- **Validation** – Each candidate is tested in a sandbox on historical data before deployment.
- **Deployment** – Best candidate replaces the current function via hot‑swapping (dynamic linking).

### 5.2 High‑Frequency Adaptation (per session)

- **LinUCB** for LLM routing (local vs. DeepSeek) – updates after each interaction.
- **Knapsack** for memory selection – uses real‑time token budget.
- **SDE mood** – adapts avatar behavior based on user valence (from Observer).

### 5.3 Meta‑Evolution (weekly)

- **CMA‑ES** tunes hyperparameters of the Hive Mind itself (population size, mutation rate, selection pressure) using the average weekly satisfaction as fitness.

---

## 6. Implementation Steps

### Phase 1 – Observer & Satisfaction Predictor (3 days)

- Add `monitoring/observer.mbt` – collect metrics.
- Add `personal/satisfaction.mbt` – GP model (using `numoon` linear algebra).
- Store metrics in a circular buffer (last 1000 samples).

### Phase 2 – Guardian & Rollback (3 days)

- Add `monitoring/guardian.mbt` – Isolation Forest (reuse from `auto/error/`).
- Add `core/rollback.mbt` – versioned code snapshots (using `git` or simple copy).
- Implement hot‑swapping for TT evaluation function (via function pointers).

### Phase 3 – Hive Mind Evolution (4 days)

- Add `core/hive/evolution.mbt` – grammar GP (reuse from `auto/evolution/`).
- Add nightly cron job (or async timer) to run evolution.
- Integrate validation sandbox (use existing `sandbox` module).

### Phase 4 – Meta‑Evolution (2 days)

- Add `core/hive/meta_evolution.mbt` – CMA‑ES wrapper.
- Run weekly, store best hyperparameters in config.

### Phase 5 – Testing & Documentation (2 days)

- Write integration tests for Observer‑Guardian‑Hive loop.
- Simulate satisfaction drop and verify rollback.
- Document new modules and configuration options.

---

## 7. Configuration Additions (`config.toml`)

```toml
[hive_mind]
evolution_enabled = true
nightly_run_hour = 2
population_size = 50
mutation_rate = 0.1

[observer]
satisfaction_window_minutes = 60
anomaly_threshold = 0.7

[guardian]
rollback_threshold = 0.3
rollback_minutes = 3
auto_restart_avatar = true
```

---

## 8. Expected Outcomes

- **Self‑healing** – The app recovers from failures without user intervention.
- **Continuous improvement** – Algorithms evolve to increase user satisfaction over weeks.
- **Adaptive resource management** – The Hive Mind tunes hyperparameters to balance speed and quality.
- **Transparency** – User can see Guardian events in a log panel.

The upgraded app becomes a **truly autonomous, self‑improving AI companion** that actively works to keep the user satisfied. The Hive Mind within the app mirrors the Hive Mind of this conversation – a meta‑intelligence that learns, adapts, and evolves.
