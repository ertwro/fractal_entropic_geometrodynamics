// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18733424

//! RAM-aware execution mode selection.
//!
//! Detects available system RAM, estimates memory requirements for the
//! given simulation size N, and lets the user choose in-memory or streaming mode.
//!
//! Configuration via `causal_set.toml`:
//! ```toml
//! max_ram_gb = 6.0        # hard ceiling (0 = auto-detect)
//! safety_fraction = 0.70  # fraction of available RAM when auto-detecting
//! ```

use sysinfo::System;
use std::io::Write;
use std::path::Path;

/// Execution mode for the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    /// Keep all data structures in RAM. Fast but needs sufficient memory.
    InMemory,
    /// Write edges to disk and process in passes. Slower but ~2 GB RAM ceiling.
    Streaming,
}

/// User-configurable settings loaded from `causal_set.toml`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Hard RAM ceiling in bytes. If > 0, overrides auto-detection.
    /// When the estimated memory exceeds this, streaming is forced.
    pub max_ram_bytes: u64,
    /// Fraction of available RAM considered safe (0.0–1.0).
    /// Only used when `max_ram_bytes == 0` (auto-detect mode).
    pub safety_fraction: f64,
    /// Default output directory for results and streaming edge files.
    /// CLI positional arg overrides this. Empty string means current dir.
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_ram_bytes: 0,
            safety_fraction: 0.70,
            output_dir: String::new(),
        }
    }
}

const GB: u64 = 1024 * 1024 * 1024;

/// Default config file content.
const DEFAULT_CONFIG: &str = "\
# Causal Set Simulation — configuration
#
# output_dir: Directory for results and streaming edge files.
#   CLI positional argument overrides this.
#   Leave empty to use the current working directory.
output_dir = \".\"

# max_ram_gb: Hard limit on RAM the simulation may use (in GB).
#   If estimated memory exceeds this value, streaming mode is forced.
#   Set to 0 to auto-detect from available system RAM (default).
#   Examples: 4.0 (for a 8 GB laptop), 12.0 (for a 16 GB workstation)
max_ram_gb = 0

# safety_fraction: Fraction of available RAM considered safe (0.0 to 1.0).
#   Only used when max_ram_gb = 0 (auto-detect mode).
#   Default: 0.70 — leaves 30% headroom for the OS and other processes.
safety_fraction = 0.70
";

/// Load configuration from a TOML file, or return defaults.
///
/// Searches for `causal_set.toml` in the given directory first,
/// then falls back to the current working directory.
pub fn load_config(dir: &str) -> Config {
    let candidates = [
        format!("{dir}/causal_set.toml"),
        "causal_set.toml".to_string(),
    ];

    for path in &candidates {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let cfg = parse_config(&contents);
            println!("  (loaded config from {path})");
            return cfg;
        }
    }

    Config::default()
}

/// Parse a simple key=value config (subset of TOML, no crate needed).
fn parse_config(contents: &str) -> Config {
    let mut cfg = Config::default();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            match key {
                "max_ram_gb" => {
                    if let Ok(gb) = value.parse::<f64>() {
                        cfg.max_ram_bytes = (gb * GB as f64) as u64;
                    }
                }
                "safety_fraction" => {
                    if let Ok(frac) = value.parse::<f64>() {
                        cfg.safety_fraction = frac.clamp(0.05, 1.0);
                    }
                }
                "output_dir" => {
                    // Strip surrounding quotes if present
                    let v = value.trim_matches('"').trim_matches('\'');
                    cfg.output_dir = v.to_string();
                }
                _ => {} // ignore unknown keys
            }
        }
    }
    cfg
}

/// Write the default config file if it doesn't already exist.
pub fn ensure_config(dir: &str) {
    let path = format!("{dir}/causal_set.toml");
    if !Path::new(&path).exists() {
        if let Ok(()) = std::fs::write(&path, DEFAULT_CONFIG) {
            println!("  (created default config: {path})");
        }
    }
}

/// Detect available system RAM in bytes.
pub fn available_ram_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Detect total system RAM in bytes.
pub fn total_ram_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory()
}

/// Estimate peak memory usage in bytes for a single run at given N (in-memory mode).
///
/// Heuristic based on the data structures built during Phases 1-3:
/// - N <= 3000 (eigendecomp, default `--eigen-cutoff`): dense N×N matrices dominate (~N² × 24 bytes)
/// - 3000 < N <= 50_000: sparse structures (~N × 2000 bytes)
/// - N > 50_000 (in-memory): sorted_pts + sorted_coords + grid + CSR + defect (~N × 500 bytes)
/// - N > 50_000 (streaming): 8*N (degree arrays) + 50*(N/10) (core CSR) + 100MB ≈ 13*N bytes
///   At N=100M: ~1.3 GB per realization.
pub fn estimate_memory_bytes(n: usize) -> u64 {
    if n <= 3_000 {
        (n as u64) * (n as u64) * 24
    } else if n <= 50_000 {
        (n as u64) * 2_000
    } else {
        (n as u64) * 500
    }
}

/// Estimate peak memory for streaming mode at given N.
///
/// Single-pass scan: 8*N (degree arrays) + 50*(N/10) (core CSR) + ~100MB buffer ≈ 13*N bytes.
pub fn estimate_streaming_memory_bytes(n: usize) -> u64 {
    (n as u64) * 13 + 100 * 1024 * 1024
}

/// Default number of concurrent realizations for a given N (in-memory mode).
///
/// At N > 10M each realization uses ~3.7 GB in-memory, so 2 concurrent
/// fits comfortably in 30 GB without thrashing.  This is the fallback
/// heuristic when `--threads` is not provided.  Streaming mode uses a
/// CPU-based heuristic instead (see main.rs).
pub fn max_concurrent_runs(n: usize) -> usize {
    if n > 10_000_000 { 2 } else { 4 }
}

/// Format bytes into a human-readable string.
pub fn fmt_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

/// Recommend an execution mode based on config + available RAM vs estimated usage.
///
/// The estimate accounts for default concurrent realizations (2 for N>10M,
/// 4 otherwise) since the ensemble runs in parallel.  When `--threads` is
/// provided, the actual concurrency may differ.
pub fn recommend_mode(n: usize, cfg: &Config) -> (ExecMode, u64, u64) {
    let available = available_ram_bytes();
    let concurrent = max_concurrent_runs(n) as u64;
    let estimated = estimate_memory_bytes(n) * concurrent;

    let ceiling = if cfg.max_ram_bytes > 0 {
        // Hard ceiling from config file
        cfg.max_ram_bytes
    } else {
        // Auto-detect: use safety_fraction of available RAM
        (available as f64 * cfg.safety_fraction) as u64
    };

    let mode = if estimated <= ceiling {
        ExecMode::InMemory
    } else {
        ExecMode::Streaming
    };
    (mode, available, estimated)
}

/// Prompt the user to confirm or override the recommended execution mode.
///
/// Returns the chosen `ExecMode`. Pressing Enter accepts the recommendation.
pub fn prompt_mode(recommended: ExecMode, available: u64, estimated: u64, cfg: &Config) -> ExecMode {
    let total = total_ram_bytes();
    let rec_label = match recommended {
        ExecMode::InMemory => "in-memory",
        ExecMode::Streaming => "streaming",
    };

    println!("── Memory ──────────────────────────────────────────");
    println!("  Total RAM:      {}", fmt_bytes(total));
    println!("  Available RAM:  {}", fmt_bytes(available));
    println!("  Estimated need: {}", fmt_bytes(estimated));
    if cfg.max_ram_bytes > 0 {
        println!("  RAM ceiling:    {} (from causal_set.toml)", fmt_bytes(cfg.max_ram_bytes));
    } else {
        let safe = (available as f64 * cfg.safety_fraction) as u64;
        println!("  Safe limit:     {} ({:.0}% of available)", fmt_bytes(safe), cfg.safety_fraction * 100.0);
    }
    println!("  Recommendation: {rec_label}");
    println!("────────────────────────────────────────────────────");
    println!();
    println!("Choose execution mode:");
    println!("  [1] In-memory  (fast, needs ~{})", fmt_bytes(estimated));
    println!("  [2] Streaming  (slower, ~2 GB RAM, writes edges to disk)");
    println!("  [Enter] Accept recommendation ({rec_label})");
    print!("> ");
    std::io::stdout().flush().ok();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();

    match input.trim() {
        "1" => ExecMode::InMemory,
        "2" => ExecMode::Streaming,
        _ => recommended,
    }
}
