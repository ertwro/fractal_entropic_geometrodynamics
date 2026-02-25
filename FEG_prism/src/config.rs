// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! RAM-aware execution mode selection and TOML configuration.

use sysinfo::System;
use std::io::Write;
use std::path::Path;

/// Execution mode for the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    InMemory,
    Streaming,
}

/// User-configurable settings loaded from `causal_set.toml`.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_ram_bytes: u64,
    pub safety_fraction: f64,
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

const DEFAULT_CONFIG: &str = "\
# FEG Prism Simulation — configuration
#
# output_dir: Directory for results and streaming edge files.
#   CLI positional argument overrides this.
output_dir = \".\"

# max_ram_gb: Hard limit on RAM (in GB). 0 = auto-detect.
max_ram_gb = 0

# safety_fraction: Fraction of available RAM considered safe (0.0 to 1.0).
safety_fraction = 0.70
";

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

fn parse_config(contents: &str) -> Config {
    let mut cfg = Config::default();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
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
                        cfg.safety_fraction = frac.clamp(0.001, 1.0);
                    }
                }
                "output_dir" => {
                    let v = value.trim_matches('"').trim_matches('\'');
                    cfg.output_dir = v.to_string();
                }
                _ => {}
            }
        }
    }
    cfg
}

pub fn ensure_config(dir: &str) {
    let path = format!("{dir}/causal_set.toml");
    if !Path::new(&path).exists() {
        if let Ok(()) = std::fs::write(&path, DEFAULT_CONFIG) {
            println!("  (created default config: {path})");
        }
    }
}

pub fn available_ram_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

pub fn total_ram_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory()
}

pub fn estimate_memory_bytes(n: usize) -> u64 {
    if n <= 3_000 {
        (n as u64) * (n as u64) * 24
    } else if n <= 50_000 {
        (n as u64) * 2_000
    } else {
        (n as u64) * 1_000
    }
}

pub fn estimate_streaming_memory_bytes(n: usize) -> u64 {
    (n as u64) * 13 + 100 * 1024 * 1024
}

/// Estimate how many concurrent in-memory runs fit in available RAM.
pub fn max_concurrent_runs(n: usize, cfg: &Config) -> usize {
    let available = available_ram_bytes();
    let safe = if cfg.max_ram_bytes > 0 {
        cfg.max_ram_bytes
    } else {
        (available as f64 * cfg.safety_fraction) as u64
    };
    let per_run = estimate_memory_bytes(n);
    if per_run == 0 { return 4; }
    (safe / per_run).max(1) as usize
}

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

pub fn recommend_mode(n: usize, cfg: &Config) -> (ExecMode, u64, u64) {
    let available = available_ram_bytes();
    let concurrent = max_concurrent_runs(n, cfg) as u64;
    let estimated = estimate_memory_bytes(n) * concurrent;
    let ceiling = if cfg.max_ram_bytes > 0 {
        cfg.max_ram_bytes
    } else {
        (available as f64 * cfg.safety_fraction) as u64
    };
    let mode = if estimated <= ceiling { ExecMode::InMemory } else { ExecMode::Streaming };
    (mode, available, estimated)
}

pub fn prompt_mode(recommended: ExecMode, available: u64, estimated: u64, cfg: &Config) -> ExecMode {
    let total = total_ram_bytes();
    let rec_label = match recommended {
        ExecMode::InMemory => "in-memory",
        ExecMode::Streaming => "streaming",
    };

    println!("\u{2500}\u{2500} Memory \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
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
    println!("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    println!();
    println!("Choose execution mode:");
    println!("  [1] In-memory  (fast, needs ~{})", fmt_bytes(estimated));
    println!("  [2] Streaming  (slower, ~2 GB RAM)");
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
