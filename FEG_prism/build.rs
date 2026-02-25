// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

fn main() {
    let hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    let dirty = std::process::Command::new("git")
        .args(["diff", "--quiet"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(false);
    println!("cargo:rustc-env=GIT_HASH={}", hash.trim());
    println!(
        "cargo:rustc-env=GIT_DIRTY={}",
        if dirty { "dirty" } else { "clean" }
    );
    println!(
        "cargo:rustc-env=PROVENANCE_PREIMAGE={}",
        "Juan Pablo Silva Alvarado:10.5281/zenodo.18733424:FEG-Kuratowski-2026"
    );
}
