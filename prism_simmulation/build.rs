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
}
