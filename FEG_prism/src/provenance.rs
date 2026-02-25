// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Five-layer provenance system binding the codebase to its author and DOI.
//!
//! | Layer | Purpose |
//! |-------|---------|
//! | 1 | Compile-time SHA-256 canary (`#[test]`) |
//! | 2 | Runtime verification at startup |
//! | 3 | CSV/output file header stamps |
//! | 4 | Checkpoint provenance (reject cross-fork resume) |
//! | 5 | Build-time `env!()` embedding (git hash + preimage) |

// ── Layer 1 & 4: Canary constants ──────────────────────────────────────────

pub const PROVENANCE_PREIMAGE: &[u8] =
    b"Juan Pablo Silva Alvarado:10.5281/zenodo.18733424:FEG-Kuratowski-2026";

pub const PROVENANCE_HASH: [u8; 32] = [
    0xa4, 0xfe, 0xe3, 0x37, 0x37, 0x30, 0x5b, 0x5e,
    0xb7, 0x8a, 0x16, 0xbe, 0x2f, 0xf2, 0x33, 0xbd,
    0xce, 0x12, 0x5b, 0xff, 0xa3, 0x52, 0x12, 0x7f,
    0xcb, 0xcd, 0x99, 0x7d, 0x01, 0x8c, 0x05, 0xaa,
];

pub const PROVENANCE_HASH_HEX: &str =
    "a4fee33737305b5eb78a16be2ff233bdce125bffa352127fcbcd997d018c05aa";

// ── Layer 5: Build-time embedding ──────────────────────────────────────────

pub fn git_hash() -> &'static str {
    env!("GIT_HASH")
}

pub fn git_dirty() -> bool {
    env!("GIT_DIRTY") == "dirty"
}

pub fn commit_string() -> String {
    if git_dirty() {
        format!("{} (dirty)", git_hash())
    } else {
        git_hash().to_string()
    }
}

// ── Layer 2: Runtime verification ──────────────────────────────────────────

/// Verify provenance at runtime. Call in main() before any simulation.
pub fn verify_provenance() -> bool {
    sha256(PROVENANCE_PREIMAGE) == PROVENANCE_HASH
}

/// Print provenance verification result to stdout.
pub fn print_provenance() {
    if verify_provenance() {
        println!(
            "Provenance: SHA-256 verified \u{2713} \
             (Juan Pablo Silva Alvarado, DOI:10.5281/zenodo.18769707)"
        );
    } else {
        eprintln!("WARNING: Provenance verification FAILED");
    }
}

// ── Layer 3: Output file headers ───────────────────────────────────────────

/// Generate the standard provenance header block for CSV/output files.
pub fn file_header(timestamp: &str) -> String {
    format!(
        "# FEG Kuratowski Calculus Engine (FEG_prism)\n\
         # Author: Juan Pablo Silva Alvarado\n\
         # DOI: 10.5281/zenodo.18769707\n\
         # Framework: Fractal Entropic Geometrodynamics\n\
         # Provenance SHA-256: {}\n\
         # Commit: {}\n\
         # Timestamp: {}",
        PROVENANCE_HASH_HEX,
        commit_string(),
        timestamp,
    )
}

// ── Minimal SHA-256 (no external crate) ────────────────────────────────────

const K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

pub fn sha256(msg: &[u8]) -> [u8; 32] {
    assert!(msg.len() <= 119, "two-block SHA-256 limit");
    let padded_len = if msg.len() + 9 <= 64 { 64 } else { 128 };
    let mut buf = vec![0u8; padded_len];
    buf[..msg.len()].copy_from_slice(msg);
    buf[msg.len()] = 0x80;
    let bit_len = (msg.len() as u64) * 8;
    buf[padded_len - 8..].copy_from_slice(&bit_len.to_be_bytes());

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    for chunk in buf.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4*i], chunk[4*i+1], chunk[4*i+2], chunk[4*i+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for i in 0..8 { out[4*i..4*i+4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}

// ── Timestamp utility ──────────────────────────────────────────────────────

pub fn utc_timestamp() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let (s, m, h) = (secs % 60, (secs / 60) % 60, (secs / 3600) % 24);
    let days = secs / 86400;
    let (y, mo, dy) = {
        let mut y = 1970u64;
        let mut rem = days;
        loop {
            let ylen = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
            if rem < ylen { break; }
            rem -= ylen;
            y += 1;
        }
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let mdays = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut mo = 0u64;
        for &ml in &mdays {
            if rem < ml { break; }
            rem -= ml;
            mo += 1;
        }
        (y, mo + 1, rem + 1)
    };
    format!("{y:04}-{mo:02}-{dy:02}T{h:02}:{m:02}:{s:02}Z")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_canary() {
        let digest = sha256(PROVENANCE_PREIMAGE);
        assert_eq!(
            digest, PROVENANCE_HASH,
            "Provenance canary: embedded hash must match SHA-256 of author:DOI:framework triple"
        );
    }

    #[test]
    fn runtime_verify() {
        assert!(verify_provenance());
    }

    #[test]
    fn hex_matches_bytes() {
        let hex: String = PROVENANCE_HASH.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, PROVENANCE_HASH_HEX);
    }
}
