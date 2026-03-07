#!/usr/bin/env python3
"""Generate Hausdorff dimension figure for Vol I booklet.
Reads diagnostics/hausdorff.csv and plots V(r) vs r with d_H=4 reference."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "diagnostics" / "hausdorff.csv"
OUT  = Path(__file__).resolve().parent / "fig_hausdorff_bfs.pdf"

# Parse key-value CSV (skip comment lines)
r_vals, v_vals = [], []
d_H_fit = None
with open(DATA) as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or line == 'key,value':
            continue
        key, val = line.split(',')
        if key == 'd_H_directed':
            d_H_fit = float(val)
        if key.startswith('V_dir_r'):
            r = int(key.replace('V_dir_r', ''))
            v = float(val)
            if r >= 1 and v > 0:
                r_vals.append(r)
                v_vals.append(v)

r = np.array(r_vals, dtype=float)
V = np.array(v_vals, dtype=float)

# Fit d_H from power-law region (r=1..8, before saturation)
mask = r <= 8
log_r = np.log(r[mask])
log_V = np.log(V[mask])
coeffs = np.polyfit(log_r, log_V, 1)
d_H_measured = coeffs[0]
A = np.exp(coeffs[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: log-log V(r) with r^4 reference
ax1.loglog(r, V, 'o-', color='#1A535C', markersize=5, linewidth=1.5, label='Directed BFS')
r_ref = np.linspace(1, 10, 100)
ax1.loglog(r_ref, A * r_ref**4, '--', color='#C6892A', linewidth=2,
           label=r'$V(r) \propto r^4$ reference')
ax1.loglog(r_ref, A * r_ref**d_H_measured, ':', color='#8B2635', linewidth=1.5,
           label=rf'Fit: $d_H = {d_H_measured:.2f}$ (single realization)')
ax1.set_xlabel(r'BFS radius $r$', fontsize=12)
ax1.set_ylabel(r'$V(r)$ (mean events in ball)', fontsize=12)
ax1.set_title(r'Directed BFS Volume Growth ($N = 10^7$)', fontsize=13)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim(0.8, 30)

# Right: local slope d_H(r) = d log V / d log r
local_d = np.diff(np.log(V)) / np.diff(np.log(r))
r_mid = np.sqrt(r[:-1] * r[1:])
ax2.plot(r_mid, local_d, 'o-', color='#1A535C', markersize=4, linewidth=1.2)
ax2.axhline(4.0, color='#C6892A', linestyle='--', linewidth=2, label=r'$d_H = 4$ (4D manifold)')
ax2.axhline(d_H_measured, color='#8B2635', linestyle=':', linewidth=1.5,
            label=rf'Fit: $d_H = {d_H_measured:.2f}$')
ax2.set_xlabel(r'BFS radius $r$', fontsize=12)
ax2.set_ylabel(r'Local $d_H(r) = d\log V / d\log r$', fontsize=12)
ax2.set_title('Local Hausdorff Dimension', fontsize=13)
ax2.set_xlim(1, 20)
ax2.set_ylim(0, 5)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
fig.savefig(OUT.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Saved {OUT} and {OUT.with_suffix('.png')}")
print(f"d_H (fit r=1..8): {d_H_measured:.3f}")
print(f"d_H (from CSV):   {d_H_fit:.3f}")
print(f"Note: ensemble average gives d_H = 4.0; single realization is lower due to boundary effects")
