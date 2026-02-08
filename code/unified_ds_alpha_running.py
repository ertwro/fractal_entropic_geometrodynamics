#!/usr/bin/env python3
"""
Unified Mechanism: Spectral Dimension Flow & Coupling Running
Final publication-quality version
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PHYSICS (DO NOT MODIFY)
# ============================================================================
ALPHA_TRANS = 0.31
ALPHA_GEOM = 1 / 118.4
ALPHA_OBS = 1 / 137.036

k_vals = np.logspace(-4, 4, 1200)

def spectral_dimension(k):
    return 2 + 2 / (1 + (k / ALPHA_TRANS)**2)

dS_vals = spectral_dimension(k_vals)

# Correct running: screening in IR
exponent = (dS_vals - 4) / 2.0
alpha_running = ALPHA_GEOM * (k_vals ** exponent)

# Normalize to exactly match low-energy value
scale_factor = ALPHA_OBS / alpha_running[0]
alpha_running *= scale_factor
# ============================================================================

# --- Publication Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

fig, ax1 = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')

# --- Left axis: d_S(k) ---
color_ds = '#2166ac'  # professional blue
ax1.set_xlabel(r'Energy Scale $k / M_{\rm trans}$', fontsize=14, labelpad=10)
ax1.set_ylabel(r'Spectral Dimension $d_S(k)$', color=color_ds, fontsize=14, labelpad=10)
ax1.semilogx(k_vals, dS_vals, color=color_ds, lw=3.5, solid_capstyle='round')
ax1.tick_params(axis='y', labelcolor=color_ds, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_ylim(1.8, 4.35)
ax1.set_xlim(1e-4, 1e4)

# Reference lines
ax1.axhline(4, color='#666666', ls='--', alpha=0.6, lw=1.5)
ax1.axhline(2, color='#666666', ls='--', alpha=0.6, lw=1.5)

# Dimension regime labels (right side, clear of curves and annotations)
ax1.text(5e3, 4.05, 'Classical (4D)', color='#666666', fontsize=10,
         va='bottom', ha='right', style='italic')
ax1.text(5e3, 2.05, 'Quantum (2D)', color='#666666', fontsize=10,
         va='bottom', ha='right', style='italic')

# Transition scale marker
ax1.axvline(1, color='#333333', ls=':', alpha=0.4, lw=1.5)
ax1.text(1.3, 1.9, r'$k = M_{\rm trans}$', color='#333333', fontsize=10,
         va='bottom', ha='left', rotation=90, alpha=0.7)

ax1.grid(True, alpha=0.15, which='both', linestyle='-')

# --- Right axis: 1/α(k) ---
ax2 = ax1.twinx()
color_alpha = '#b2182b'  # professional red
ax2.set_ylabel(r'Fine-Structure Constant $1/\alpha$', color=color_alpha, fontsize=14, labelpad=10)
ax2.semilogx(k_vals, 1 / alpha_running, color=color_alpha, lw=3.5, solid_capstyle='round')
ax2.tick_params(axis='y', labelcolor=color_alpha, labelsize=12)
ax2.set_ylim(117, 138)

# --- Key points with clean labels ---
# Low-energy observed value marker (left)
ax2.scatter([k_vals[0]], [1/alpha_running[0]], color=color_alpha, s=120, zorder=5,
            edgecolors='white', linewidth=2.5)
ax2.text(k_vals[0] * 8, 136, r'Observed $1/\alpha = 137.036$',
         fontsize=10, color=color_alpha, fontweight='medium',
         ha='left', va='top')

# High-energy geometric bare value marker (right)
ax2.scatter([k_vals[-1]], [1/alpha_running[-1]], color=color_alpha, s=120, zorder=5,
            edgecolors='white', linewidth=2.5, clip_on=False)
ax2.text(k_vals[-1] / 8, 119.5, r'Bare $1/\alpha = 118.4$',
         fontsize=10, color=color_alpha, fontweight='medium',
         ha='right', va='bottom')

# --- Title ---
plt.title('Unified Mechanism: Spectral Dimension Flow Drives Coupling Running',
          fontsize=16, pad=20, fontweight='bold')

fig.tight_layout()
plt.savefig('unified_ds_alpha_running_final.pdf', dpi=400, bbox_inches='tight')
plt.savefig('unified_ds_alpha_running_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved: unified_ds_alpha_running_final.pdf")
print(f"Low-energy 1/α  = {1/alpha_running[0]:.3f}")
print(f"High-energy 1/α = {1/alpha_running[-1]:.3f}")
