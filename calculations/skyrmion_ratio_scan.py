#!/usr/bin/env python3
"""
Skyrmion Mass Ratio Scan vs Core Size (λ_decay)
Memory-efficient version with vacuum fluctuations for finite ratios
N = 20,000 nodes, KDTree neighbors, max_dist=2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from numba import jit
from tqdm import tqdm
import time

np.random.seed(42)

# =============================================================================
# SU(2) utilities (Numba JIT)
# =============================================================================

@jit(nopython=True, cache=True)
def trace_diff(U1, U2):
    diff = U2 - U1
    return (diff[0,0].real**2 + diff[0,0].imag**2 +
            diff[0,1].real**2 + diff[0,1].imag**2 +
            diff[1,0].real**2 + diff[1,0].imag**2 +
            diff[1,1].real**2 + diff[1,1].imag**2)

@jit(nopython=True, cache=True)
def trace_commutator(U1, U2):
    comm00 = U1[0,0]*U2[0,0] + U1[0,1]*U2[1,0] - U2[0,0]*U1[0,0] - U2[0,1]*U1[1,0]
    comm01 = U1[0,0]*U2[0,1] + U1[0,1]*U2[1,1] - U2[0,0]*U1[0,1] - U2[0,1]*U1[1,1]
    comm10 = U1[1,0]*U2[0,0] + U1[1,1]*U2[1,0] - U2[1,0]*U1[0,0] - U2[1,1]*U1[1,0]
    comm11 = U1[1,0]*U2[0,1] + U1[1,1]*U2[1,1] - U2[1,0]*U1[0,1] - U2[1,1]*U1[1,1]
    return (comm00.real**2 + comm00.imag**2 +
            comm01.real**2 + comm01.imag**2 +
            comm10.real**2 + comm10.imag**2 +
            comm11.real**2 + comm11.imag**2)

@jit(nopython=True, cache=True)
def su2_from_hedgehog(F, r_hat):
    cosF = np.cos(F)
    sinF = np.sin(F)
    rx, ry, rz = r_hat
    U = np.zeros((2, 2), dtype=np.complex128)
    U[0,0] = cosF + 1j * sinF * rz
    U[0,1] = 1j * sinF * (rx - 1j*ry)
    U[1,0] = 1j * sinF * (rx + 1j*ry)
    U[1,1] = cosF - 1j * sinF * rz
    return U

# =============================================================================
# Causal set generation
# =============================================================================

def generate_causal_set(N=20000, radius=8.0):
    points = []
    while len(points) < N:
        p = np.random.uniform(-radius, radius, (10000, 4))
        r = np.sqrt(np.sum(p[:,1:]**2, axis=1))
        inside = (r < radius) & (p[:,0] > 0)
        points.append(p[inside])
    points = np.vstack(points)[:N].astype(np.float64)
    points = points[np.argsort(points[:,0])]
    tree = cKDTree(points[:,1:])
    return points, tree

# =============================================================================
# Field generation
# =============================================================================

def hedgehog_field_su2(points, lambda_decay):
    r_vec = points[:,1:]
    r_norm = np.sqrt(np.sum(r_vec**2, axis=1))
    r_norm = np.maximum(r_norm, 1e-6)
    r_hat = r_vec / r_norm[:, None]
    F = np.pi * np.exp(-r_norm / lambda_decay)
    
    U_field = np.zeros((len(points), 2, 2), dtype=np.complex128)
    for i in range(len(points)):
        U_field[i] = su2_from_hedgehog(F[i], r_hat[i])
    return U_field

def zero_mode_field(N, fluctuation_amplitude=0.01):
    U_zero = np.zeros((N, 2, 2), dtype=np.complex128)
    for i in range(N):
        phase = np.random.uniform(-fluctuation_amplitude, fluctuation_amplitude)
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        U_zero[i, 0, 0] = cos_p + 1j * sin_p
        U_zero[i, 1, 1] = cos_p - 1j * sin_p
    return U_zero

# =============================================================================
# Energy computation
# =============================================================================

def compute_energy(points, U_field, tree, max_dist=2.0):
    N = len(points)
    kinetic = 0.0
    quartic = 0.0
    
    neighbor_lists = tree.query_ball_point(points[:, 1:], max_dist)
    
    for i in range(N):
        ti = points[i, 0]
        xi = points[i, 1:4]
        U_i = U_field[i]
        
        for j in neighbor_lists[i]:
            if j <= i: continue
            tj = points[j, 0]
            dt = tj - ti
            if dt <= 0: continue
            dist = np.sqrt(np.sum((points[j, 1:4] - xi)**2))
            if dist >= dt: continue
            
            U_j = U_field[j]
            kinetic += trace_diff(U_i, U_j) / (dt * dt)
            quartic += trace_commutator(U_i, U_j)
    
    return kinetic + quartic

# =============================================================================
# PARAMETER SCAN
# =============================================================================

if __name__ == "__main__":
    N_nodes = 20000
    radius = 8.0
    fluctuation_amplitude = 0.01
    lambda_decays = np.linspace(0.6, 3.0, 8)
    mean_ratios = []
    std_ratios = []
    
    print(f"Memory-efficient scan with N={N_nodes}, max_dist=2.0")
    print("Starting parameter scan over λ_decay...\n")
    
    start_total = time.time()
    
    for ld in lambda_decays:
        print(f"λ_decay = {ld:.2f}")
        ratios_run = []
        
        for run in range(2):
            t0 = time.time()
            points, tree = generate_causal_set(N=N_nodes, radius=radius)
            
            U_soliton = hedgehog_field_su2(points, ld)
            U_zero = zero_mode_field(N_nodes, fluctuation_amplitude)
            
            E_soliton = compute_energy(points, U_soliton, tree, max_dist=2.0)
            E_zero = compute_energy(points, U_zero, tree, max_dist=2.0)
            
            ratio = E_soliton / E_zero if E_zero > 0 else np.inf
            ratios_run.append(ratio)
            
            print(f"   Run {run+1}: {time.time()-t0:.1f}s, ratio={ratio:.1f}x")
        
        mean_ratios.append(np.mean(ratios_run))
        std_ratios.append(np.std(ratios_run))
        print(f"   → Mean = {mean_ratios[-1]:.1f} ± {std_ratios[-1]:.1f}x\n")
    
    print(f"Total scan time: {time.time() - start_total:.1f}s")
    
    # Plot
    plt.figure(figsize=(9, 6))
    plt.errorbar(lambda_decays, mean_ratios, yerr=std_ratios, fmt='o-',
                 color='crimson', lw=2.5, markersize=8, capsize=5, label='Computed ratio')
    plt.axhline(1836, color='black', ls='--', alpha=0.7, lw=2, label='Observed m_p/m_e = 1836')
    plt.xlabel('Core decay length λ_decay')
    plt.ylabel('Geometric mass ratio m_p / m_e')
    plt.title('Skyrmion Mass Ratio vs Core Size on Causal Set')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('skyrmion_ratio_vs_lambda_decay.pdf', dpi=400, bbox_inches='tight')
    plt.savefig('skyrmion_ratio_vs_lambda_decay.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Scan complete. Plots saved.")
