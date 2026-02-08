#!/usr/bin/env python3
"""
Discrete Skyrmion on Causal Set — Memory-Efficient SU(2) Version with Vacuum Fluctuations
Now gives finite ratios
"""

import numpy as np
from scipy.spatial import cKDTree
from numba import jit, prange, complex128, float64
import time
from tqdm import tqdm

np.random.seed(42)

# =============================================================================
# SU(2) utilities
# =============================================================================

@jit(nopython=True, cache=True)
def su2_from_hedgehog(F, r_hat):
    cosF = np.cos(F)
    sinF = np.sin(F)
    rx, ry, rz = r_hat
    U = np.zeros((2, 2), dtype=complex128)
    U[0,0] = cosF + 1j * sinF * rz
    U[0,1] = 1j * sinF * (rx - 1j*ry)
    U[1,0] = 1j * sinF * (rx + 1j*ry)
    U[1,1] = cosF - 1j * sinF * rz
    return U

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

# =============================================================================
# Causal set generation
# =============================================================================

def generate_causal_set(N=20000, radius=8.0):
    print(f"Generating causal set N={N}...")
    start = time.time()
    
    points = []
    batch_size = 10000
    while len(points) < N:
        p = np.random.uniform(-radius, radius, (batch_size, 4))
        r = np.sqrt(np.sum(p[:,1:]**2, axis=1))
        inside = (r < radius) & (p[:,0] > 0)
        points.append(p[inside])
    
    points = np.vstack(points)[:N].astype(np.float64)
    idx = np.argsort(points[:,0])
    points = points[idx]
    
    tree = cKDTree(points[:,1:])
    
    print(f"Done in {time.time()-start:.1f}s. Nodes: {len(points)}")
    return points, tree

# =============================================================================
# Field generation
# =============================================================================

def hedgehog_field_su2(points, lambda_decay=1.8):
    print("Assigning SU(2) hedgehog field...")
    r_vec = points[:,1:]
    r_norm = np.sqrt(np.sum(r_vec**2, axis=1))
    r_norm = np.maximum(r_norm, 1e-6)
    r_hat = r_vec / r_norm[:, None]
    F = np.pi * np.exp(-r_norm / lambda_decay)
    
    U_field = np.zeros((len(points), 2, 2), dtype=np.complex128)
    for i in tqdm(range(len(points)), desc="Hedgehog matrices"):
        U_field[i] = su2_from_hedgehog(F[i], r_hat[i])
    return U_field

def zero_mode_field(N, fluctuation_amplitude=0.01):
    """Zero-mode with small random phase per node (vacuum fluctuation)"""
    print("Assigning zero-mode field with small fluctuations...")
    U_zero = np.zeros((N, 2, 2), dtype=np.complex128)
    for i in tqdm(range(N)):
        # Small random phase (mimics vacuum zero-point)
        phase = np.random.uniform(-fluctuation_amplitude, fluctuation_amplitude)
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        U_zero[i, 0, 0] = cos_p + 1j * sin_p
        U_zero[i, 1, 1] = cos_p - 1j * sin_p
        U_zero[i, 0, 1] = 0j
        U_zero[i, 1, 0] = 0j
    return U_zero

# =============================================================================
# Energy computation
# =============================================================================

def compute_energy(points, U_field, tree, max_dist=2.0):
    N = len(points)
    kinetic = 0.0
    quartic = 0.0
    
    print("Computing energy (KDTree neighbors)...")
    neighbor_lists = tree.query_ball_point(points[:, 1:], max_dist)
    
    for i in tqdm(range(N), desc="Nodes"):
        ti = points[i, 0]
        xi = points[i, 1:4]
        U_i = U_field[i]
        
        neighbors = neighbor_lists[i]
        for j in neighbors:
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

def run_skyrmion(N=20000, radius=8.0, lambda_decay=1.8, fluctuation_amplitude=0.01):
    start = time.time()
    
    points, tree = generate_causal_set(N=N, radius=radius)
    
    print("Computing soliton field...")
    U_soliton = hedgehog_field_su2(points, lambda_decay)
    
    print("Computing soliton energy...")
    E_soliton = compute_energy(points, U_soliton, tree, max_dist=2.0)
    
    print("Computing zero-mode baseline with small fluctuations...")
    U_zero = zero_mode_field(N, fluctuation_amplitude)
    E_zero = compute_energy(points, U_zero, tree, max_dist=2.0)
    
    ratio = E_soliton / E_zero if E_zero > 0 else np.inf
    
    print("\n" + "="*70)
    print(f"N = {N}, λ_decay = {lambda_decay}, fluctuation_amplitude = {fluctuation_amplitude}")
    print(f"Soliton energy   = {E_soliton:.2f}")
    print(f"Zero-mode energy = {E_zero:.2f}")
    print(f"Ratio            = {ratio:.1f}x")
    print(f"Target           = 1836x")
    print(f"Hierarchy factor = {ratio / 1836:.3f}")
    print(f"Total time       = {time.time() - start:.1f} s")
    print("="*70)
    
    return ratio, E_soliton, E_zero

if __name__ == "__main__":
    print("Running with N=20000, λ_decay=1.5, fluctuation=0.01")
    run_skyrmion(N=20000, radius=8.0, lambda_decay=1.5, fluctuation_amplitude=0.01)
    
    print("\nRunning with sharper core λ_decay=0.8")
    run_skyrmion(N=20000, radius=8.0, lambda_decay=0.8, fluctuation_amplitude=0.01)
