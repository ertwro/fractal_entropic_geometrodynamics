import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURACIÓN ESTÉTICA (Physical Review Style) ---
plt.style.use('seaborn-v0_8-paper')
params = {
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
}
plt.rcParams.update(params)

def analyze_and_plot():
    print("[*] Leyendo telemetría completa...")
    df_res = pd.read_csv('results.csv', comment='#')
    df_mass = pd.read_csv('mass_spectrum.csv', comment='#')
    df_top = pd.read_csv('topology_summary.csv', comment='#')
    top_dict = dict(zip(df_top['key'], df_top['value']))

    df_res['inv_alpha'] = df_res.apply(
        lambda row: row['Flux_Repu'] / row['Flux_Attr'] if row['Flux_Attr'] > 0 else np.nan, axis=1
    )

    n_nodes = int(top_dict.get('total_nodes', 0))
    metadata_str = f"N = {n_nodes:,}"

    # =========================================================
    # FIGURA 1: GEOMETRÍA MACROSCÓPICA (Para el Capítulo 2)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_res['step'], df_res['dS_vac'], '--', color='gray', label='Vacuum ($K_3$-free Base)')
    ax1.plot(df_res['step'], df_res['dS_def'], '-', color='black', linewidth=2, label='Global Defect Core')
    ax1.plot(df_res['step'], df_res['dS_Gen1'], '-', color='blue', label='Matter (Gen 1)')
    ax1.plot(df_res['step'], df_res['dS_Anti1'], '-.', color='orange', label='Antimatter')
    
    ax1.axhline(y=4.0, color='purple', linestyle=':', alpha=0.8, linewidth=2, label='4D Limit')
    ax1.set_title(f'Figure 1: Emergence of 4D Geometry ({metadata_str})')
    ax1.set_xlabel('Diffusion Time ($t$)')
    ax1.set_ylabel('Spectral Dimension ($d_S$)')
    ax1.set_xlim(0, 15)
    ax1.set_ylim(1, 8)
    ax1.legend()
    fig1.savefig('fig1_macroscopic_geometry.png')
    print(" -> Generado: fig1_macroscopic_geometry.png")

    # =========================================================
    # FIGURA 2: INERCIA TOPOLÓGICA (Para Emma y Capítulo 5)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(df_res['step'], df_res['P_vac'], '--', color='gray', label='Vacuum (No Delay)')
    ax2.plot(df_res['step'], df_res['P_Gen1'], '-', color='blue', label='Gen 1 (Inertial Delay)')
    ax2.plot(df_res['step'], df_res['P_Gen3'], '-', color='red', label='Gen 3 (Heavy Delay)')
    
    ax2.set_title(f'Figure 2: Topological Inertia via Walker Delay ({metadata_str})')
    ax2.set_xlabel('Diffusion Time ($t$)')
    ax2.set_ylabel('Return Probability $P(t)$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    fig2.savefig('fig2_topological_inertia.png')
    print(" -> Generado: fig2_topological_inertia.png")

    # =========================================================
    # FIGURA 3: ESPECTRO DE MATERIA (Para el Capítulo 5/6)
    # =========================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma
    ax3a.bar(df_mass['intermediates_N'], df_mass['frequency'], color='teal', edgecolor='black')
    ax3a.set_title('Mass Spectrum ($K_{2,N}$ Prisms)')
    ax3a.set_xlabel('Intermediates ($N$)')
    ax3a.set_ylabel('Frequency')
    ax3a.set_yscale('log')
    max_x = min(max(df_mass['intermediates_N']) + 1, 30)
    ax3a.set_xlim(2, max_x)
    
    # Asimetría
    counts = {
        'Antimatter': float(top_dict.get('count_antigen1', 0)),
        'Gen 3': float(top_dict.get('count_gen3', 0)),
        'Gen 2': float(top_dict.get('count_gen2', 0)),
        'Gen 1': float(top_dict.get('count_gen1', 0))
    }
    ax3b.bar(counts.keys(), counts.values(), color=['orange', 'red', 'green', 'blue'], edgecolor='black')
    ax3b.set_title('Matter vs Antimatter Asymmetry')
    ax3b.set_ylabel('Absolute Count')
    ax3b.set_yscale('log')

    fig3.suptitle(f'Figure 3: Topological Origin of Particles ({metadata_str})', fontsize=14, y=1.02)
    fig3.tight_layout()
    fig3.savefig('fig3_matter_spectrum.png')
    print(" -> Generado: fig3_matter_spectrum.png")

    # =========================================================
    # FIGURA 4: FUERZAS Y CONSTANTE DE ESTRUCTURA FINA
    # =========================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw Flux
    valid_flux = df_res[df_res['step'] <= 15]
    ax4a.plot(valid_flux['step'], valid_flux['Flux_Attr'], 'o-', color='blue', label='Attractive Flux')
    ax4a.plot(valid_flux['step'], valid_flux['Flux_Repu'], 's-', color='red', label='Repulsive Flux')
    ax4a.set_title('Raw Causal Flux (EM)')
    ax4a.set_xlabel('Time ($t$)')
    ax4a.set_ylabel('Transmission Probability')
    ax4a.set_yscale('log')
    ax4a.legend()

    # Alpha
    valid_alpha = df_res[(df_res['step'] >= 2) & (df_res['step'] <= 15)]
    ax4b.plot(valid_alpha['step'], valid_alpha['inv_alpha'], 'o-', color='darkred', label='Simulated $1/\\alpha$')
    ax4b.axhline(y=137.036, color='gold', linestyle='--', linewidth=2, label='SM Limit (137.036)')
    ax4b.set_title('Fine Structure Constant ($1/\\alpha$)')
    ax4b.set_xlabel('Time ($t$)')
    ax4b.set_ylabel('Flux Ratio')
    ax4b.set_ylim(40, 260)
    ax4b.legend()

    fig4.suptitle(f'Figure 4: Emergent Electromagnetism ({metadata_str})', fontsize=14, y=1.02)
    fig4.tight_layout()
    fig4.savefig('fig4_forces_and_alpha.png')
    print(" -> Generado: fig4_forces_and_alpha.png")
    
    print("[*] Todas las figuras han sido generadas y listas para LaTeX.")

if __name__ == "__main__":
    analyze_and_plot()
