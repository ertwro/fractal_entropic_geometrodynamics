"""Summary tables (plain text + LaTeX)."""

import pathlib

import numpy as np

from .style import savefig  # only for out_dir consistency


def summary(data, out_dir):
    """Generate summary_table.txt and table_observables.tex."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = data.results
    ms = data.mass_spectrum
    sigma = data.sigma

    N_total = data.N_total
    N_prisms = data.N_prisms
    N_gen1, N_gen2, N_gen3 = data.N_gen1, data.N_gen2, data.N_gen3
    N_anti1 = data.N_anti1
    m_gen1, m_gen2, m_gen3 = data.m_gen1, data.m_gen2, data.m_gen3
    m_anti1 = data.m_anti1
    M_LABEL = data.M_label

    P_total = N_gen1 + N_gen2 + N_gen3

    N_int = ms["intermediates_N"].values
    freq = ms["frequency"].values

    # Computed values
    ds_vac_uv = df["dS_vac"].values[0]
    ds_vac_core_idx = np.argmin(np.abs(sigma - 4))
    ds_vac_core = df["dS_vac"].values[ds_vac_core_idx]
    ds_def_peak = df["dS_def"].values[:15].max()

    attr_1 = df["Flux_Attr"].values[0]
    repu_1 = df["Flux_Repu"].values[0]
    flux_ratio = repu_1 / attr_1 if attr_1 > 0 else float("nan")

    vis_count = freq[N_int <= 5].sum()
    dark_count = freq[N_int > 5].sum()
    mass_contribution = N_int * freq
    total_mass = mass_contribution.sum()
    vis_mass = mass_contribution[N_int <= 5].sum()
    dark_mass = mass_contribution[N_int > 5].sum()
    dm_ratio = dark_mass / vis_mass if vis_mass > 0 else float("nan")

    cpt_dev = abs(m_gen1 - m_anti1) / m_gen1 * 100

    # ── Plain text ──
    lines = [
        "",
        "=" * 72,
        f"  SUMMARY TABLE — FEG Simulation at N = {N_total/1e6:.0f}M ({M_LABEL}, seed 42)",
        "=" * 72,
        "",
        f"  {'Observable':<42} {'Value':>12}  {'Note':<20}",
        f"  {'-'*42} {'-'*12}  {'-'*20}",
        "",
        "  SPECTRAL DIMENSION",
        f"  {'dS vacuum (UV, sigma=1)':<42} {ds_vac_uv:>12.4f}  {'expected ~2':20}",
        f"  {'dS vacuum (core, sigma=4)':<42} {ds_vac_core:>12.4f}  {'approaching 4D':20}",
        f"  {'dS defect (peak)':<42} {ds_def_peak:>12.4f}  {'trapping > 4':20}",
        "",
        "  GENERATION CLASSIFICATION (prism counts)",
        f"  {'Total prisms':<42} {N_prisms:>12,}",
        f"  {'Gen 1 (g=1) prisms':<42} {N_gen1:>12,}  {f'{N_gen1/P_total:.1%}':20}",
        f"  {'Gen 2 (g=2) prisms':<42} {N_gen2:>12,}  {f'{N_gen2/P_total:.1%}':20}",
        f"  {'Gen 3 (g=3) prisms':<42} {N_gen3:>12,}  {f'{N_gen3/P_total:.1%}':20}",
        f"  {'Anti-Gen1 count':<42} {N_anti1:>12,}",
        "",
        "  MASS HIERARCHY",
        f"  {'Mass Gen 1':<42} {m_gen1:>12.4f}  {'lightest':20}",
        f"  {'Mass Gen 2':<42} {m_gen2:>12.4f}  {'':20}",
        f"  {'Mass Gen 3':<42} {m_gen3:>12.4f}  {'heaviest':20}",
        f"  {'Mass Anti-1':<42} {m_anti1:>12.4f}  {'':20}",
        f"  {'Ratio m1 : m2 : m3':<42} {'1 : {:.2f} : {:.2f}'.format(m_gen2/m_gen1, m_gen3/m_gen1):>12}",
        "",
        "  CPT TEST",
        f"  {'|m_Gen1 - m_Anti1| / m_Gen1':<42} {cpt_dev:>11.1f}%  {'converges w/ M':20}",
        "",
        "  DARK MATTER",
        f"  {'Visible prisms (n <= 5)':<42} {vis_count:>12,}",
        f"  {'Dark prisms (n > 5)':<42} {dark_count:>12,}",
        f"  {'Omega_dark / Omega_vis (mass-weighted)':<42} {dm_ratio:>12.2f}  {'observed: 5.4':20}",
        f"  {'Max belly size':<42} {N_int.max():>12d}",
        "",
        "  CAUSAL FLUX (sigma=1)",
        f"  {'Flux_Attraction':<42} {attr_1:>12.6e}",
        f"  {'Flux_Repulsion':<42} {repu_1:>12.6e}",
        f"  {'Repu / Attr (~ 1/alpha?)':<42} {flux_ratio:>12.1f}  {'SM: 137':20}",
        "",
        "  COMPUTATION",
        f"  {'Total events N':<42} {N_total:>12,}",
        "",
        "=" * 72,
    ]

    table = "\n".join(lines)
    print(table)

    with open(out / "summary_table.txt", "w") as f:
        f.write(table)
    print(f"\n  [+] summary_table.txt")

    # ── LaTeX ──
    latex = (
        r"\begin{table}[h]"
        "\n"
        r"\caption{Key observables at $N = 10^7$, $" + M_LABEL + r"""$ (seed\,42).}
\label{tab:full_results}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Observable} & \textbf{Value} & \textbf{Note} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Spectral dimension}} \\[2pt]
$d_S$ vacuum (UV, $\sigma{=}1$)     & """ + f"{ds_vac_uv:.2f}" + r""" & expected $\approx 2$ \\
$d_S$ vacuum (core, $\sigma{=}4$)   & """ + f"{ds_vac_core:.2f}" + r""" & approaching 4D \\
$d_S$ defect (peak)                 & """ + f"{ds_def_peak:.2f}" + r""" & trapping $> 4$ \\
\midrule
\multicolumn{3}{@{}l}{\emph{Generation classification}} \\[2pt]
Total prisms                        & """ + f"{N_prisms:,}" + r""" & \\
Gen\,1 ($g{=}1$) / Gen\,2 / Gen\,3 & """ + f"{N_gen1:,} / {N_gen2:,} / {N_gen3:,}" + r""" & """ + f"{N_gen1/P_total:.1%} / {N_gen2/P_total:.1%} / {N_gen3/P_total:.1%}" + r""" \\
Anti-Gen\,1                         & """ + f"{N_anti1:,}" + r""" & \\
\midrule
\multicolumn{3}{@{}l}{\emph{Mass hierarchy}} \\[2pt]
$m_1 : m_2 : m_3$                  & """ + f"$1 : {m_gen2/m_gen1:.2f} : {m_gen3/m_gen1:.2f}$" + r""" & topological units \\
Mass: Gen\,1 / Gen\,2 / Gen\,3     & """ + f"{m_gen1:.2f} / {m_gen2:.2f} / {m_gen3:.2f}" + r""" & \\
CPT: Gen\,1 vs Anti-1              & """ + f"{m_gen1:.2f} vs {m_anti1:.2f}" + r""" & $\Delta m/m = """ + f"{cpt_dev:.1f}" + r"""\%$ \\
\midrule
\multicolumn{3}{@{}l}{\emph{Dark matter}} \\[2pt]
$\Omega_{\mathrm{dark}} / \Omega_{\mathrm{vis}}$ & """ + f"{dm_ratio:.2f}" + r""" & observed: $5.4$ \\
Max belly size $n_{\max}$           & """ + f"{N_int.max()}" + r""" & \\
\midrule
\multicolumn{3}{@{}l}{\emph{Causal flux ($\sigma{=}1$)}} \\[2pt]
$F_{\mathrm{repu}} / F_{\mathrm{attr}}$ & """ + f"{flux_ratio:.1f}" + r""" & SM: $1/\alpha = 137$ \\
\bottomrule
\end{tabular}
\end{table}"""
    )

    with open(out / "table_observables.tex", "w") as f:
        f.write(latex)
    print(f"  [+] table_observables.tex")
