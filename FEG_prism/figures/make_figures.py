#!/usr/bin/env python3
"""Unified figure generator for the FEG paper.

Usage:
    python FEG_prism/figures/make_figures.py --data data/ensemble_10M_final --all
    python FEG_prism/figures/make_figures.py --data data/ensemble_10M_final --fig spectral mass
    python FEG_prism/figures/make_figures.py --list
"""

import argparse
import importlib
import inspect
import sys
import traceback

# ── Registry: name → (module_import_path, function, needs_fss) ──────────────
REGISTRY = {
    "spectral":    ("fig_spectral",    ["fig1", "fig2", "fig3"], False),
    "mass":        ("fig_mass",        ["fig4"],                 False),
    "dark_matter": ("fig_dark_matter", ["fig5"],                 False),
    "census":      ("fig_census",      ["fig6"],                 False),
    "flux":        ("fig_flux",        ["fig7"],                 False),
    "cpt":         ("fig_cpt",         ["fig8"],                 False),
    "composite":   ("fig_composite",   ["composite"],            False),
    "fss":         ("fig_fss",         ["fss"],                  True),
    "born_rule":   ("fig_born_rule",   ["born_rule"],            False),
    "vacuum_pol":  ("fig_vacuum_pol",
                    ["vp_mass_vs_charge", "vp_mu_effective",
                     "vp_q_running", "vp_inv_alpha"],            True),
    "tables":      ("tables",          ["summary"],              False),
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified figure generator for FEG paper.")
    parser.add_argument("--data", type=str,
                        help="Path to FEG simulation output directory")
    parser.add_argument("--out", type=str, default="paper/figures",
                        help="Output directory for PDFs/PNGs (default: paper/figures)")
    parser.add_argument("--all", action="store_true",
                        help="Generate all figures and tables")
    parser.add_argument("--fig", nargs="+", action="append", default=[],
                        help="Figure name(s) to generate (repeatable)")
    parser.add_argument("--fss-json", type=str, default=None,
                        help="Path to fss_comprehensive_results.json")
    parser.add_argument("--list", action="store_true",
                        help="List available figure names and exit")
    args = parser.parse_args()

    if args.list:
        print("Available figure names:")
        for name, (mod, funcs, needs_fss) in sorted(REGISTRY.items()):
            fss_note = "  (needs --fss-json)" if needs_fss else ""
            func_list = ", ".join(funcs)
            print(f"  {name:14s}  [{func_list}]{fss_note}")
        return

    if not args.data:
        parser.error("--data is required (unless --list)")

    # Flatten --fig lists
    fig_names = [name for group in args.fig for name in group]

    if not args.all and not fig_names:
        parser.error("Specify --all or --fig NAME [NAME ...]")

    targets = sorted(REGISTRY.keys()) if args.all else fig_names

    # Validate names
    for name in targets:
        if name not in REGISTRY:
            print(f"Unknown figure name: '{name}'")
            print(f"Available: {', '.join(sorted(REGISTRY.keys()))}")
            sys.exit(1)

    # ── Load data & apply style ──
    from .loader import FEGData
    from .style import apply_style

    apply_style()
    data = FEGData(args.data)
    print(f"FEG Figure Generator")
    print(f"  Data: {data.data_dir} ({data.M_label})")
    print(f"  Output: {args.out}")
    print()

    # ── Dispatch ──
    ok, fail = 0, 0
    for name in targets:
        mod_name, func_names, needs_fss = REGISTRY[name]
        try:
            mod = importlib.import_module(f".{mod_name}", package=__package__)
        except Exception:
            print(f"  [FAIL] {name}: could not import module")
            traceback.print_exc()
            fail += 1
            continue

        for func_name in func_names:
            try:
                fn = getattr(mod, func_name)
                sig = inspect.signature(fn)
                kwargs = {}
                if "fss_json" in sig.parameters:
                    kwargs["fss_json"] = args.fss_json
                fn(data, args.out, **kwargs)
                ok += 1
            except Exception:
                print(f"  [FAIL] {name}/{func_name}")
                traceback.print_exc()
                fail += 1

    print()
    print(f"Done: {ok} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
