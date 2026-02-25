"""Lazy data loader for FEG simulation output."""

import re
import pathlib
from glob import glob

import pandas as pd


class FEGData:
    """Lazy-loading wrapper for a FEG simulation data directory.

    Automatically detects the latest ``_M{nn}`` snapshot and exposes
    each CSV as a property that is read only on first access.
    """

    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Auto-detect latest M snapshot
        results_files = sorted(glob(str(self.data_dir / "results_M*.csv")))
        if results_files:
            self._latest_M = max(
                int(re.search(r"M(\d+)", f).group(1)) for f in results_files
            )
            self._pad = f"{self._latest_M:02d}"
        else:
            # Fallback: try results.csv (no M suffix)
            if (self.data_dir / "results.csv").exists():
                self._latest_M = None
                self._pad = None
            else:
                raise FileNotFoundError(
                    f"No results_M*.csv or results.csv found in {self.data_dir}"
                )

        self._cache = {}

    @property
    def M_label(self):
        return f"M={self._latest_M}" if self._latest_M else "M=?"

    def _csv_path(self, base):
        """Resolve CSV path, trying M-suffixed first then plain."""
        if self._pad is not None:
            p = self.data_dir / f"{base}_M{self._pad}.csv"
            if p.exists():
                return p
        p = self.data_dir / f"{base}.csv"
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Neither {base}_M{self._pad}.csv nor {base}.csv in {self.data_dir}"
        )

    def _load(self, key, base):
        if key not in self._cache:
            path = self._csv_path(base)
            self._cache[key] = pd.read_csv(path, comment="#")
        return self._cache[key]

    # ── Core CSVs ────────────────────────────────────────────────────────────

    @property
    def results(self):
        return self._load("results", "results")

    @property
    def mass_spectrum(self):
        return self._load("mass_spectrum", "mass_spectrum")

    @property
    def topology(self):
        """Topology summary as a dict mapping key → value."""
        if "topology" not in self._cache:
            df = self._load("_topology_df", "topology_summary")
            self._cache["topology"] = dict(zip(df["key"], df["value"]))
        return self._cache["topology"]

    # ── Optional CSVs (may not exist in every data directory) ────────────────

    def _load_optional(self, key, base):
        try:
            return self._load(key, base)
        except FileNotFoundError:
            return None

    @property
    def vacuum_pol(self):
        return self._load_optional("vacuum_pol", "vacuum_polarization")

    @property
    def electroweak(self):
        return self._load_optional("electroweak", "electroweak")

    @property
    def born_rule(self):
        return self._load_optional("born_rule", "born_rule")

    @property
    def decoherence(self):
        return self._load_optional("decoherence", "decoherence")

    @property
    def half_life(self):
        return self._load_optional("half_life", "half_life")

    @property
    def traversal_mass(self):
        return self._load_optional("traversal_mass", "traversal_mass")

    @property
    def neutrino(self):
        return self._load_optional("neutrino", "neutrino")

    @property
    def pmns(self):
        return self._load_optional("pmns", "pmns")

    @property
    def higgs(self):
        return self._load_optional("higgs", "higgs")

    @property
    def modulo_interference(self):
        return self._load_optional("modulo_interference", "modulo_interference")

    # ── Convenience scalars ──────────────────────────────────────────────────

    @property
    def sigma(self):
        return self.results["step"].values

    @property
    def N_total(self):
        return int(float(self.topology["total_nodes"]))

    @property
    def N_prisms(self):
        return int(float(self.topology["total_prisms"]))

    @property
    def N_gen1(self):
        return int(float(self.topology["count_gen1"]))

    @property
    def N_gen2(self):
        return int(float(self.topology["count_gen2"]))

    @property
    def N_gen3(self):
        return int(float(self.topology["count_gen3"]))

    @property
    def N_anti1(self):
        return int(float(self.topology["count_antigen1"]))

    @property
    def m_gen1(self):
        return float(self.topology["avg_mass_gen1"])

    @property
    def m_gen2(self):
        return float(self.topology["avg_mass_gen2"])

    @property
    def m_gen3(self):
        return float(self.topology["avg_mass_gen3"])

    @property
    def m_anti1(self):
        return float(self.results["Mass_Anti1"].iloc[0])
