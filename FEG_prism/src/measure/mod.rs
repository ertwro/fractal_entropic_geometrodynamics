// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Measurement dispatch: M1–M10 observables.

pub mod context;
pub mod m01_traversal;
pub mod m02_halflife;
pub mod m03_modulo;
pub mod m04_vacuum_pol;
pub mod m05_electroweak;
pub mod m06_decoherence;
pub mod m07_neutrino;
pub mod m08_pmns;
pub mod m09_higgs;
pub mod m10_lagrangian;

pub use context::MeasureContext;

/// Flags controlling which measurements to run.
#[derive(Clone)]
pub struct MeasureFlags {
    pub mass: bool,
    pub halflife: bool,
    pub modulo: bool,
    pub vacuum: bool,
    pub electroweak: bool,
    pub decoherence: bool,
    pub neutrino: bool,
    pub pmns: bool,
    pub higgs: bool,
    pub lagrangian: bool,
    pub modulo_config: ModuloConfig,
    /// Run M6 (decoherence) only every N-th realization.  Default: 1 (every).
    pub decoherence_every: usize,
}

impl MeasureFlags {
    pub fn any_active(&self) -> bool {
        self.mass || self.halflife || self.modulo || self.vacuum
            || self.electroweak || self.decoherence || self.neutrino
            || self.pmns || self.higgs || self.lagrangian
    }
}

/// NTT configuration for M3/M6 modulo path integral.
#[derive(Debug, Clone, Copy)]
pub struct ModuloConfig {
    pub prime: u64,
    pub root: u64,
    pub steps: u32,
}

/// Aggregated measurement results for one realization.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MeasureResults {
    pub traversal: Option<m01_traversal::TraversalMassResult>,
    pub half_life: Option<m02_halflife::HalfLifeResult>,
    pub modulo: Option<m03_modulo::ModuloPathResult>,
    pub vacuum_pol: Option<m04_vacuum_pol::VacuumPolResult>,
    pub electroweak: Option<m05_electroweak::ElectroweakResult>,
    pub decoherence: Option<m06_decoherence::DecoherenceResult>,
    pub neutrino: Option<m07_neutrino::NeutrinoResult>,
    pub pmns: Option<m08_pmns::PMNSResult>,
    pub higgs: Option<m09_higgs::HiggsResult>,
    /// M10 is computed post-hoc in main.rs after the ensemble finishes.
    /// Skipped during checkpoint serialization (always None in checkpoints).
    #[serde(skip)]
    pub lagrangian: Option<m10_lagrangian::LagrangianCard>,
}

/// Run all enabled measurements on a single realization.
///
/// Independent measurements (M1–M6, M9) run concurrently via
/// `std::thread::scope`.  M7→M8 remains sequential (M8 depends on M7).
pub fn run_all(flags: &MeasureFlags, ctx: &MeasureContext) -> MeasureResults {
    // ── Independent measurements — run concurrently ──────────────────
    let (traversal, half_life, modulo, vacuum_pol, electroweak, decoherence, higgs) =
        std::thread::scope(|s| {
            let h1 = s.spawn(|| if flags.mass {
                println!("  [M1] Traversal mass ratios (prism-confined)...");
                Some(m01_traversal::run(ctx))
            } else { None });
            let h2 = s.spawn(|| if flags.halflife {
                println!("  [M2] Half-life census...");
                Some(m02_halflife::run(ctx))
            } else { None });
            let h3 = s.spawn(|| if flags.modulo {
                println!("  [M3] Modulo path integral...");
                Some(m03_modulo::run(ctx))
            } else { None });
            let h4 = s.spawn(|| if flags.vacuum {
                println!("  [M4] Vacuum polarization...");
                Some(m04_vacuum_pol::run(ctx))
            } else { None });
            let h5 = s.spawn(|| if flags.electroweak {
                println!("  [M5] Electroweak sector...");
                Some(m05_electroweak::run(ctx))
            } else { None });
            let h6 = s.spawn(|| if flags.decoherence {
                println!("  [M6] Quantum decoherence...");
                Some(m06_decoherence::run(ctx))
            } else { None });
            let h7 = s.spawn(|| if flags.higgs {
                println!("  [M9] Higgs mechanism (topological drag)...");
                Some(m09_higgs::run(ctx))
            } else { None });
            (
                h1.join().unwrap(),
                h2.join().unwrap(),
                h3.join().unwrap(),
                h4.join().unwrap(),
                h5.join().unwrap(),
                h6.join().unwrap(),
                h7.join().unwrap(),
            )
        });

    // ── Sequential: M7→M8 dependency chain ───────────────────────────
    let neutrino = if flags.neutrino || flags.pmns {
        println!("  [M7] Neutrino census...");
        Some(m07_neutrino::run(ctx))
    } else { None };

    let pmns = if flags.pmns {
        if let Some(ref nu) = neutrino {
            println!("  [M8] PMNS mixing matrix...");
            Some(m08_pmns::run(ctx, nu))
        } else { None }
    } else { None };

    MeasureResults {
        traversal, half_life, modulo, vacuum_pol, electroweak,
        decoherence, neutrino, pmns, higgs, lagrangian: None,
    }
}

/// Aggregate measurement results across ensemble realizations.
pub fn aggregate_all(results: &[MeasureResults]) -> MeasureResults {
    MeasureResults {
        traversal: collect_and_agg(results, |r| r.traversal.as_ref(), m01_traversal::aggregate),
        half_life: collect_and_agg(results, |r| r.half_life.as_ref(), m02_halflife::aggregate),
        modulo: collect_and_agg(results, |r| r.modulo.as_ref(), m03_modulo::aggregate),
        vacuum_pol: collect_and_agg(results, |r| r.vacuum_pol.as_ref(), m04_vacuum_pol::aggregate),
        electroweak: collect_and_agg(results, |r| r.electroweak.as_ref(), m05_electroweak::aggregate),
        decoherence: collect_and_agg(results, |r| r.decoherence.as_ref(), m06_decoherence::aggregate),
        neutrino: collect_and_agg(results, |r| r.neutrino.as_ref(), m07_neutrino::aggregate),
        pmns: collect_and_agg(results, |r| r.pmns.as_ref(), m08_pmns::aggregate),
        higgs: collect_and_agg(results, |r| r.higgs.as_ref(), m09_higgs::aggregate),
        lagrangian: None,
    }
}

fn collect_and_agg<T: Clone>(
    results: &[MeasureResults],
    extract: impl Fn(&MeasureResults) -> Option<&T>,
    agg: impl Fn(&[T]) -> T,
) -> Option<T> {
    let items: Vec<T> = results.iter().filter_map(|r| extract(r).cloned()).collect();
    if items.is_empty() { None }
    else if items.len() == 1 { Some(items.into_iter().next().unwrap()) }
    else { Some(agg(&items)) }
}

/// Write all measurement CSVs via per-measurement dispatch.
pub fn write_all_csv(
    results: &MeasureResults,
    dir: &str,
    meta: &crate::output::Metadata,
) {
    use crate::output::CsvWriter;
    if let Some(ref r) = results.traversal {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/traversal_mass.csv"), meta) {
            m01_traversal::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.half_life {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/half_life.csv"), meta) {
            m02_halflife::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.modulo {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/modulo_interference.csv"), meta) {
            m03_modulo::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.vacuum_pol {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/vacuum_polarization.csv"), meta) {
            m04_vacuum_pol::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.electroweak {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/electroweak.csv"), meta) {
            m05_electroweak::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.decoherence {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/decoherence.csv"), meta) {
            m06_decoherence::write_csv(r, &mut w);
        }
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/born_rule.csv"), meta) {
            m06_decoherence::write_born_rule_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.neutrino {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/neutrino.csv"), meta) {
            m07_neutrino::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.pmns {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/pmns.csv"), meta) {
            m08_pmns::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.higgs {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/higgs.csv"), meta) {
            m09_higgs::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.lagrangian {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/lagrangian.csv"), meta) {
            m10_lagrangian::write_csv(r, &mut w);
        }
    }
}

/// Print all measurement summaries to stdout.
pub fn print_all_summaries(results: &MeasureResults) {
    if let Some(ref r) = results.traversal { m01_traversal::print_summary(r); }
    if let Some(ref r) = results.half_life { m02_halflife::print_summary(r); }
    if let Some(ref r) = results.modulo { m03_modulo::print_summary(r); }
    if let Some(ref r) = results.vacuum_pol { m04_vacuum_pol::print_summary(r); }
    if let Some(ref r) = results.electroweak { m05_electroweak::print_summary(r); }
    if let Some(ref r) = results.decoherence { m06_decoherence::print_summary(r); }
    if let Some(ref r) = results.neutrino { m07_neutrino::print_summary(r); }
    if let Some(ref r) = results.pmns { m08_pmns::print_summary(r); }
    if let Some(ref r) = results.higgs { m09_higgs::print_summary(r); }
    if let Some(ref r) = results.lagrangian { m10_lagrangian::print_summary(r); }
}
