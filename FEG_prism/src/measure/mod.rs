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
pub mod m11_hausdorff;
pub mod m12_zigzag;
pub mod m13_collider;
pub mod m14_mass_formula;

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
    pub hausdorff: bool,
    pub zigzag: bool,
    pub collider: bool,
    pub mass_formula: bool,
    pub modulo_config: ModuloConfig,
    /// Run M6 (decoherence) only every N-th realization.  Default: 1 (every).
    pub decoherence_every: usize,
}

impl MeasureFlags {
    pub fn any_active(&self) -> bool {
        self.mass || self.halflife || self.modulo || self.vacuum
            || self.electroweak || self.decoherence || self.neutrino
            || self.pmns || self.higgs || self.lagrangian
            || self.hausdorff || self.zigzag || self.collider || self.mass_formula
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
    pub hausdorff: Option<m11_hausdorff::HausdorffResult>,
    pub zigzag: Option<m12_zigzag::ZigzagResult>,
    pub collider: Option<m13_collider::ColliderResult>,
    pub mass_formula: Option<m14_mass_formula::MassFormulaResult>,
}

/// Run all enabled measurements on a single realization.
///
/// Independent measurements (M1–M6, M9) run concurrently via
/// `std::thread::scope`.  M7→M8 remains sequential (M8 depends on M7).
pub fn run_all(flags: &MeasureFlags, ctx: &MeasureContext) -> MeasureResults {
    // Build reverse CSR once if M12 (zigzag) or M13 (collider) needs it
    let rev_csr = if flags.zigzag || flags.collider {
        Some(ctx.vacuum_csr.reverse())
    } else {
        None
    };

    // ── Independent measurements — run concurrently ──────────────────
    let (traversal, half_life, modulo, vacuum_pol, electroweak, decoherence, higgs,
         hausdorff, zigzag, collider, mass_formula) =
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
            let h11 = s.spawn(|| if flags.hausdorff {
                println!("  [M11] Hausdorff dimension (BFS volume growth)...");
                Some(m11_hausdorff::run(ctx))
            } else { None });
            let h12 = s.spawn(|| if flags.zigzag {
                println!("  [M12] Zigzag KK dimension...");
                Some(m12_zigzag::run(ctx, rev_csr.as_ref().unwrap()))
            } else { None });
            let h13 = s.spawn(|| if flags.collider {
                println!("  [M13] Topological collider...");
                Some(m13_collider::run(ctx, rev_csr.as_ref().unwrap()))
            } else { None });
            let h14 = s.spawn(|| if flags.mass_formula {
                println!("  [M14] Exact mass formula (genus verification)...");
                Some(m14_mass_formula::run(ctx))
            } else { None });
            (
                h1.join().unwrap(),
                h2.join().unwrap(),
                h3.join().unwrap(),
                h4.join().unwrap(),
                h5.join().unwrap(),
                h6.join().unwrap(),
                h7.join().unwrap(),
                h11.join().unwrap(),
                h12.join().unwrap(),
                h13.join().unwrap(),
                h14.join().unwrap(),
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
        hausdorff, zigzag, collider, mass_formula,
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
        hausdorff: collect_and_agg(results, |r| r.hausdorff.as_ref(), m11_hausdorff::aggregate),
        zigzag: collect_and_agg(results, |r| r.zigzag.as_ref(), m12_zigzag::aggregate),
        collider: collect_and_agg(results, |r| r.collider.as_ref(), m13_collider::aggregate),
        mass_formula: collect_and_agg(results, |r| r.mass_formula.as_ref(), m14_mass_formula::aggregate),
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
    if let Some(ref r) = results.hausdorff {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/hausdorff.csv"), meta) {
            m11_hausdorff::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.zigzag {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/zigzag.csv"), meta) {
            m12_zigzag::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.collider {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/collider.csv"), meta) {
            m13_collider::write_csv(r, &mut w);
        }
    }
    if let Some(ref r) = results.mass_formula {
        if let Ok(mut w) = CsvWriter::new(&format!("{dir}/mass_formula.csv"), meta) {
            m14_mass_formula::write_csv(r, &mut w);
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
    if let Some(ref r) = results.hausdorff { m11_hausdorff::print_summary(r); }
    if let Some(ref r) = results.zigzag { m12_zigzag::print_summary(r); }
    if let Some(ref r) = results.collider { m13_collider::print_summary(r); }
    if let Some(ref r) = results.mass_formula { m14_mass_formula::print_summary(r); }
}
