//! Dual-clock timeline: causal ticks (τ) and presentation seconds (t).

/// The kind of pacing for a timeline segment.
#[derive(Debug, Clone)]
pub enum PaceKind {
    /// Compress many ticks into a short duration (fast-forward).
    Rush { ticks: u64, duration_secs: f64 },
    /// Stretch a few ticks over a long duration (slow motion).
    SlowMotion { ticks: u64, duration_secs: f64 },
    /// Freeze τ, advance only presentation time (for narration).
    Pause { duration_secs: f64 },
    /// Constant pace: N causal ticks per second of video.
    Normal { ticks_per_second: f64 },
}

/// A segment of the timeline with its pacing and starting clocks.
#[derive(Debug, Clone)]
pub struct TimeSegment {
    pub kind: PaceKind,
    pub causal_start: u64,
    pub presentation_start: f64,
}

impl TimeSegment {
    /// Presentation duration of this segment.
    pub fn duration_secs(&self) -> f64 {
        match &self.kind {
            PaceKind::Rush { duration_secs, .. }
            | PaceKind::SlowMotion { duration_secs, .. }
            | PaceKind::Pause { duration_secs } => *duration_secs,
            PaceKind::Normal { .. } => 0.0, // open-ended; closed by next segment
        }
    }

    /// Number of causal ticks in this segment.
    pub fn causal_ticks(&self) -> u64 {
        match &self.kind {
            PaceKind::Rush { ticks, .. } | PaceKind::SlowMotion { ticks, .. } => *ticks,
            PaceKind::Pause { .. } => 0,
            PaceKind::Normal { .. } => 0,
        }
    }
}

/// The dual-clock timeline.
///
/// Maintains an ordered list of segments and tracks running totals for
/// both clocks.
#[derive(Debug, Clone)]
pub struct Timeline {
    segments: Vec<TimeSegment>,
    current_causal: u64,
    current_presentation: f64,
    default_pace: f64, // ticks per second (for wait_ticks)
}

impl Timeline {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            current_causal: 0,
            current_presentation: 0.0,
            default_pace: 10.0,
        }
    }

    pub fn rush(&mut self, ticks: u64, duration_secs: f64) {
        self.segments.push(TimeSegment {
            kind: PaceKind::Rush { ticks, duration_secs },
            causal_start: self.current_causal,
            presentation_start: self.current_presentation,
        });
        self.current_causal += ticks;
        self.current_presentation += duration_secs;
    }

    pub fn slow_motion(&mut self, ticks: u64, duration_secs: f64) {
        self.segments.push(TimeSegment {
            kind: PaceKind::SlowMotion { ticks, duration_secs },
            causal_start: self.current_causal,
            presentation_start: self.current_presentation,
        });
        self.current_causal += ticks;
        self.current_presentation += duration_secs;
    }

    pub fn pause(&mut self, duration_secs: f64) {
        self.segments.push(TimeSegment {
            kind: PaceKind::Pause { duration_secs },
            causal_start: self.current_causal,
            presentation_start: self.current_presentation,
        });
        self.current_presentation += duration_secs;
    }

    pub fn set_pace(&mut self, ticks_per_second: f64) {
        self.default_pace = ticks_per_second;
    }

    /// Add a normal-pace segment that advances `ticks` causal steps at the
    /// currently configured pace.
    pub fn wait_ticks(&mut self, ticks: u64) {
        let duration = ticks as f64 / self.default_pace;
        self.segments.push(TimeSegment {
            kind: PaceKind::Normal { ticks_per_second: self.default_pace },
            causal_start: self.current_causal,
            presentation_start: self.current_presentation,
        });
        self.current_causal += ticks;
        self.current_presentation += duration;
    }

    pub fn total_duration(&self) -> f64 {
        self.current_presentation
    }

    pub fn total_causal_ticks(&self) -> u64 {
        self.current_causal
    }

    /// Map presentation time t → interpolated causal tick.
    pub fn presentation_to_causal(&self, t: f64) -> f64 {
        for seg in self.segments.iter().rev() {
            if t >= seg.presentation_start {
                let dt = t - seg.presentation_start;
                return match &seg.kind {
                    PaceKind::Rush { ticks, duration_secs }
                    | PaceKind::SlowMotion { ticks, duration_secs } => {
                        let progress = (dt / duration_secs).clamp(0.0, 1.0);
                        seg.causal_start as f64 + progress * *ticks as f64
                    }
                    PaceKind::Pause { .. } => seg.causal_start as f64,
                    PaceKind::Normal { ticks_per_second } => {
                        seg.causal_start as f64 + dt * ticks_per_second
                    }
                };
            }
        }
        0.0
    }

    /// Map presentation time t → fractional progress within the active
    /// segment (0.0 – 1.0).  Useful for easing animations.
    pub fn segment_progress(&self, t: f64) -> f64 {
        for seg in self.segments.iter().rev() {
            if t >= seg.presentation_start {
                let dt = t - seg.presentation_start;
                let dur = seg.duration_secs();
                return if dur > 0.0 { (dt / dur).clamp(0.0, 1.0) } else { 1.0 };
            }
        }
        0.0
    }

    pub fn segments(&self) -> &[TimeSegment] {
        &self.segments
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}
