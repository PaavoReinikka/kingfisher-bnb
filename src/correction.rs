//! Tarone (1990) multiple-testing correction — the *effective* number of tests.
//!
//! Multiple-testing correction is defined on p-values, so this is **Fisher-only**.
//! For a 2x2 Fisher test with margins `(s_a, s_b, n)` the *minimum attainable*
//! p-value depends only on the margins — which is exactly `Measures::bound` for
//! the Fisher measure. A hypothesis (item pair) is "testable" at a corrected
//! level only if that minimum can reach it; Tarone counts only the testable ones
//! to get a valid effective number of tests `m_eff` that is usually far smaller
//! than `C(n_items, 2)`, and hence far more powerful than plain Bonferroni.
//!
//! Pairwise (single-item antecedent, single consequent) hypotheses only.

use crate::{BitMatrix, Measures};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Result of a Tarone analysis over the pairwise Fisher hypothesis space.
#[pyclass]
pub struct TaroneResult {
    /// Number of items (columns).
    #[pyo3(get)]
    pub n_items: usize,
    /// Number of transactions (rows).
    #[pyo3(get)]
    pub n_transactions: usize,
    /// Total hypotheses considered = C(n_items, 2).
    #[pyo3(get)]
    pub n_hypotheses: usize,
    /// The alpha passed in (None if not given).
    #[pyo3(get)]
    pub alpha: Option<f64>,
    /// Tarone effective number of tests at `alpha` (None if alpha not given).
    #[pyo3(get)]
    pub m_eff: Option<usize>,
    /// Corrected raw-p rejection cutoff = `alpha / m_eff` (None if alpha not given).
    #[pyo3(get)]
    pub threshold: Option<f64>,
    /// (min_log_p, pair_count), ascending by min_log_p. Internal; the `spectrum`
    /// getter exposes it as (min_p, count).
    spectrum_log: Vec<(f64, usize)>,
}

/// Count pairs whose minimal attainable p is <= exp(log_thresh). `spectrum_log`
/// is ascending by min_log_p, so we can stop early.
fn count_le(spectrum_log: &[(f64, usize)], log_thresh: f64) -> usize {
    let mut cum = 0;
    for &(lp, c) in spectrum_log {
        if lp <= log_thresh + 1e-12 {
            cum += c;
        } else {
            break;
        }
    }
    cum
}

/// Tarone's k*: the smallest k in [1, m_total] with #{min_p <= alpha/k} <= k.
/// The predicate is monotone in k (m(k) is non-increasing, k increasing), so we
/// binary-search. Returns 1 when there are no hypotheses (no correction).
fn tarone_k(spectrum_log: &[(f64, usize)], m_total: usize, alpha: f64) -> usize {
    if m_total == 0 {
        return 1;
    }
    let log_alpha = alpha.ln();
    let pred = |k: usize| count_le(spectrum_log, log_alpha - (k as f64).ln()) <= k;
    let (mut lo, mut hi) = (1usize, m_total);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Build the minimal-attainable-p spectrum from item supports. Items are bucketed
/// by support (min_p depends only on the two supports + n), so this is
/// O(distinct_supports^2), not O(n_items^2). Returns (ascending (min_log_p,
/// count) buckets, total pair count = C(n_items, 2)).
fn build_spectrum(supports: &[usize], n: usize, measures: &Measures) -> (Vec<(f64, usize)>, usize) {
    let mut by_support: BTreeMap<usize, usize> = BTreeMap::new();
    for &s in supports {
        *by_support.entry(s).or_insert(0) += 1;
    }
    let svals: Vec<(usize, usize)> = by_support.into_iter().collect();
    let mut entries: Vec<(f64, usize)> = Vec::new();
    let mut total = 0usize;
    for i in 0..svals.len() {
        let (s1, c1) = svals[i];
        let same = c1 * c1.saturating_sub(1) / 2; // pairs of items sharing this support
        if same > 0 {
            entries.push((measures.bound(1, s1, s1, n), same));
            total += same;
        }
        for j in (i + 1)..svals.len() {
            let (s2, c2) = svals[j];
            let cnt = c1 * c2;
            entries.push((measures.bound(1, s1, s2, n), cnt));
            total += cnt;
        }
    }
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let mut merged: Vec<(f64, usize)> = Vec::new();
    for (lp, c) in entries {
        match merged.last_mut() {
            Some(last) if (last.0 - lp).abs() < 1e-12 => last.1 += c,
            _ => merged.push((lp, c)),
        }
    }
    (merged, total)
}

#[pymethods]
impl TaroneResult {
    /// The minimal-p histogram as (min_p, pair_count), ascending by min_p.
    #[getter]
    fn spectrum(&self) -> Vec<(f64, usize)> {
        self.spectrum_log.iter().map(|&(lp, c)| (lp.exp(), c)).collect()
    }

    /// Tarone effective number of tests for `alpha` — reuses the precomputed
    /// spectrum, so any alpha is free (no recomputation over the data).
    fn m_eff_at(&self, alpha: f64) -> usize {
        tarone_k(&self.spectrum_log, self.n_hypotheses, alpha)
    }

    /// Corrected per-hypothesis rejection cutoff `alpha / m_eff` for `alpha`.
    fn threshold_at(&self, alpha: f64) -> f64 {
        alpha / self.m_eff_at(alpha) as f64
    }

    fn __repr__(&self) -> String {
        format!(
            "TaroneResult(n_items={}, n_hypotheses={}, alpha={:?}, m_eff={:?}, threshold={:?})",
            self.n_items, self.n_hypotheses, self.alpha, self.m_eff, self.threshold
        )
    }
}

/// Compute Tarone's effective number of tests over the pairwise Fisher hypothesis
/// space of a transactional dataset.
///
/// `data` / `k` are the same as `find_rules_from_data` (sparse rows; k = max
/// attribute index). If `alpha` is given, `m_eff` and `threshold` (= alpha/m_eff)
/// are filled in; either way `spectrum` and `m_eff_at(alpha)` are available.
#[pyfunction]
#[pyo3(signature = (data, k, alpha=None))]
pub fn tarone(data: Vec<Vec<usize>>, k: usize, alpha: Option<f64>) -> PyResult<TaroneResult> {
    let matrix = BitMatrix::from_rows(data, k + 1);
    let n = matrix.num_rows;
    let measures = Measures::new(n);
    let (spectrum_log, n_hypotheses) = build_spectrum(&matrix.attr_freqs, n, &measures);
    let (m_eff, threshold) = match alpha {
        Some(a) => {
            let ke = tarone_k(&spectrum_log, n_hypotheses, a);
            (Some(ke), Some(a / ke as f64))
        }
        None => (None, None),
    };
    Ok(TaroneResult {
        n_items: matrix.num_cols,
        n_transactions: n,
        n_hypotheses,
        alpha,
        m_eff,
        threshold,
        spectrum_log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tarone_all_testable() {
        // 3 items, each support 5 of n=10: all 3 pairs are testable at 0.05
        // (min_p = 1/252 << 0.05), so Tarone gives the full Bonferroni count.
        let n = 10;
        let m = Measures::new(n);
        let (spec, total) = build_spectrum(&[5, 5, 5], n, &m);
        assert_eq!(total, 3);
        assert_eq!(tarone_k(&spec, total, 0.05), 3);
    }

    #[test]
    fn test_tarone_excludes_untestable() {
        // supports [1,5,5]: the two (1,5) pairs have min_p=0.5 (untestable at
        // 0.05); only the (5,5) pair is testable -> m_eff = 1 (no penalty).
        let n = 10;
        let m = Measures::new(n);
        let (spec, total) = build_spectrum(&[1, 5, 5], n, &m);
        assert_eq!(total, 3);
        assert_eq!(tarone_k(&spec, total, 0.05), 1);
    }

    #[test]
    fn test_tarone_k_monotone_threshold() {
        // m_eff is non-decreasing as alpha shrinks is NOT guaranteed, but the
        // corrected cutoff alpha/m_eff must never exceed alpha.
        let n = 50;
        let m = Measures::new(n);
        let (spec, total) = build_spectrum(&[25, 25, 25, 25, 25], n, &m);
        for &a in &[0.05_f64, 0.01, 0.001] {
            let ke = tarone_k(&spec, total, a);
            assert!(ke >= 1 && ke <= total);
            assert!(a / ke as f64 <= a + 1e-12);
        }
    }
}
