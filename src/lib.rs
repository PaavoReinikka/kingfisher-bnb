pub mod bnb;
mod correction;

use crate::bnb::{SearchProblem, SearchState, OptimizationGoal, solvers::BestFirstSolver};
use bitvec::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use pyo3::prelude::*;

// --- Data Structure: Rule (for Python) ---
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    #[pyo3(get)]
    pub antecedent: Vec<usize>,
    #[pyo3(get)]
    pub consequent: usize,
    #[pyo3(get)]
    pub is_negative: bool,
    #[pyo3(get)]
    pub measure_value: f64,
    #[pyo3(get)]
    pub frequency_x: usize,
    #[pyo3(get)]
    pub frequency_xa: usize,
    #[pyo3(get)]
    pub frequency_a: usize,
}

#[derive(Debug, Clone)]
pub struct RuleEntry {
    pub rule: Rule,
    pub is_decreasing: bool,
}

impl Eq for RuleEntry {}
impl PartialEq for RuleEntry {
    fn eq(&self, other: &Self) -> bool {
        self.rule.measure_value == other.rule.measure_value &&
        self.rule.antecedent == other.rule.antecedent &&
        self.rule.consequent == other.rule.consequent &&
        self.rule.is_negative == other.rule.is_negative
    }
}

impl PartialOrd for RuleEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RuleEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        let measure_cmp = if self.is_decreasing {
            self.rule.measure_value.partial_cmp(&other.rule.measure_value).unwrap_or(Ordering::Equal)
        } else {
            other.rule.measure_value.partial_cmp(&self.rule.measure_value).unwrap_or(Ordering::Equal)
        };
        measure_cmp.then(self.rule.is_negative.cmp(&other.rule.is_negative))
            .then(self.rule.antecedent.cmp(&other.rule.antecedent))
            .then(self.rule.consequent.cmp(&other.rule.consequent))
    }
}

#[derive(Debug)]
pub struct RuleSet {
    pub max_k: usize,
    pub rules: BinaryHeap<RuleEntry>,
    pub is_decreasing: bool,
    pub initial_threshold: f64,
}

impl RuleSet {
    pub fn new(max_k: usize, _is_decreasing: bool, initial_threshold: f64) -> Self {
        // Since all measures (including negated ones) are to be minimized,
        // we always want the 'worst' (largest) value at the top of the heap.
        Self {
            max_k,
            rules: BinaryHeap::with_capacity(max_k + 1),
            is_decreasing: true,
            initial_threshold,
        }
    }

    pub fn add(&mut self, rule: Rule) {
        // All measures are transformed such that smaller is better.
        // Fisher's p -> ln(p) (max 0.0)
        // Others -> -Value (e.g., -Chi2)
        if rule.measure_value > self.initial_threshold {
            return;
        }

        let entry = RuleEntry { rule, is_decreasing: true };
        if self.rules.len() < self.max_k {
            self.rules.push(entry);
        } else if let Some(worst) = self.rules.peek() {
            if entry < *worst {
                self.rules.pop();
                self.rules.push(entry);
            }
        }
    }

    pub fn worst_value(&self) -> f64 {
        self.rules.peek().map(|e| e.rule.measure_value).unwrap_or(self.initial_threshold)
    }

    pub fn into_sorted_vec(self) -> Vec<Rule> {
        let mut v: Vec<_> = self.rules.into_iter().collect();
        v.sort();
        v.into_iter().map(|e| e.rule).collect()
    }
}

// --- Data Structure: BitMatrix ---
#[derive(Clone)]
pub struct BitMatrix {
    pub attributes: Vec<BitVec<u64, Lsb0>>,
    pub attr_freqs: Vec<usize>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl BitMatrix {
    pub fn load_from_file(path: &str, num_cols: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut attributes = vec![BitVec::repeat(false, 1000); num_cols];
        let mut row_count = 0;
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<usize> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            for &attr in &parts {
                if attr < num_cols {
                    if row_count >= attributes[attr].len() { attributes[attr].resize(row_count + 1000, false); }
                    attributes[attr].set(row_count, true);
                }
            }
            row_count += 1;
        }
        let mut attr_freqs = Vec::with_capacity(num_cols);
        for attr in 0..num_cols {
            attributes[attr].truncate(row_count);
            attr_freqs.push(attributes[attr].count_ones());
        }
        Ok(BitMatrix { attributes, attr_freqs, num_rows: row_count, num_cols })
    }

    pub fn from_rows(rows: Vec<Vec<usize>>, num_cols: usize) -> Self {
        let num_rows = rows.len();
        let mut attributes = vec![BitVec::repeat(false, num_rows); num_cols];
        for (row_idx, attrs) in rows.into_iter().enumerate() {
            for attr in attrs {
                if attr < num_cols { attributes[attr].set(row_idx, true); }
            }
        }
        let mut attr_freqs = Vec::with_capacity(num_cols);
        for i in 0..num_cols { attr_freqs.push(attributes[i].count_ones()); }
        BitMatrix { attributes, attr_freqs, num_rows, num_cols }
    }

    pub fn frequency(&self, path: &[usize]) -> usize {
        if path.is_empty() { return self.num_rows; }
        let mut res = self.attributes[path[0]].clone();
        for &attr in &path[1..] { res &= &self.attributes[attr]; }
        res.count_ones()
    }
}

// --- Logic: Statistical Measures ---
#[derive(Clone)]
pub struct Measures {
    pub ln_factorials: Vec<f64>,
}

impl Measures {
    pub fn new(n: usize) -> Self {
        let mut ln_factorials = Vec::with_capacity(n + 1);
        let mut sum = 0.0;
        ln_factorials.push(0.0);
        for i in 1..=n {
            sum += (i as f64).ln();
            ln_factorials.push(sum);
        }
        Measures { ln_factorials }
    }

    pub fn ln_combination(&self, n: usize, k: usize) -> f64 {
        if k > n { return f64::NEG_INFINITY; }
        self.ln_factorials[n] - self.ln_factorials[k] - self.ln_factorials[n - k]
    }

    pub fn ln_fishers_p(&self, fr_xa: usize, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        let log_denom = self.ln_combination(n, fr_x);
        let min_xa = fr_xa;
        let max_xa = fr_x.min(fr_a);
        if min_xa > max_xa { return 0.0; }
        let mut terms = Vec::new();
        let mut max_log_p = f64::NEG_INFINITY;
        for i in min_xa..=max_xa {
            let log_p = self.ln_combination(fr_a, i) + self.ln_combination(n - fr_a, fr_x - i) - log_denom;
            if log_p > max_log_p { max_log_p = log_p; }
            terms.push(log_p);
        }
        if max_log_p == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
        let sum_p: f64 = terms.iter().map(|&p| (p - max_log_p).exp()).sum();
        max_log_p + sum_p.ln()
    }

    pub fn chi_squared(&self, fr_xa: usize, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        let n = n as f64;
        let fr_xa = fr_xa as f64;
        let fr_x = fr_x as f64;
        let fr_a = fr_a as f64;
        if fr_x == 0.0 || fr_a == 0.0 || fr_x == n || fr_a == n { return 0.0; }
        let expected = (fr_x * fr_a) / n;
        let numer = (fr_xa - expected).abs() - 0.5;
        let numer = if numer < 0.0 { 0.0 } else { numer * numer };
        let var = expected * (1.0 - fr_x / n) * (1.0 - fr_a / n);
        if var == 0.0 { 0.0 } else { numer / var }
    }

    pub fn mutual_information(&self, fr_xa: usize, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        let n = n as f64;
        let f_xa = fr_xa as f64;
        let f_x = fr_x as f64;
        let f_a = fr_a as f64;
        let mut mi = 0.0;
        let log2 = |x: f64| x.log2();
        if f_xa > 0.0 { mi += (f_xa / n) * log2((n * f_xa) / (f_x * f_a)); }
        let f_x_not_a = f_x - f_xa;
        if f_x_not_a > 0.0 { mi += (f_x_not_a / n) * log2((n * f_x_not_a) / (f_x * (n - f_a))); }
        let f_not_x_a = f_a - f_xa;
        if f_not_x_a > 0.0 { mi += (f_not_x_a / n) * log2((n * f_not_x_a) / ((n - f_x) * f_a)); }
        let f_not_x_not_a = n - f_x - f_a + f_xa;
        if f_not_x_not_a > 0.0 { mi += (f_not_x_not_a / n) * log2((n * f_not_x_not_a) / ((n - f_x) * (n - f_a))); }
        mi
    }

    pub fn leverage(&self, fr_xa: usize, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        let n = n as f64;
        (fr_xa as f64 / n) - (fr_x as f64 * fr_a as f64 / (n * n))
    }

    pub fn bound(&self, measure_type: u8, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        let max_fr_xa = fr_x.min(fr_a);
        match measure_type {
            1 | 2 => { if fr_x <= fr_a { self.ln_combination(fr_a, fr_x) - self.ln_combination(n, fr_x) } else { self.ln_combination(fr_x, fr_a) - self.ln_combination(n, fr_a) } },
            3 => -self.chi_squared(max_fr_xa, fr_x, fr_a, n),
            4 => -self.mutual_information(max_fr_xa, fr_x, fr_a, n),
            5 => -self.leverage(max_fr_xa, fr_x, fr_a, n),
            _ => self.ln_fishers_p(max_fr_xa, fr_x, fr_a, n),
        }
    }
}

// --- Implementation: SearchState ---
#[derive(Clone, Debug)]
pub struct KingfisherState {
    pub path: Vec<usize>,
    pub freq: usize,
}

impl SearchState for KingfisherState {
    type Key = Vec<usize>;
    fn key(&self) -> Self::Key { self.path.clone() }
    fn depth(&self) -> usize { self.path.len() }
}

// --- Implementation: SearchProblem ---
pub struct KingfisherProblem {
    pub matrix: BitMatrix,
    pub measures: Measures,
    pub l_max: usize,
    pub min_fr: usize,
    pub min_cf: f64,
    pub t_type: u8,
    pub measure_type: u8,
    pub initial_threshold: f64,
    pub best_p_cache: DashMap<(Vec<usize>, usize, bool), f64>,
    pub ruleset: Arc<Mutex<RuleSet>>,
    pub required_consequents: Option<Vec<usize>>,
    pub excluded_consequents: Option<Vec<usize>>,
    pub excluded_attributes: Option<Vec<usize>>,
    pub constraints: Option<Vec<(usize, usize)>>,
    pub consequent_only: Option<Vec<usize>>,
}

impl KingfisherProblem {
    pub fn new(
        matrix: BitMatrix,
        measures: Measures,
        q: usize,
        l_max: usize,
        min_fr: usize,
        min_cf: f64,
        t_type: u8,
        measure_type: u8,
        initial_threshold: f64,
        required_consequents: Option<Vec<usize>>,
        excluded_consequents: Option<Vec<usize>>,
        excluded_attributes: Option<Vec<usize>>,
        constraints: Option<Vec<(usize, usize)>>,
        consequent_only: Option<Vec<usize>>,
    ) -> Self {
        let is_decreasing = measure_type == 1 || measure_type == 2;
        Self {
            matrix,
            measures,
            l_max,
            min_fr,
            min_cf,
            t_type,
            measure_type,
            initial_threshold,
            best_p_cache: DashMap::new(),
            ruleset: Arc::new(Mutex::new(RuleSet::new(q, is_decreasing, initial_threshold))),
            required_consequents,
            excluded_consequents,
            excluded_attributes,
            constraints,
            consequent_only,
        }
    }

    fn get_measure(&self, fr_xa: usize, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        match self.measure_type {
            1 | 2 => self.measures.ln_fishers_p(fr_xa, fr_x, fr_a, n),
            3 => -self.measures.chi_squared(fr_xa, fr_x, fr_a, n),
            4 => -self.measures.mutual_information(fr_xa, fr_x, fr_a, n),
            5 => -self.measures.leverage(fr_xa, fr_x, fr_a, n),
            _ => self.measures.ln_fishers_p(fr_xa, fr_x, fr_a, n),
        }
    }

    fn is_excluded(&self, attr: usize) -> bool {
        if let Some(ref excl) = self.excluded_attributes {
            if excl.contains(&attr) { return true; }
        }
        false
    }

    fn is_consequent_only(&self, attr: usize) -> bool {
        if let Some(ref co) = self.consequent_only {
            if co.contains(&attr) { return true; }
        }
        false
    }

    fn is_constrained(&self, antecedent: &[usize], consequent: usize) -> bool {
        if let Some(ref constr) = self.constraints {
            for &ant_attr in antecedent {
                // Check if pair (ant_attr, consequent) is in constraints
                if constr.iter().any(|&(a, b)| a == ant_attr && b == consequent) {
                    return true;
                }
            }
        }
        false
    }
}

impl SearchProblem<KingfisherState, f64> for KingfisherProblem {
    fn root_states(&self) -> Vec<KingfisherState> {
        (0..self.matrix.num_cols)
            .filter(|&i| !self.is_excluded(i))
            // We can still start from a "consequent_only" attribute,
            // but it cannot be part of an antecedent later.
            // Wait, if it's consequent_only, it can only be the ONE attribute being moved to consequent.
            // If path has length 1, it will be the consequent in evaluate.
            .map(|i| KingfisherState { path: vec![i], freq: self.matrix.attr_freqs[i] })
            .filter(|s| s.freq >= self.min_fr).collect()
    }

    fn branch(&self, state: &KingfisherState) -> Vec<KingfisherState> {
        if state.depth() >= self.l_max { return vec![]; }

        // If any attribute in the path is 'consequent_only', we can only branch if
        // that attribute is the only one (it will eventually become the consequent).
        // Actually, in our Best-First Search, the path [A, B, C] evaluates ALL
        // rules where {A, B, C} is the set of items.
        // If 'B' is consequent_only, it MUST be the consequent.
        // If BOTH 'B' and 'C' are consequent_only, then [A, B, C] can never form a valid rule
        // because one of them would have to be in the antecedent.
        let co_count = state.path.iter().filter(|&&attr| self.is_consequent_only(attr)).count();
        if co_count > 1 { return vec![]; }

        let last = *state.path.last().unwrap();
        (last + 1..self.matrix.num_cols)
            .filter(|&i| !self.is_excluded(i))
            .filter(|&i| {
                // If we already have a consequent_only attribute, we can't add another one.
                if co_count > 0 && self.is_consequent_only(i) { return false; }
                true
            })
            .map(|i| {
                let mut new_path = state.path.clone();
                new_path.push(i);
                let freq = self.matrix.frequency(&new_path);
                KingfisherState { path: new_path, freq }
            }).filter(|s| s.freq >= self.min_fr).collect()
    }

    fn evaluate(&self, state: &KingfisherState) -> Option<f64> {
        let n = self.matrix.num_rows;
        let mut best_improvement = f64::INFINITY;

        for &consequent in &state.path {
            // 1. Check if consequent is allowed
            if let Some(ref req) = self.required_consequents {
                if !req.contains(&consequent) { continue; }
            }
            if let Some(ref excl) = self.excluded_consequents {
                if excl.contains(&consequent) { continue; }
            }

            let mut antecedent = state.path.clone();
            antecedent.retain(|&x| x != consequent);
            if antecedent.is_empty() { continue; }

            // 2. Check if any attribute in antecedent is "consequent_only"
            if antecedent.iter().any(|&attr| self.is_consequent_only(attr)) {
                continue;
            }

            // 3. Check compatibility constraints
            if self.is_constrained(&antecedent, consequent) {
                continue;
            }

            let freq_x_ant = self.matrix.frequency(&antecedent);
            let freq_xa = state.freq;
            let freq_a = self.matrix.attr_freqs[consequent];

            // Positive Rules
            if self.t_type == 1 || self.t_type == 3 {
                if freq_xa as f64 / freq_x_ant as f64 >= self.min_cf {
                    let m = self.get_measure(freq_xa, freq_x_ant, freq_a, n);
                    let mut p_best = 0.0; // p=1.0 for Fisher, value=0.0 for others (after negation)
                    for i in 0..antecedent.len() {
                        let mut p_ant = antecedent.clone(); p_ant.remove(i);
                        let mut p_full = p_ant; p_full.push(consequent); p_full.sort();
                        if let Some(v) = self.best_p_cache.get(&(p_full, consequent, false)) { if *v < p_best { p_best = *v; } }
                    }
                    if m < p_best {
                        let mut full_path = state.path.clone(); full_path.sort();
                        self.best_p_cache.insert((full_path, consequent, false), m);
                        let rule = Rule { antecedent: antecedent.clone(), consequent, is_negative: false, measure_value: m, frequency_x: freq_x_ant, frequency_xa: freq_xa, frequency_a: freq_a };
                        self.ruleset.lock().unwrap().add(rule);
                        if m < best_improvement { best_improvement = m; }
                    }
                }
            }

            // Negative Rules
            if self.t_type == 2 || self.t_type == 3 {
                let freq_x_not_a = freq_x_ant - freq_xa;
                let freq_not_a = n - freq_a;
                if freq_x_not_a as f64 / freq_x_ant as f64 >= self.min_cf {
                    let m = self.get_measure(freq_x_not_a, freq_x_ant, freq_not_a, n);
                    let mut p_best = 0.0;
                    for i in 0..antecedent.len() {
                        let mut p_ant = antecedent.clone(); p_ant.remove(i);
                        let mut p_full = p_ant; p_full.push(consequent); p_full.sort();
                        if let Some(v) = self.best_p_cache.get(&(p_full, consequent, true)) { if *v < p_best { p_best = *v; } }
                    }
                    if m < p_best {
                        let mut full_path = state.path.clone(); full_path.sort();
                        self.best_p_cache.insert((full_path, consequent, true), m);
                        let rule = Rule { antecedent: antecedent.clone(), consequent, is_negative: true, measure_value: m, frequency_x: freq_x_ant, frequency_xa: freq_x_not_a, frequency_a: freq_not_a };
                        self.ruleset.lock().unwrap().add(rule);
                        if m < best_improvement { best_improvement = m; }
                    }
                }
            }
        }
        if best_improvement == f64::INFINITY { None } else { Some(best_improvement) }
    }

    fn bound(&self, state: &KingfisherState) -> f64 {
        let n = self.matrix.num_rows;
        let mut b_min = f64::INFINITY;
        for i in 0..self.matrix.num_cols {
            if !state.path.contains(&i) {
                if self.t_type == 1 || self.t_type == 3 {
                    let b = self.measures.bound(self.measure_type, state.freq, self.matrix.attr_freqs[i], n);
                    if b < b_min { b_min = b; }
                }
                if self.t_type == 2 || self.t_type == 3 {
                    let b = self.measures.bound(self.measure_type, state.freq, n - self.matrix.attr_freqs[i], n);
                    if b < b_min { b_min = b; }
                }
            }
        }
        b_min
    }

    fn prune_children(&self, _state: &KingfisherState, children: &mut Vec<KingfisherState>, threshold: f64) {
        children.retain(|c| !self.is_worse(self.bound(c), threshold));
    }

    fn goal(&self) -> OptimizationGoal { OptimizationGoal::Minimize }
}

#[pyfunction]
#[pyo3(signature = (data, k, q=100, l_max=3, t_type=1, m_threshold=0.05, min_fr=1, min_cf=0.0, measure_type=1, required_consequents=None, excluded_consequents=None, excluded_attributes=None, constraints=None, consequent_only=None))]
fn find_rules_from_data(
    data: Vec<Vec<usize>>,
    k: usize,
    q: usize,
    l_max: usize,
    t_type: u8,
    m_threshold: f64,
    min_fr: usize,
    min_cf: f64,
    measure_type: u8,
    required_consequents: Option<Vec<usize>>,
    excluded_consequents: Option<Vec<usize>>,
    excluded_attributes: Option<Vec<usize>>,
    constraints: Option<Vec<(usize, usize)>>,
    consequent_only: Option<Vec<usize>>,
) -> PyResult<Vec<Rule>> {
    let matrix = BitMatrix::from_rows(data, k + 1);
    let measures = Measures::new(matrix.num_rows);
    let transformed_threshold = if measure_type == 1 || measure_type == 2 { m_threshold.ln() } else { -m_threshold };
    let problem = KingfisherProblem::new(matrix, measures, q, l_max, min_fr, min_cf, t_type, measure_type, transformed_threshold, required_consequents, excluded_consequents, excluded_attributes, constraints, consequent_only);

    BestFirstSolver::search(&problem, q, transformed_threshold);

    Ok(Arc::try_unwrap(problem.ruleset).unwrap().into_inner().unwrap().into_sorted_vec())
}
#[pymodule]
fn kingfisher_bnb_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_rules_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(crate::correction::tarone, m)?)?;
    m.add_class::<Rule>()?;
    m.add_class::<crate::correction::TaroneResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measures_fishers_p() {
        let n = 10;
        let measures = Measures::new(n);
        let p = measures.ln_fishers_p(2, 2, 2, 10).exp();
        assert!((p - 1.0/45.0).abs() < 1e-10);
    }

    #[test]
    fn test_bitmatrix_frequency() {
        let mut matrix = BitMatrix {
            attributes: vec![BitVec::repeat(false, 4); 3],
            attr_freqs: vec![0; 3],
            num_rows: 4,
            num_cols: 3,
        };
        matrix.attributes[0].set(0, true); matrix.attributes[0].set(1, true); matrix.attributes[0].set(3, true);
        matrix.attributes[1].set(0, true); matrix.attributes[1].set(2, true); matrix.attributes[1].set(3, true);
        matrix.attributes[2].set(1, true); matrix.attributes[2].set(2, true); matrix.attributes[2].set(3, true);
        for i in 0..3 { matrix.attr_freqs[i] = matrix.attributes[i].count_ones(); }
        assert_eq!(matrix.frequency(&[0]), 3);
        assert_eq!(matrix.frequency(&[0, 1]), 2);
        assert_eq!(matrix.frequency(&[0, 1, 2]), 1);
    }
}
