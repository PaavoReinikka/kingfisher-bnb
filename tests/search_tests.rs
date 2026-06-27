use kingfisher_bnb_extension::{KingfisherProblem, BitMatrix, Measures};
use kingfisher_bnb_extension::bnb::solvers::BestFirstSolver;
use bitvec::prelude::*;

#[test]
fn test_kingfisher_min_fr_cf() {
    // 10 rows, 3 columns
    let mut matrix = BitMatrix {
        attributes: vec![BitVec::repeat(false, 10); 3],
        attr_freqs: vec![0; 3],
        num_rows: 10,
        num_cols: 3,
    };

    // Attribute 0: Rows 0-4 (fr=5)
    for i in 0..5 { matrix.attributes[0].set(i, true); }
    // Attribute 1: Rows 0-2 (fr=3)
    for i in 0..3 { matrix.attributes[1].set(i, true); }
    // Attribute 2: Row 0 (fr=1)
    matrix.attributes[2].set(0, true);

    for i in 0..3 { matrix.attr_freqs[i] = matrix.attributes[i].count_ones(); }

    let measures = Measures::new(10);

    // Case 1: min_fr=4. Only attr 0 should pass root check.
    let problem1 = KingfisherProblem::new(
        matrix.clone(),
        measures.clone(),
        10, // q
        3,
        4,
        0.0,
        1,
        1,
        0.0, // initial_threshold (ln(1.0))
        None,
        None,
        None,
        None,
        None,
    );
    let res1 = BestFirstSolver::search(&problem1, 10, 0.0);
    assert_eq!(res1.len(), 0);

    // Case 2: high min_cf (0.9).
    // Rule {1} -> 0 has fr_xa=3, fr_x=3, conf=1.0. Should pass.
    let problem2 = KingfisherProblem::new(
        matrix,
        measures,
        10, // q
        3,
        1,
        0.9,
        1,
        1,
        0.0, // initial_threshold (ln(1.0))
        None,
        None,
        None,
        None,
        None,
    );
    let res2 = BestFirstSolver::search(&problem2, 10, 0.0);
    assert!(res2.iter().any(|r| r.state.path.contains(&0) && r.state.path.contains(&1)));
}

#[test]
fn test_kingfisher_pruning_consistency() {
    let mut matrix = BitMatrix {
        attributes: vec![BitVec::repeat(false, 30); 4],
        attr_freqs: vec![0; 4],
        num_rows: 30,
        num_cols: 4,
    };
    // A0, A1 always together (Rows 0-9) - fr=10
    for i in 0..10 {
        matrix.attributes[0].set(i, true);
        matrix.attributes[1].set(i, true);
    }
    // A2, A3 always together (Rows 15-22) - fr=8
    for i in 15..23 {
        matrix.attributes[2].set(i, true);
        matrix.attributes[3].set(i, true);
    }
    for i in 0..4 { matrix.attr_freqs[i] = matrix.attributes[i].count_ones(); }
    let measures = Measures::new(30);

    let problem = KingfisherProblem::new(
        matrix.clone(),
        measures.clone(),
        50, // q
        3,
        1,
        0.0,
        1,
        1,
        0.0, // initial_threshold (ln(1.0))
        None,
        None,
        None,
        None,
        None,
    );
    // ln(1.0) is 0.0, everything is valid
    let res_no_prune = BestFirstSolver::search(&problem, 50, 0.0);

    // Tight threshold ln(0.05)
    let threshold_prune = 0.05f64.ln();
    let problem_prune = KingfisherProblem::new(
        matrix.clone(),
        measures.clone(),
        50, // q
        3,
        1,
        0.0,
        1,
        1,
        threshold_prune,
        None,
        None,
        None,
        None,
        None,
    );
    let res_prune = BestFirstSolver::search(&problem_prune, 50, threshold_prune);

    assert!(res_prune.len() > 0);
    for r in &res_prune {
        assert!(res_no_prune.iter().any(|np| np.state.path == r.state.path && (np.value - r.value).abs() < 1e-10));
    }
}
