use crate::bnb::{SearchProblem, SearchState, ObjectiveValue, ResultCollector, ResultNode};
use std::sync::Arc;

pub struct DfsSolver;

impl DfsSolver {
    pub fn search<S, V, P>(problem: &P, k: usize, initial_threshold: V) -> Vec<ResultNode<S, V>>
    where
        S: SearchState + 'static,
        V: ObjectiveValue + 'static,
        P: SearchProblem<S, V> + Sync + Send,
    {
        let collector = Arc::new(ResultCollector::new(k, problem.goal(), initial_threshold));

        for root in problem.root_states() {
            Self::dfs(problem, root, &collector);
        }

        Arc::try_unwrap(collector).ok().unwrap().into_sorted_vec()
    }

    fn dfs<S, V, P>(problem: &P, state: S, collector: &ResultCollector<S, V>)
    where
        S: SearchState,
        V: ObjectiveValue,
        P: SearchProblem<S, V>,
    {
        // Evaluation
        if let Some(val) = problem.evaluate(&state) {
            collector.add(state.clone(), val);
        }

        let threshold = collector.threshold();
        let mut children = problem.branch(&state);
        problem.prune_children(&state, &mut children, threshold);

        for child in children {
            let b = problem.bound(&child);
            if !problem.is_worse(b, threshold) {
                Self::dfs(problem, child, collector);
            }
        }
    }
}
