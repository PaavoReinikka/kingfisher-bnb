use crate::bnb::{SearchProblem, SearchState, ObjectiveValue, ResultCollector, ResultNode};
use dashmap::DashMap;
use rayon::prelude::*;
use std::sync::Arc;

pub struct BfsSolver;

impl BfsSolver {
    pub fn search<S, V, P>(problem: &P, k: usize, initial_threshold: V) -> Vec<ResultNode<S, V>>
    where
        S: SearchState + 'static,
        V: ObjectiveValue + 'static,
        P: SearchProblem<S, V> + Sync + Send,
    {
        let collector = Arc::new(ResultCollector::new(k, problem.goal(), initial_threshold));
        let mut current_level = DashMap::new();

        // Initialize roots
        for root in problem.root_states() {
            if let Some(val) = problem.evaluate(&root) {
                collector.add(root.clone(), val);
            }
            current_level.insert(root.key(), root);
        }

        while !current_level.is_empty() {
            let next_level = DashMap::new();
            let threshold = collector.threshold();

            current_level.into_par_iter().for_each(|(_, state)| {
                let mut children = problem.branch(&state);
                problem.prune_children(&state, &mut children, threshold);

                for child in children {
                    let b = problem.bound(&child);
                    if problem.is_worse(b, threshold) {
                        return;
                    }

                    if let Some(val) = problem.evaluate(&child) {
                        collector.add(child.clone(), val);
                    }
                    next_level.insert(child.key(), child);
                }
            });

            current_level = next_level;
        }

        Arc::try_unwrap(collector).ok().unwrap().into_sorted_vec()
    }
}
