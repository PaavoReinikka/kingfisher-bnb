use crate::bnb::{SearchProblem, SearchState, ObjectiveValue, ResultCollector, ResultNode, OptimizationGoal};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;

pub struct BestFirstSolver;

struct Node<S: SearchState, V: ObjectiveValue> {
    state: S,
    bound: V,
    goal: OptimizationGoal,
}

impl<S: SearchState, V: ObjectiveValue> PartialEq for Node<S, V> {
    fn eq(&self, other: &Self) -> bool {
        self.bound.partial_cmp(&other.bound) == Some(Ordering::Equal)
    }
}
impl<S: SearchState, V: ObjectiveValue> Eq for Node<S, V> {}

impl<S: SearchState, V: ObjectiveValue> PartialOrd for Node<S, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.goal {
            // In Best-First, we want the "best" bound to be at the top of the MaxHeap.
            // If maximizing, "best" bound is the largest value.
            // If minimizing, "best" bound is the smallest value (so we flip comparison).
            OptimizationGoal::Maximize => self.bound.partial_cmp(&other.bound),
            OptimizationGoal::Minimize => other.bound.partial_cmp(&self.bound),
        }
    }
}

impl<S: SearchState, V: ObjectiveValue> Ord for Node<S, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl BestFirstSolver {
    pub fn search<S, V, P>(problem: &P, k: usize, initial_threshold: V) -> Vec<ResultNode<S, V>>
    where
        S: SearchState + 'static,
        V: ObjectiveValue + 'static,
        P: SearchProblem<S, V> + Sync + Send,
    {
        let collector = Arc::new(ResultCollector::new(k, problem.goal(), initial_threshold));
        let mut queue = BinaryHeap::new();
        let goal = problem.goal();

        for root in problem.root_states() {
            let bound = problem.bound(&root);
            queue.push(Node { state: root, bound, goal });
        }

        while let Some(current) = queue.pop() {
            let threshold = collector.threshold();
            if problem.is_worse(current.bound, threshold) {
                // Since it's a priority queue, if this bound is worse than threshold,
                // all subsequent bounds in the queue will also be worse.
                break;
            }

            // Evaluation
            if let Some(val) = problem.evaluate(&current.state) {
                collector.add(current.state.clone(), val);
            }

            // Branching
            let mut children = problem.branch(&current.state);
            problem.prune_children(&current.state, &mut children, threshold);

            for child in children {
                let b = problem.bound(&child);
                if !problem.is_worse(b, threshold) {
                    queue.push(Node { state: child, bound: b, goal });
                }
            }
        }

        Arc::try_unwrap(collector).ok().unwrap().into_sorted_vec()
    }
}
