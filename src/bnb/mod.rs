//! Generic Branch & Bound core (vendored from the `branch_and_bound` crate of
//! github.com/PaavoReinikka/BranchAndBound). Kingfisher uses the Best-First
//! solver; BFS/DFS are included for completeness. Keep in sync upstream if the
//! core engine changes.

use parking_lot::RwLock;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Defines whether we are maximizing or minimizing the objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationGoal {
    Maximize,
    Minimize,
}

/// A value that can be compared for optimization purposes.
pub trait ObjectiveValue: PartialOrd + Copy + Send + Sync {}
impl<T: PartialOrd + Copy + Send + Sync> ObjectiveValue for T {}

/// Represents a node in the search tree.
pub trait SearchState: Send + Sync + Clone {
    type Key: std::hash::Hash + Eq + Send + Sync + Clone;

    /// Unique identifier for this state (to avoid cycles or redundant searches).
    fn key(&self) -> Self::Key;

    /// Current depth in the search tree.
    fn depth(&self) -> usize;
}

/// A found solution with its evaluated value.
#[derive(Debug, Clone)]
pub struct ResultNode<S: SearchState, V: ObjectiveValue> {
    pub state: S,
    pub value: V,
}

impl<S: SearchState, V: ObjectiveValue> PartialEq for ResultNode<S, V> {
    fn eq(&self, other: &Self) -> bool {
        self.value.partial_cmp(&other.value) == Some(Ordering::Equal)
    }
}

impl<S: SearchState, V: ObjectiveValue> Eq for ResultNode<S, V> {}

/// Wrapper to store in BinaryHeap based on optimization goal.
struct HeapNode<S: SearchState, V: ObjectiveValue> {
    node: ResultNode<S, V>,
    goal: OptimizationGoal,
}

impl<S: SearchState, V: ObjectiveValue> PartialEq for HeapNode<S, V> {
    fn eq(&self, other: &Self) -> bool {
        self.node.value.partial_cmp(&other.node.value) == Some(Ordering::Equal)
    }
}
impl<S: SearchState, V: ObjectiveValue> Eq for HeapNode<S, V> {}

impl<S: SearchState, V: ObjectiveValue> PartialOrd for HeapNode<S, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.goal {
            // We want a Min-Heap of the BEST results, so the "worst" of the best is at the top.
            // If maximizing, "better" is larger value. Min-Heap puts smallest value at top.
            // So for Maximize, we use normal comparison (smaller value -> top).
            // If minimizing, "better" is smaller value. Min-Heap should put largest value at top.
            // So for Minimize, we flip comparison (larger value -> top).
            OptimizationGoal::Maximize => other.node.value.partial_cmp(&self.node.value),
            OptimizationGoal::Minimize => self.node.value.partial_cmp(&other.node.value),
        }
    }
}

impl<S: SearchState, V: ObjectiveValue> Ord for HeapNode<S, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Manages the top-K results and provides the pruning threshold.
pub struct ResultCollector<S: SearchState, V: ObjectiveValue> {
    results: RwLock<BinaryHeap<HeapNode<S, V>>>,
    k: usize,
    goal: OptimizationGoal,
    initial_threshold: V,
}

impl<S: SearchState, V: ObjectiveValue> ResultCollector<S, V> {
    pub fn new(k: usize, goal: OptimizationGoal, initial_threshold: V) -> Self {
        Self {
            results: RwLock::new(BinaryHeap::with_capacity(k)),
            k,
            goal,
            initial_threshold,
        }
    }

    pub fn add(&self, state: S, value: V) {
        // First check if the value is even better than the initial threshold
        let is_valid = match self.goal {
            OptimizationGoal::Maximize => value >= self.initial_threshold,
            OptimizationGoal::Minimize => value <= self.initial_threshold,
        };
        if !is_valid {
            return;
        }

        let mut heap = self.results.write();
        if heap.len() < self.k {
            heap.push(HeapNode { node: ResultNode { state, value }, goal: self.goal });
        } else {
            let worst = heap.peek().unwrap();
            let is_better = match self.goal {
                OptimizationGoal::Maximize => value > worst.node.value,
                OptimizationGoal::Minimize => value < worst.node.value,
            };
            if is_better {
                heap.pop();
                heap.push(HeapNode { node: ResultNode { state, value }, goal: self.goal });
            }
        }
    }

    /// Returns the current pruning threshold (the worst value in Top-K).
    pub fn threshold(&self) -> V {
        let heap = self.results.read();
        if heap.len() < self.k {
            self.initial_threshold
        } else {
            heap.peek().unwrap().node.value
        }
    }

    pub fn into_sorted_vec(self) -> Vec<ResultNode<S, V>> {
        let mut heap = self.results.into_inner();
        let mut res = Vec::new();
        while let Some(hn) = heap.pop() {
            res.push(hn.node);
        }
        // res is [worst, ..., best].
        // so we keep the reverse.
        res.reverse();
        res
    }
}

/// Defines the problem logic.
pub trait SearchProblem<S: SearchState, V: ObjectiveValue> {
    /// Initial states to start the search.
    fn root_states(&self) -> Vec<S>;

    /// Generate descendants of a state.
    fn branch(&self, state: &S) -> Vec<S>;

    /// Calculate the objective value of a state.
    /// Returns None if the state is not a valid solution.
    fn evaluate(&self, state: &S) -> Option<V>;

    /// Calculate the optimistic bound for descendants.
    fn bound(&self, state: &S) -> V;

    /// Optional: Perform pruning on children before they are even created or evaluated.
    fn prune_children(&self, _state: &S, _children: &mut Vec<S>, _threshold: V) {}

    /// Optimization goal (Minimize/Maximize).
    fn goal(&self) -> OptimizationGoal;

    /// Whether the current bound is worse than the target (pruning logic).
    fn is_worse(&self, bound: V, target: V) -> bool {
        match self.goal() {
            OptimizationGoal::Maximize => bound < target,
            OptimizationGoal::Minimize => bound > target,
        }
    }
}

pub mod solvers;
