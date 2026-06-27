"""Type stubs for the compiled Rust extension (pyo3)."""
from typing import List, Optional, Tuple

class Rule:
    """A single mined dependency rule: antecedent => (NOT) consequent."""
    antecedent: List[int]
    consequent: int
    is_negative: bool
    measure_value: float
    """ln(p) for Fisher (measure_type 1/2); a negated statistic otherwise (smaller is always better)."""
    frequency_x: int
    frequency_xa: int
    frequency_a: int

def find_rules_from_data(
    data: List[List[int]],
    k: int,
    q: int = 100,
    l_max: int = 3,
    t_type: int = 1,
    m_threshold: float = 0.05,
    min_fr: int = 1,
    min_cf: float = 0.0,
    measure_type: int = 1,
    required_consequents: Optional[List[int]] = None,
    excluded_consequents: Optional[List[int]] = None,
    excluded_attributes: Optional[List[int]] = None,
    constraints: Optional[List[Tuple[int, int]]] = None,
    consequent_only: Optional[List[int]] = None,
) -> List[Rule]:
    """Mine the top-q non-redundant significant rules from sparse transactions.

    Args:
        data: transactions as lists of attribute indices (sparse rows).
        k: maximum attribute index (number of columns - 1).
        q: number of top rules to keep.
        l_max: maximum rule length (antecedent + consequent).
        t_type: rule direction -- 1=positive, 2=negative, 3=both.
        m_threshold: raw cutoff on the measure (p for Fisher; statistic otherwise).
        min_fr: minimum itemset frequency.
        min_cf: minimum confidence.
        measure_type: 1/2=Fisher's exact, 3=chi-squared, 4=mutual information, 5=leverage.
        required_consequents: if set, only these attributes may be consequents.
        excluded_consequents: attributes forbidden as consequents.
        excluded_attributes: attributes removed from the search space entirely.
        constraints: (antecedent_attr, consequent_attr) pairs that are forbidden.
        consequent_only: attributes that may only ever appear as the consequent.
    """
    ...
