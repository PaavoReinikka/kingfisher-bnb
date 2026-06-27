import math

import kingfisher_bnb as kingfisher

print("=== Kingfisher (BnB) Python Example ===")

dense_data = [
    [1, 1, 0, 0],  # Apple, Banana
    [1, 1, 1, 0],  # Apple, Banana, Cherry
    [0, 1, 1, 0],  # Banana, Cherry
    [1, 0, 0, 1],  # Apple, Date
    [1, 1, 0, 1],  # Apple, Banana, Date
    [0, 0, 0, 1],  # Date
    [0, 1, 0, 1],  # Banana, Date
]
column_names = ["Apple", "Banana", "Cherry", "Date"]

print("Transforming dense data to sparse...")
sparse_data, id_to_name, name_to_id = kingfisher.dense_to_sparse(dense_data, column_names)
print(f"Sparse data: {sparse_data}")

print("\nRunning Kingfisher search (Best-First)...")
k_max = len(column_names) - 1
rules = kingfisher.find_rules_from_data(
    data=sparse_data,
    k=k_max,
    q=5,
    l_max=3,
    t_type=3,         # Both positive and negative
    m_threshold=1.0,  # 1.0 => return all top-q (Fisher)
)

print(f"\nFound {len(rules)} rules:")
for i, rule in enumerate(rules):
    ant_names = [id_to_name[idx] for idx in rule.antecedent]
    ant_str = " AND ".join(ant_names)
    cons_name = id_to_name[rule.consequent]
    sign = "" if not rule.is_negative else "NOT "
    p_val = math.exp(rule.measure_value)
    print(f"Rule {i + 1:>2}: IF {ant_str:<20} THEN {sign}{cons_name:<10} (p={p_val})")
