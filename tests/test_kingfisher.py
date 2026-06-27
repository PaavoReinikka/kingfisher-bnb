"""Python smoke tests for the built kingfisher_bnb extension.

Run after `maturin develop`:  pytest -q
"""
import kingfisher_bnb as kf


def test_dense_to_sparse_roundtrip():
    dense = [[1, 0, 1], [0, 1, 1]]
    sparse, id_to_name, name_to_id = kf.dense_to_sparse(dense, ["A", "B", "C"])
    assert sparse == [[0, 2], [1, 2]]
    assert id_to_name[0] == "A"
    assert name_to_id["C"] == 2
    assert kf.sparse_to_dense(sparse, 2) == dense


def test_find_rules_basic():
    # A and B co-occur perfectly; C is separate.
    dense = [
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]
    sparse, _id_to_name, _ = kf.dense_to_sparse(dense, ["A", "B", "C"])
    rules = kf.find_rules_from_data(
        data=sparse, k=2, q=10, l_max=2, t_type=1, m_threshold=1.0
    )
    assert len(rules) > 0
    for r in rules:
        assert isinstance(r, kf.Rule)
        assert r.measure_value <= 0.0  # ln(p), p in (0, 1]

    # The perfect A<->B association must surface as a positive rule over {0, 1}.
    ab = [
        r
        for r in rules
        if set(r.antecedent) | {r.consequent} == {0, 1} and not r.is_negative
    ]
    assert ab, "expected an A=>B (or B=>A) positive rule"
