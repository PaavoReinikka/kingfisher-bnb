# kingfisher-bnb

Fast **Kingfisher** rule mining — the top-K *non-redundant, statistically
significant* dependency rules of a binary/transactional dataset — implemented as
a parallel **Best-First Branch & Bound** search in Rust with a small Python API.

Kingfisher finds rules `A ⇒ B` (and negative rules `A ⇒ ¬B`) ranked by a
statistical measure, **without a minimum-support threshold**. False discovery is
controlled by algorithmic design — top-K + a non-redundancy rule (a specific rule
must beat all its generalizations) + a raw significance cutoff — *not* by
family-wise/FDR correction, so the Fisher scores it returns are **raw,
uncorrected p-values**.

## Install

```bash
pip install kingfisher-bnb        # prebuilt wheels (Linux/macOS/Windows)
```

To build from source you need a Rust toolchain (>= 1.85) and
[maturin](https://www.maturin.rs/):

```bash
maturin develop --release         # builds + installs into the active venv
```

## Python usage

```python
import math
import kingfisher_bnb as kf

# Dense rows -> sparse item lists (and an id<->name mapping)
dense = [[1, 1, 0],
         [1, 0, 1],
         [0, 1, 1]]
sparse, id_to_name, _ = kf.dense_to_sparse(dense, ["Apple", "Banana", "Cherry"])

rules = kf.find_rules_from_data(
    data=sparse,
    k=2,            # max attribute index (n_columns - 1)
    q=10,           # top-K rules to keep
    l_max=3,        # max rule length (antecedent + consequent)
    t_type=3,       # 1=positive, 2=negative, 3=both
    m_threshold=1.0,  # raw cutoff (p<=1 => keep all top-q for Fisher)
    measure_type=1,   # 1/2=Fisher's exact, 3=chi^2, 4=mutual info, 5=leverage
)

for r in rules:
    ant = " AND ".join(id_to_name[i] for i in r.antecedent)
    cons = id_to_name[r.consequent]
    sign = "NOT " if r.is_negative else ""
    p = math.exp(r.measure_value)      # Fisher: measure_value = ln(p)
    print(f"IF {ant} THEN {sign}{cons}  (p={p:.4g})")
```

Each `Rule` exposes: `antecedent` (list[int]), `consequent` (int),
`is_negative` (bool), `measure_value` (float — `ln(p)` for Fisher, a negated
statistic otherwise so that smaller is always better), and the contingency
counts `frequency_x`, `frequency_xa`, `frequency_a`.

## CLI

```bash
kingfisher --data data/test_data.txt --cols 4 --t-type 3 --measure-type 1
```

Data format: one transaction per line, space-separated attribute indices.

## Measures

| `measure_type` | Measure | Returns |
|---|---|---|
| 1, 2 | Fisher's exact test | `ln(p)` (raw p-value) |
| 3 | Chi-squared | test statistic |
| 4 | Mutual information | bits |
| 5 | Leverage | effect size |

`t_type` is a separate axis (rule direction): `1` positive, `2` negative, `3` both.

## Multiple-testing correction (Tarone)

`find_rules_from_data` returns **raw** p-values. To control the family-wise error
rate over the many pairwise Fisher tests, `tarone()` computes the *effective*
number of tests via Tarone's method — counting only **testable** hypotheses
(pairs whose minimum attainable p, given the margins, can reach significance). It
reuses the same bounds the search uses, so it is cheap, and far more powerful than
plain Bonferroni over `C(n_items, 2)`.

```python
res = kf.tarone(sparse, k=2, alpha=0.05)
res.m_eff           # effective number of tests (<= C(n_items, 2))
res.threshold       # corrected raw-p cutoff = alpha / m_eff
res.m_eff_at(0.01)  # any other alpha, no recomputation over the data
res.spectrum        # [(min_p, count), ...] minimal-p histogram

# a pair is significant when its raw Fisher p <= res.threshold
```

Correction is defined on p-values, so it applies to the **Fisher** measure only
(`measure_type` 1/2); chi²/MI/leverage are effect sizes with no p-value to correct.
Pairwise (single-item) hypotheses.

## Attribution

This is an independent Rust implementation of the Kingfisher algorithm from
Wilhelmiina Hämäläinen's published work — written from the papers, not ported
from the original C source. If you use it in research, please cite:

> Hämäläinen, W. *Efficient discovery of the top-K optimal dependency rules with
> Fisher's exact test of significance.* ICDM 2010.

The `tarone()` correction follows:

> Tarone, R. E. *A modified Bonferroni method for discrete data.* Biometrics
> 46(2):515–522, 1990. PMID: 2364136.

For its application to pattern mining, see also Terada et al., *Statistical
significance of combinatorial regulations* (PNAS 2013, "LAMP"), and Hämäläinen &
Webb, *A tutorial on statistically sound pattern discovery* (DMKD 2019).

## License

MIT — see [LICENSE](LICENSE).
