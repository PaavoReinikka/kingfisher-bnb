# Releasing

Wheels (Linux x86_64/aarch64, Windows x64, macOS arm64) and an sdist are built by
the `CI` GitHub Actions workflow. Publishing uses **Trusted Publishing (OIDC)** —
no API tokens are stored anywhere.

## One-time setup (per index)

Configure a Trusted Publisher on each index *before* the first publish. You can
add it as a **pending publisher** even though the project does not exist yet.

For **TestPyPI** (https://test.pypi.org/manage/account/publishing/) and again for
**PyPI** (https://pypi.org/manage/account/publishing/), add:

| Field | Value |
| --- | --- |
| PyPI Project Name | `kingfisher-bnb` |
| Owner | `PaavoReinikka` |
| Repository name | `kingfisher-bnb` |
| Workflow name | `CI.yml` |
| Environment | *(leave blank)* |

## Dry run: publish to TestPyPI

1. GitHub → **Actions → CI → Run workflow**, set **publish_testpypi = true**, run.
2. It builds every wheel + sdist and uploads them to TestPyPI.
3. Verify a clean install (wheels live on TestPyPI, but `numpy` is pulled from real PyPI):

   ```bash
   uv venv /tmp/kf && uv pip install --python /tmp/kf \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     kingfisher-bnb
   /tmp/kf/bin/python -c "import kingfisher_bnb as kf; print(kf.find_rules_from_data([[0,1]], k=1, q=5))"
   ```

## Real release: publish to PyPI

Bump `version` in `Cargo.toml` and `pyproject.toml`, commit, then tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `release` job builds all artifacts and publishes to PyPI. After it lands:

```bash
pip install kingfisher-bnb
```
