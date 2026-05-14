# Initial diagnosis: reproducibility issues in `dimension_influence`

## 1. No experiment metadata stored with results (critical)

`saveperform_ds` saves two files per run — `*_dsresults.pkl` and `*_problems.pkl` — but stores **zero metadata**: no date, no seed, no `N`, no `mdims`/`codims`, no algorithm parameters. The only trace is the filename prefix `exp{N}_maniftype{M}_obj{O}_`. If you rename, move, or reuse the same experiment number, you have no way to know what parameters produced those results.

Compare with `synchronization_rotations`, where every experiment directory contains `metadata.json`, `config.json`, and `notes.md`.

---

## 2. Seeding is a notebook-global `np.random.seed(1)`, not stored anywhere

`np.random.seed(1)` is called once in a notebook cell. It is:
- **Not passed to** `perform_ds` or `saveperform_ds` — no `seed` parameter exists
- **Not saved** in the output files
- **Order-dependent**: re-running any upstream cell changes the random state for all downstream cells

This means results are reproducible only if you re-run the notebook **from the top**, **in order**, **with no interruptions** — and even then only if no numpy version changes occurred.

---

## 3. Bug: `directsearch` return format doesn't match what `perform_ds` expects

When `returnforeuclsimplex=1`, `directsearch` returns:
```python
return [ { ..., "euclideansimplex": 1, "rsimplex_run": {...} } ]  # list of ONE dict
```
But the docstring says "list of two dicts", and `perform_ds` indexes it as:
```python
dsresults[...][key1] = dsresult[1]   # IndexError — list has only 1 element
dsresults[...][key0] = dsresult[0]   # this one works
```
This means **running with `euclideansimplex=1` crashes**. The riemannian sub-run result is embedded as `dsresult[0]["rsimplex_run"]` instead of being a second list element.

---

## 4. Hardcoded relative paths, no CLI runner

Both `saveperform_ds`, `plotting_dp`, and `performancetable` hardcode `"dsresults_folder/"` and `"tables_and_plots/"` as relative paths. These only work if you run from inside `dimension_influence/`. There is no `--exppath` argument like `synchronization_rotations/run.py` has. Combined with the notebook dependency, you can't run experiments headlessly or on a remote machine without a browser.

---

## 5. `dsresults_folder` doesn't exist in the repo

There is no `dsresults_folder/` directory tracked. Since `.pkl` files are gitignored, there is **no record anywhere in the repo of which experiments have been run**. In `synchronization_rotations`, the `experiments/` directory is tracked (only the non-`.pkl` files: `config.json`, `metadata.json`, `notes.md`), so the git history shows what was run and when.

---

## 6. Naming inconsistencies across the codebase

The same concept goes by different names in different places:

| Location | Name |
|---|---|
| `directsearch()` parameter | `returnforeuclsimplex` |
| `perform_ds()` parameter | `euclideansimplex` |
| `ds_key` field | `euclideansimplex` |
| Return dict key | `"euclideansimplex"` |
| Nested return dict key | `"euclidean_simplex"` (underscore!) |
| `synchronization_rotations` | `euclsimplex` |

Also: `"sucess_indices"` (typo, missing 'c') appears in the return dict.

---

## 7. Minor issues

- **`print("hey")`** leftover debug statement in `plotandsave_dataprofile.py:241`
- **`stopping_criterion` potentially uninitialized**: it's set inside the while loop but accessed after it; if `budget ≤ 1` the loop never executes and it's undefined
- **`imdim = 1` and `icodim = 2`** are hardcoded magic indices inside `dataprofile()` with a comment saying "to change which mdim is fixed, change this line" — a parameter would be cleaner
