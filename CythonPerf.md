# Cython vs. Python Performance

The vendored `featureHandler` package was intentionally written in
straight Python to make the extraction as simple and dependency‑free as
possible.  As a side effect we can now see where the original qlib code
gets its speed.  Running the recalculation scripts produces these
observations (benchmarks from `benchm.txt`):

* **Alpha360**
  * original qlib: ~140 s to load/process 3 239×361 frame
  * vendored Python: ~2 s (0.15 s on the trimmed AAPL sample)
  * speed‑up factor > 70×
  * reason: our `ProcessInf`/`ZScoreNorm`/etc. are simple loops that act
    on the entire dataframe once.  qlib’s versions group by `datetime`
    and apply functions row‑by‑row, then sort the index – an expensive
    pandas/Cython pipeline.  Removing the group‑by reduces the cost to a
    fraction of a second.

* **Alpha158**
  * original qlib: ~2.35 s for the full universe
  * vendored Python: 8–10 s (8.7 s in the 14‑day AAPL test)
  * slowdown factor ~4×
  * reason: our loader evaluates every factor expression using Python’s
    `eval()` and pandas `rolling.apply()`; the original qlib provider
    implements many of the rolling operators in Cython and vectorises the
    computation, so it executes much faster.

The timing logs confirm this: the long step for Alpha360 in qlib is
`ProcessInf` (137 s), whereas for Alpha158 the costly work is inside the
`Loading data` phase (8–9 s) when expressions are evaluated.  Replacing the
provider with qlib’s original one restores the fast behaviour, which is a
clear sign that the difference lies in the expression engine rather than
the rest of the handler.

## Migration plan: adopt the Cython provider

To close the performance gap without re‑writing hundreds of operators in
pure Python, follow these steps:

1. **Identify the hot provider code**
   * inspect `qlib/data/dataset/provider.py` (contains the Cython‑enabled
     expression evaluator and rolling mechanics).
   * copy only the necessary classes/utility functions into
     `scripts/Alpha/featureHandler`. Never import from qlib directly.

2. **Factor out the expression parser/evaluator**
   * the `_ExprEnv` and `evaluate_expression` in our provider are the
     Python versions; the qlib implementation uses primitives defined in
     C/Cython for rolling windows (`rolling_op`, `_rolling_apply`, …).
   * port those Cython utilities over.

3. **Re‑implement `LocalProvider.features()`** using the optimized
   routines.
   * the current method loops over instruments and builds the expression
     map in Python – replace this with migrated code from qlib.  The Cython
     implementation parallelises some operations and caches results.

4. **Bench and verify**
   * run `calc158.py` before/after each step to ensure timings improve and
     outputs remain identical.
   * write a small benchmark script (e.g. single‑instrument versus
     full‑universe) and commit results to `CythonPerf.md` as proof.

5. **Optional: compile the provider**
   * since we copy the qlib Cython code, add a lightweight `setup.py` or
     `pyproject.toml` entry to build the extension when installing the
     `featureHandler` package.

6. **Clean up Python fallback**
   * keep the pure‑Python code as a fallback for users who want a
     self‑contained version; guard it with `try/except ImportError` or a
     `USE_CYTHON = False` flag.
   * document the performance trade‑off in `CythonPerf.md` and `README`.

7. **Update tests**
   * extend existing tests (`test_dataloader` etc.) to run with both
     provider implementations and assert identical outputs.

Once these steps are complete you will retain the self‑contained nature
of `featureHandler` while regaining the high speeds of the original qlib
implementation.

---

The text above has been saved in `scripts/Alpha/CythonPerf.md` as you
requested; feel free to expand it with actual benchmark numbers when you
do further measurements.
