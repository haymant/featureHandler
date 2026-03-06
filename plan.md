# Minimal qlib extraction plan

This document describes the steps required to build a **minimal
standalone library** containing only the pieces of `qlib` that are
necessary to run `scripts/Alpha/calc158.py`.

The goal is to avoid copying the entire `qlib` package; instead we will
re‑implement or extract just the symbols referenced by the script and
their transitive dependencies.  The new package will live in
`scripts/Alpha/featureHandler` (and possibly adjacent modules) and the
script will import from it instead of from `qlib`.

## Overview

The script uses the following top‑level imports:

```python
from qlib.config import C, REG_US
from qlib.data import D
from qlib.contrib.data.handler import Alpha158, Alpha360
``` 

In order to satisfy these names we will need the underlying
implementations of:

1. `C` – configuration manager, including `DataPathManager` methods used
   by `init()` and `dpm.get_data_uri`.
2. `REG_US` – a constant (just a string or small struct) used by
   `qlib.init`.
3. `D` – the data provider facade, at least the `features` method called
   by `QlibDataLoader` and `instruments` helper.
4. `qlib.init()` – minimal initialisation that sets up `C` and possibly
   prepares the environment.

5. `DataHandlerLP` base class and its supporting code (fetch data,
   processing pipeline, `_run_proc_l`, etc.).
6. `Alpha158` and `Alpha360` classes (subclasses of
   `DataHandlerLP`) with their `get_feature_config` helpers.
7. Auxiliary modules used by these classes: `processor` definitions
   (e.g. `DropnaLabel`, `CSZScoreNorm`, `ZScoreNorm`, `Fillna`,
   `ProcessInf`, `MinMaxNorm` etc.) that are actually exercised during
   `setup_data`/`process_data`.
8. `qlib.data.dataset.loader` simplified version of `QlibDataLoader`
   which simply calls `D.features` and rearranges the index.
9. Utility functions: `fetch_df_by_index`, `fetch_df_by_col`,
   `lazy_sort_index`, `init_instance_by_config`, `get_full_argspec` or
   similar used by `check_transform_proc`, plus any time logger stubs.

Because the script only needs to load a handful of days for one
instrument, the implementations can be extremely lightweight: the
provider may simply load hard–coded CSV/Parquet files or return dummy
frames.  However, to reproduce the original behaviour we will simply
call into the existing `D` and handler code.  For test purposes we can
vendor enough of qlib such that `D.features` returns the same results
as before.

## Implementation Plan

1. **Create package skeleton**

   ```text
   scripts/Alpha/featureHandler/
       __init__.py
       config.py
       data.py
       contrib/
           __init__.py
           data/
               __init__.py
               handler.py
       data/dataset/
           __init__.py
           handler.py
           loader.py
           processor.py
       utils.py
       log.py           # maybe simple stub
   ```

   (paths may vary; adjust to keep imports simple e.g. `from featureHandler.config import C`.)

2. **Populate `config.py`** with the minimal `C` class, `REG_US` constant,
   and required `DataPathManager` methods used by the script (`get_data_uri`,
   `get_uri_type`).  Implementation can copy relevant parts of
   `qlib/config.py` and `qlib/utils/dpm.py`.

3. **Implement `data.py`** replicating the public `D` interface: at
   least `instruments()` (with same behaviour) and `features()` that
   actually delegates to whatever backend is available (possibly the
   original qlib provider).  To minimise copying we can import the real
   provider (if available) or replicate minimal logic.

4. **Copy `DataHandlerLP`, `DataHandler`, `DataHandlerABC` and
   auxiliary methods** from `qlib/data/dataset/handler.py` into the
   new package; trim out comments and unneeded methods (caching,
   serialization, etc.).  Ensure `fetch_df_by_index`/`fetch_df_by_col`
   and `Processor` base class are also included.

5. **Create `loader.py`** with a reduced `QlibDataLoader` that only
   needs to call `D.features` and swap levels.  The `DLWParser` logic can
   be retained or simplified; the goal is to keep support for the
   expression config used by Alpha158/360.

6. **Extract processors** that are actually exercised during a run of
   the script with `AAPL` and the given time period.  We can determine
   which processors are used by looking at the log (DropnaLabel,
   CSZScoreNorm, ProcessInf, ZScoreNorm, Fillna, MinMaxNorm).  Copy
   minimal implementations of these (their `fit` and `__call__` methods).
   Only include helpers they rely on (`get_group_columns`,
   `datetime_groupby_apply`, etc.).

7. **Copy the `Alpha158`/`Alpha360` classes** into
   `contrib/data/handler.py` and ensure they import from the local
   `DataHandlerLP` and `Alpha158DL`/`Alpha360DL` loaders.  Also copy
   the corresponding loaders or flatten them into the handler.

8. **Stub or replicate logging/time utilities** (e.g. `qlib.log.TimeInspector`)
   with no‑op implementations so that the script can still call
   `with TimeInspector.logt(...)` without error.

9. **Adjust imports in `calc158.py`** to import from the local
   `featureHandler` package instead of `qlib`.

10. **Test**: run `uv run scripts/Alpha/calc158.py` and compare output to
   `benchm.txt`.  The results should match exactly (same dataframe
   shapes, identical first five rows and column names).  If they
   differ, inspect which part of the package is missing and add it to
   the minimal library.

11. **Iterate**: add any missing function or class needed to make the
   script run.  As soon as the script completes successfully and the
   printed output agrees with `benchm.txt`, the minimal library is
   complete.

12. **Clean up**: remove unused code and verify that the package contains
   only the necessary modules (no extraneous utilities).  Document the
   extraction in `plan.md` for future reference.

## Testing Plan

*Run the refactored script and compare to the baseline output.*

1. Execute the original script with the full `qlib` installation and
   capture output (already done: `benchm.txt`).
2. Refactor `calc158.py` to import from `featureHandler` instead of
   `qlib`; update `sys.path` if needed.  Ensure the script no longer
   mentions `qlib` anywhere.
3. Run the refactored script (`uv run scripts/Alpha/calc158.py`) in a
   clean environment where `qlib` is *not* installed (you can uninstall
   it temporarily or manipulate `sys.path`).
4. Verify that:
   * the script terminates without import errors or exceptions,
   * the printed `DataHandlerLP` timing/logging messages still appear
     (they may be simpler if logging is stubbed),
   * the data shapes and first few rows match those in `benchm.txt`.
   * the column lists printed are identical.
5. If any discrepancy appears, trace it back to missing code in the
   minimal package (e.g. a helper used by a processor) and add it.
6. Once the outputs match fully, the extraction is successful.

Optionally, run a few additional experiments, e.g. change the
instrument list or date range, to make sure no hidden dependency is
locked to the original environment.

## Deliverable

- `scripts/Alpha/featureHandler/` – a self‑contained, importable mini‑qlib.
- `scripts/Alpha/calc158.py` – updated script using the local package.
- `scripts/Alpha/plan.md` – this file, explaining the plan and test
  procedure.

With this setup you can ship the script plus the minimal library to a
machine that does not have `qlib` installed, allowing others to run the
alpha‑feature calculation without pulling in the entire project.