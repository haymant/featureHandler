import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


def read_bin(file_path):
    file_path = Path(file_path).expanduser().resolve()
    with file_path.open("rb") as file_obj:
        data = np.fromfile(file_obj, dtype="<f")
    if data.size == 0:
        return None, pd.Series(dtype=np.float32)
    start_index = int(data[0])
    series = pd.Series(data[1:], index=pd.RangeIndex(start_index, start_index + data.size - 1), dtype=np.float32)
    return start_index, series


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    idx = df.index if axis == 0 else df.columns
    if not idx.is_monotonic_increasing:
        return df.sort_index(axis=axis)
    return df


def fetch_df_by_index(df, selector=slice(None, None), level="datetime", fetch_orig=True):
    _ = fetch_orig
    if level is None:
        return df.loc[selector]
    if isinstance(selector, slice):
        return df.loc[pd.IndexSlice[selector, :], :]
    return df.xs(selector, level=level, drop_level=False)


def fetch_df_by_col(df, col_set="__all"):
    if col_set in ("__all", None):
        return df
    if col_set == "__raw":
        return df
    if isinstance(col_set, list):
        return df.loc[:, col_set]
    if isinstance(df.columns, pd.MultiIndex):
        return df[col_set]
    return df.loc[:, col_set]


def get_module_by_module_path(module_path):
    if isinstance(module_path, ModuleType):
        return module_path
    return importlib.import_module(module_path)


def split_module_path(module_path):
    *module_parts, cls_name = module_path.split(".")
    return ".".join(module_parts), cls_name


def get_callable_kwargs(config, default_module=None):
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            module_path, cls_name = split_module_path(config[key])
            module = get_module_by_module_path(default_module if module_path == "" else module_path)
            callable_obj = getattr(module, cls_name)
        else:
            callable_obj = config[key]
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        module_path, cls_name = split_module_path(config)
        module = get_module_by_module_path(default_module if module_path == "" else module_path)
        callable_obj = getattr(module, cls_name)
        kwargs = {}
    else:
        raise NotImplementedError(f"Unsupported config type: {type(config)}")
    return callable_obj, kwargs


def init_instance_by_config(config, default_module=None, accept_types=None):
    if accept_types is not None and isinstance(config, accept_types):
        return config
    klass, kwargs = get_callable_kwargs(config, default_module)
    return klass(**kwargs)
