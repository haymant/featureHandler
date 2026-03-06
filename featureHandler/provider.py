import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import C
from .log import get_module_logger
from .utils import read_bin


_FIELD_NAMES = ("close", "open", "high", "low", "volume", "factor", "change", "vwap")
_FIELD_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


class LocalProvider:
    def __init__(self):
        self.logger = get_module_logger("provider")
        self._calendar = None
        self._raw_cache = {}

    def reset(self):
        self._calendar = None
        self._raw_cache.clear()

    @property
    def root(self):
        return Path(C.dpm.get_data_uri("day"))

    def calendar(self):
        if self._calendar is None:
            calendar_path = self.root / "calendars" / "day.txt"
            self._calendar = pd.DatetimeIndex(pd.read_csv(calendar_path, header=None).iloc[:, 0])
        return self._calendar

    def instruments(self, market="all", filter_pipe=None, start_time=None, end_time=None, as_list=False):
        _ = filter_pipe, start_time, end_time, as_list
        if isinstance(market, list):
            return market
        if isinstance(market, str):
            if market.lower() == "all":
                return sorted(p.name.upper() for p in (self.root / "features").iterdir() if p.is_dir())
            return [market.upper()]
        raise TypeError(f"Unsupported instruments type: {type(market)}")

    def _load_field(self, instrument, field):
        bin_path = self.root / "features" / instrument.lower() / f"{field}.day.bin"
        calendar = self.calendar()
        if not bin_path.exists():
            return pd.Series(np.nan, index=calendar, dtype=np.float32)
        _, series = read_bin(bin_path)
        full_series = pd.Series(np.nan, index=calendar, dtype=np.float32)
        full_series.loc[calendar[series.index]] = series.values
        return full_series

    def _load_raw_frame(self, instrument):
        instrument = instrument.upper()
        if instrument in self._raw_cache:
            return self._raw_cache[instrument]
        data = {field: self._load_field(instrument, field) for field in _FIELD_NAMES if field != "vwap"}
        data["vwap"] = self._load_field(instrument, "vwap")
        raw_df = pd.DataFrame(data, index=self.calendar())
        raw_df.index.name = "datetime"
        self._raw_cache[instrument] = raw_df
        return raw_df

    def features(self, instruments, exprs, start_time=None, end_time=None, freq="day", inst_processors=None):
        _ = freq, inst_processors
        instrument_list = self.instruments(instruments) if not isinstance(instruments, list) else instruments
        frames = []
        for instrument in instrument_list:
            raw_df = self._load_raw_frame(instrument)
            expr_map = {expr: evaluate_expression(expr, raw_df) for expr in exprs}
            expr_df = pd.DataFrame(expr_map, index=raw_df.index)
            expr_df = expr_df.loc[start_time:end_time]
            expr_df["instrument"] = instrument.upper()
            frames.append(expr_df.reset_index().set_index(["instrument", "datetime"]))
        if not frames:
            return pd.DataFrame(columns=exprs)
        return pd.concat(frames).sort_index()


class _FieldAccessor:
    def __init__(self, raw_df):
        self.raw_df = raw_df

    def __call__(self, field_name):
        return self.raw_df[field_name]


def _as_series(value, index):
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index)


class _ExprEnv(dict):
    def __init__(self, raw_df):
        index = raw_df.index
        field = _FieldAccessor(raw_df)

        def rolling(series, window, func_name):
            if int(window) == 0:
                obj = series.expanding(min_periods=1)
            else:
                obj = series.rolling(int(window), min_periods=1)
            return getattr(obj, func_name)()

        def Ref(series, n):
            return series.shift(int(n))

        def Mean(series, n):
            return rolling(series, n, "mean")

        def Sum(series, n):
            return rolling(series, n, "sum")

        def Std(series, n):
            return rolling(series, n, "std")

        def Max(series, n):
            return rolling(series, n, "max")

        def Min(series, n):
            return rolling(series, n, "min")

        def Quantile(series, n, qscore):
            obj = series.expanding(min_periods=1) if int(n) == 0 else series.rolling(int(n), min_periods=1)
            return obj.quantile(qscore)

        def Rank(series, n):
            obj = series.expanding(min_periods=1) if int(n) == 0 else series.rolling(int(n), min_periods=1)
            if hasattr(obj, "rank"):
                return obj.rank(pct=True)
            return obj.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        def IdxMax(series, n):
            obj = series.expanding(min_periods=1) if int(n) == 0 else series.rolling(int(n), min_periods=1)
            return obj.apply(lambda x: np.argmax(x) + 1 if len(x) else np.nan, raw=True)

        def IdxMin(series, n):
            obj = series.expanding(min_periods=1) if int(n) == 0 else series.rolling(int(n), min_periods=1)
            return obj.apply(lambda x: np.argmin(x) + 1 if len(x) else np.nan, raw=True)

        def _binary_op(left, right, op):
            left_s = _as_series(left, index)
            right_s = _as_series(right, index)
            return pd.Series(op(left_s.to_numpy(), right_s.to_numpy()), index=index)

        def Greater(left, right):
            return _binary_op(left, right, np.maximum)

        def Less(left, right):
            return _binary_op(left, right, np.minimum)

        def Abs(series):
            return np.abs(series)

        def Log(series):
            return np.log(series)

        def Corr(left, right, n):
            window = int(n)
            if window == 0:
                return left.expanding(min_periods=1).corr(right)
            return left.rolling(window, min_periods=1).corr(right)

        def _linreg(values):
            x = np.arange(1, len(values) + 1, dtype=float)
            mask = ~np.isnan(values)
            x = x[mask]
            y = values[mask]
            if y.size < 2:
                return np.nan, np.nan, np.nan
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            rsquare = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
            residual = y[-1] - y_pred[-1]
            return slope, rsquare, residual

        def _rolling_regression(series, window, value_index):
            win = int(window)
            obj = series.expanding(min_periods=1) if win == 0 else series.rolling(win, min_periods=1)
            return obj.apply(lambda arr: _linreg(arr)[value_index], raw=True)

        def Slope(series, n):
            return _rolling_regression(series, n, 0)

        def Rsquare(series, n):
            return _rolling_regression(series, n, 1)

        def Resi(series, n):
            return _rolling_regression(series, n, 2)

        super().__init__(
            {
                "field": field,
                "Ref": Ref,
                "Mean": Mean,
                "Sum": Sum,
                "Std": Std,
                "Max": Max,
                "Min": Min,
                "Quantile": Quantile,
                "Rank": Rank,
                "IdxMax": IdxMax,
                "IdxMin": IdxMin,
                "Greater": Greater,
                "Less": Less,
                "Abs": Abs,
                "Log": Log,
                "Corr": Corr,
                "Slope": Slope,
                "Rsquare": Rsquare,
                "Resi": Resi,
                "np": np,
            }
        )


def evaluate_expression(expr, raw_df):
    translated = _FIELD_PATTERN.sub(lambda match: f'field("{match.group(1).lower()}")', expr)
    env = _ExprEnv(raw_df)
    result = eval(translated, {"__builtins__": {}}, env)
    if isinstance(result, pd.Series):
        return result.astype(np.float32)
    return pd.Series(result, index=raw_df.index, dtype=np.float32)


D = LocalProvider()
