import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import C
from .log import get_module_logger
from .utils import read_bin

try:
    from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
except ImportError:
    rolling_slope = None
    rolling_rsquare = None
    rolling_resi = None


_FIELD_NAMES = ("close", "open", "high", "low", "volume", "factor", "change", "vwap")


def parse_field(field):
    if not isinstance(field, str):
        field = str(field)
    for pattern, new in [
        (r"\$([\w]+)", r'Feature("\1")'),
        (r"(\w+\s*)\(", r"Operators.\1("),
    ]:
        field = re.sub(pattern, new, field)
    return field


class Expression:
    _cache = {}

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Ge(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Le(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def load(self, instrument, start_index, end_index, freq):
        cache_key = (id(self), instrument, start_index, end_index, freq)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_internal(instrument, start_index, end_index, freq)
        return self._cache[cache_key]

    def _load_internal(self, instrument, start_index, end_index, freq):
        raise NotImplementedError

    def get_extended_window_size(self):
        raise NotImplementedError


class Feature(Expression):
    def __init__(self, name):
        self.name = name.lower()

    def __str__(self):
        return "$" + self.name

    def _load_internal(self, instrument, start_index, end_index, freq):
        _ = freq
        return D.feature(instrument, self.name, start_index, end_index)

    def get_extended_window_size(self):
        return 0, 0


class ExpressionOps(Expression):
    pass


class ElemOperator(ExpressionOps):
    def __init__(self, feature):
        self.feature = feature

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class PairOperator(ExpressionOps):
    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def get_extended_window_size(self):
        left_left, left_right = self.feature_left.get_extended_window_size() if isinstance(self.feature_left, Expression) else (0, 0)
        right_left, right_right = self.feature_right.get_extended_window_size() if isinstance(self.feature_right, Expression) else (0, 0)
        return max(left_left, right_left), max(left_right, right_right)

    def _load_feature(self, feature, instrument, start_index, end_index, freq, index):
        if isinstance(feature, Expression):
            return feature.load(instrument, start_index, end_index, freq)
        return pd.Series(feature, index=index)


class NpElemOperator(ElemOperator):
    def __init__(self, feature, func):
        super().__init__(feature)
        self.func = func

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return getattr(np, self.func)(series)


class Abs(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "abs")


class Log(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "log")


class BinaryOperator(PairOperator):
    func = None

    def _load_internal(self, instrument, start_index, end_index, freq):
        left_is_expr = isinstance(self.feature_left, Expression)
        right_is_expr = isinstance(self.feature_right, Expression)
        if left_is_expr:
            left = self.feature_left.load(instrument, start_index, end_index, freq)
            index = left.index
        elif right_is_expr:
            right = self.feature_right.load(instrument, start_index, end_index, freq)
            index = right.index
            left = pd.Series(self.feature_left, index=index)
            return self.func(left, right)
        else:
            return self.func(self.feature_left, self.feature_right)

        if right_is_expr:
            right = self.feature_right.load(instrument, start_index, end_index, freq)
        else:
            right = pd.Series(self.feature_right, index=index)
        return self.func(left, right)


class Add(BinaryOperator):
    func = staticmethod(lambda left, right: left + right)


class Sub(BinaryOperator):
    func = staticmethod(lambda left, right: left - right)


class Mul(BinaryOperator):
    func = staticmethod(lambda left, right: left * right)


class Div(BinaryOperator):
    func = staticmethod(lambda left, right: left / right)


class Greater(BinaryOperator):
    func = staticmethod(lambda left, right: np.maximum(left, right))


class Less(BinaryOperator):
    func = staticmethod(lambda left, right: np.minimum(left, right))


class Gt(BinaryOperator):
    func = staticmethod(lambda left, right: left > right)


class Ge(BinaryOperator):
    func = staticmethod(lambda left, right: left >= right)


class Lt(BinaryOperator):
    func = staticmethod(lambda left, right: left < right)


class Le(BinaryOperator):
    func = staticmethod(lambda left, right: left <= right)


class Rolling(ExpressionOps):
    def __init__(self, feature, window, func):
        self.feature = feature
        self.window = window
        self.func = func

    def get_extended_window_size(self):
        left, right = self.feature.get_extended_window_size()
        if self.window == 0:
            return left, right
        return max(left + self.window - 1, left), right

    def _rolling(self, series):
        if self.window == 0:
            return getattr(series.expanding(min_periods=1), self.func)()
        return getattr(series.rolling(self.window, min_periods=1), self.func)()

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return self._rolling(series)


class Ref(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "ref")

    def get_extended_window_size(self):
        left, right = self.feature.get_extended_window_size()
        return max(left + self.window, left), max(right - self.window, right)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.window == 0:
            return pd.Series(series.iloc[0], index=series.index)
        return series.shift(self.window)


class Mean(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "mean")


class Sum(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "sum")


class Std(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "std")


class Max(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "max")


class Min(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "min")


class Quantile(Rolling):
    def __init__(self, feature, window, qscore):
        super().__init__(feature, window, "quantile")
        self.qscore = qscore

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.window == 0:
            return series.expanding(min_periods=1).quantile(self.qscore)
        return series.rolling(self.window, min_periods=1).quantile(self.qscore)


class Rank(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "rank")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        obj = series.expanding(min_periods=1) if self.window == 0 else series.rolling(self.window, min_periods=1)
        if hasattr(obj, "rank"):
            return obj.rank(pct=True)
        return obj.apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)


class IdxMax(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "idxmax")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        obj = series.expanding(min_periods=1) if self.window == 0 else series.rolling(self.window, min_periods=1)
        return obj.apply(lambda values: values.argmax() + 1, raw=True)


class IdxMin(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "idxmin")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        obj = series.expanding(min_periods=1) if self.window == 0 else series.rolling(self.window, min_periods=1)
        return obj.apply(lambda values: values.argmin() + 1, raw=True)


class Corr(PairOperator):
    def __init__(self, feature_left, feature_right, window):
        super().__init__(feature_left, feature_right)
        self.window = window

    def get_extended_window_size(self):
        left, right = super().get_extended_window_size()
        return max(left + self.window - 1, left), right

    def _load_internal(self, instrument, start_index, end_index, freq):
        left = self.feature_left.load(instrument, start_index, end_index, freq)
        right = self.feature_right.load(instrument, start_index, end_index, freq)
        if self.window == 0:
            return left.expanding(min_periods=1).corr(right)
        return left.rolling(self.window, min_periods=1).corr(right)


class Slope(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "slope")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.window == 0 or rolling_slope is None:
            return series.expanding(min_periods=2).apply(_expanding_slope, raw=True)
        return pd.Series(rolling_slope(series.values.astype(np.float64), self.window), index=series.index)


class Rsquare(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "rsquare")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.window == 0 or rolling_rsquare is None:
            return series.expanding(min_periods=2).apply(_expanding_rsquare, raw=True)
        result = pd.Series(rolling_rsquare(series.values.astype(np.float64), self.window), index=series.index)
        result.loc[np.isclose(series.rolling(self.window, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return result


class Resi(Rolling):
    def __init__(self, feature, window):
        super().__init__(feature, window, "resi")

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if self.window == 0 or rolling_resi is None:
            return series.expanding(min_periods=2).apply(_expanding_resi, raw=True)
        return pd.Series(rolling_resi(series.values.astype(np.float64), self.window), index=series.index)


def _linreg(values):
    values = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(values)
    values = values[mask]
    if values.size < 2:
        return np.nan, np.nan, np.nan
    x_values = np.arange(1, values.size + 1, dtype=np.float64)
    slope, intercept = np.polyfit(x_values, values, 1)
    y_pred = slope * x_values + intercept
    ss_res = ((values - y_pred) ** 2).sum()
    ss_tot = ((values - values.mean()) ** 2).sum()
    rsquare = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
    residual = values[-1] - y_pred[-1]
    return slope, rsquare, residual


def _expanding_slope(values):
    return _linreg(values)[0]


def _expanding_rsquare(values):
    return _linreg(values)[1]


def _expanding_resi(values):
    return _linreg(values)[2]


class OpsWrapper:
    def __getattr__(self, key):
        return globals()[key]


Operators = OpsWrapper()


class LocalProvider:
    def __init__(self):
        self.logger = get_module_logger("provider")
        self._calendar = None
        self._calendar_index = None
        self._field_cache = {}
        self._expr_cache = {}

    def reset(self):
        self._calendar = None
        self._calendar_index = None
        self._field_cache.clear()
        self._expr_cache.clear()
        Expression._cache.clear()

    @property
    def root(self):
        return Path(C.dpm.get_data_uri("day"))

    def calendar(self):
        if self._calendar is None:
            calendar_path = self.root / "calendars" / "day.txt"
            self._calendar = pd.DatetimeIndex(pd.read_csv(calendar_path, header=None).iloc[:, 0])
            self._calendar_index = {timestamp: index for index, timestamp in enumerate(self._calendar)}
        return self._calendar

    def locate_index(self, start_time=None, end_time=None):
        calendar = self.calendar()
        start_time = pd.Timestamp(start_time) if start_time is not None else calendar[0]
        end_time = pd.Timestamp(end_time) if end_time is not None else calendar[-1]
        if start_time not in self._calendar_index:
            start_time = calendar[np.searchsorted(calendar, start_time)]
        if end_time not in self._calendar_index:
            end_time = calendar[np.searchsorted(calendar, end_time, side="right") - 1]
        return self._calendar_index[start_time], self._calendar_index[end_time]

    def instruments(self, market="all", filter_pipe=None, start_time=None, end_time=None, as_list=False):
        _ = filter_pipe, start_time, end_time, as_list
        if isinstance(market, list):
            return market
        if isinstance(market, str):
            if market.lower() == "all":
                return sorted(path.name.upper() for path in (self.root / "features").iterdir() if path.is_dir())
            return [market.upper()]
        raise TypeError(f"Unsupported instruments type: {type(market)}")

    def feature(self, instrument, field, start_index, end_index):
        instrument = instrument.upper()
        cache_key = (instrument, field)
        if cache_key not in self._field_cache:
            calendar = self.calendar()
            bin_path = self.root / "features" / instrument.lower() / f"{field}.day.bin"
            if not bin_path.exists():
                self._field_cache[cache_key] = pd.Series(np.nan, index=calendar, dtype=np.float32)
            else:
                _, series = read_bin(bin_path)
                full_series = pd.Series(np.nan, index=calendar, dtype=np.float32)
                full_series.iloc[series.index.values] = series.values
                self._field_cache[cache_key] = full_series
        return self._field_cache[cache_key].iloc[start_index : end_index + 1]

    def get_expression_instance(self, field):
        if field not in self._expr_cache:
            self._expr_cache[field] = eval(parse_field(field), {"Feature": Feature, "Operators": Operators})
        return self._expr_cache[field]

    def expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
        _ = freq
        expression = self.get_expression_instance(field)
        start_index, end_index = self.locate_index(start_time, end_time)
        left_extend, right_extend = expression.get_extended_window_size()
        query_start = max(0, start_index - left_extend)
        query_end = min(len(self.calendar()) - 1, end_index + right_extend)
        series = expression.load(instrument, query_start, query_end, freq)
        series = series.astype(np.float32, copy=False)
        if not series.empty:
            series = series.iloc[start_index - query_start : end_index - query_start + 1]
        return series

    def features(self, instruments, exprs, start_time=None, end_time=None, freq="day", inst_processors=None):
        _ = inst_processors
        instrument_list = self.instruments(instruments) if not isinstance(instruments, list) else instruments
        frames = []
        for instrument in instrument_list:
            frame = pd.DataFrame({expr: self.expression(instrument, expr, start_time, end_time, freq) for expr in exprs})
            frame.index.name = "datetime"
            frame["instrument"] = instrument.upper()
            frames.append(frame.reset_index().set_index(["instrument", "datetime"]))
        if not frames:
            return pd.DataFrame(columns=exprs)
        return pd.concat(frames).sort_index()


D = LocalProvider()
