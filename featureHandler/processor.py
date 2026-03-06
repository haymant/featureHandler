import abc

import numpy as np
import pandas as pd

from .utils import fetch_df_by_index


def zscore(value):
    return (value - value.mean()).div(value.std())


def get_group_columns(df, group):
    if group is None:
        return df.columns
    return df.columns[df.columns.get_loc(group)]


class Processor:
    def fit(self, df=None):
        return None

    @abc.abstractmethod
    def __call__(self, df):
        raise NotImplementedError

    def is_for_infer(self):
        return True

    def readonly(self):
        return False

    def config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class DropnaProcessor(Processor):
    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))

    def readonly(self):
        return True


class DropnaLabel(DropnaProcessor):
    def __init__(self, fields_group="label"):
        super().__init__(fields_group=fields_group)

    def is_for_infer(self):
        return False


class ProcessInf(Processor):
    def __call__(self, df):
        for col in df.columns:
            mask = np.isinf(df[col])
            if mask.any():
                df[col] = df[col].replace([np.inf, -np.inf], df[col][~mask].mean())
        return df


class Fillna(Processor):
    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            df[self.fields_group] = df[self.fields_group].fillna(self.fill_value)
        return df


class ZScoreNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df=None):
        fit_df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(fit_df, self.fields_group)
        self.mean_train = np.nanmean(fit_df[cols].values, axis=0)
        self.std_train = np.nanstd(fit_df[cols].values, axis=0)
        self.std_train[self.std_train == 0] = 1
        self.cols = cols

    def __call__(self, df):
        df.loc(axis=1)[self.cols] = (df[self.cols].values - self.mean_train) / self.std_train
        return df


class CSZScoreNorm(Processor):
    def __init__(self, fields_group=None, method="zscore"):
        _ = method
        self.fields_group = fields_group

    def __call__(self, df):
        groups = self.fields_group if isinstance(self.fields_group, list) else [self.fields_group]
        with pd.option_context("mode.chained_assignment", None):
            for group in groups:
                cols = get_group_columns(df, group)
                df[cols] = df[cols].groupby("datetime", group_keys=False).apply(zscore)
        return df
