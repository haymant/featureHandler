import logging
from pathlib import Path

REG_CN = "cn"
REG_US = "us"
REG_TW = "tw"
DEFAULT_FREQ = "__DEFAULT_FREQ"
LOCAL_URI = "local"


class DataPathManager:
    def __init__(self, provider_uri):
        self.provider_uri = self.format_provider_uri(provider_uri)

    @staticmethod
    def format_provider_uri(provider_uri):
        if isinstance(provider_uri, dict):
            return {k: Path(v).expanduser().resolve() for k, v in provider_uri.items()}
        return {DEFAULT_FREQ: Path(provider_uri).expanduser().resolve()}

    def get_data_uri(self, freq=None):
        if freq in (None, "day"):
            return next(iter(self.provider_uri.values()))
        return self.provider_uri.get(freq, next(iter(self.provider_uri.values())))

    @staticmethod
    def get_uri_type(provider_uri):
        return LOCAL_URI


class Config:
    def __init__(self):
        self.reset()

    def reset(self):
        self.provider_uri = {DEFAULT_FREQ: Path(".").resolve()}
        self.region = REG_CN
        self.logging_level = logging.INFO
        self.dpm = DataPathManager(self.provider_uri)
        self.registered = False

    def set(self, default_conf="client", provider_uri=None, region=None, logging_level=None, **kwargs):
        _ = default_conf, kwargs
        if provider_uri is not None:
            self.provider_uri = DataPathManager.format_provider_uri(provider_uri)
        if region is not None:
            self.region = region
        if logging_level is not None:
            self.logging_level = logging_level
        self.dpm = DataPathManager(self.provider_uri)

    def register(self):
        self.registered = True

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


C = Config()
