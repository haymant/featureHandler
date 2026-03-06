import warnings

import pandas as pd

from .provider import D


class DataLoader:
    def load(self, instruments, start_time=None, end_time=None):
        raise NotImplementedError


class DLWParser(DataLoader):
    def __init__(self, config):
        self.is_group = isinstance(config, dict)
        if self.is_group:
            self.fields = {group: self._parse_fields_info(fields_info) for group, fields_info in config.items()}
        else:
            self.fields = self._parse_fields_info(config)

    def _parse_fields_info(self, fields_info):
        if isinstance(fields_info[0], str):
            return fields_info, fields_info
        return fields_info

    def load_group_df(self, instruments, exprs, names, start_time=None, end_time=None, gp_name=None):
        raise NotImplementedError

    def load(self, instruments=None, start_time=None, end_time=None):
        if self.is_group:
            return pd.concat(
                {
                    group: self.load_group_df(instruments, exprs, names, start_time, end_time, group)
                    for group, (exprs, names) in self.fields.items()
                },
                axis=1,
            )
        exprs, names = self.fields
        return self.load_group_df(instruments, exprs, names, start_time, end_time)


class QlibDataLoader(DLWParser):
    def __init__(self, config, filter_pipe=None, swap_level=True, freq="day", inst_processors=None):
        self.filter_pipe = filter_pipe
        self.swap_level = swap_level
        self.freq = freq
        self.inst_processors = inst_processors if inst_processors is not None else {}
        super().__init__(config)

    def load_group_df(self, instruments, exprs, names, start_time=None, end_time=None, gp_name=None):
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            instruments = "all"
        freq = self.freq[gp_name] if isinstance(self.freq, dict) else self.freq
        inst_processors = self.inst_processors if isinstance(self.inst_processors, list) else self.inst_processors.get(gp_name, [])
        df = D.features(instruments, exprs, start_time, end_time, freq=freq, inst_processors=inst_processors)
        df.columns = names
        if self.swap_level:
            df = df.swaplevel().sort_index()
        return df


class Alpha360DL(QlibDataLoader):
    def __init__(self, config=None, **kwargs):
        cfg = {"feature": self.get_feature_config()}
        if config is not None:
            cfg.update(config)
        super().__init__(config=cfg, **kwargs)

    @staticmethod
    def get_feature_config():
        fields = []
        names = []
        for index in range(59, 0, -1):
            fields += [f"Ref($close, {index})/$close"]
            names += [f"CLOSE{index}"]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for index in range(59, 0, -1):
            fields += [f"Ref($open, {index})/$close"]
            names += [f"OPEN{index}"]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for index in range(59, 0, -1):
            fields += [f"Ref($high, {index})/$close"]
            names += [f"HIGH{index}"]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for index in range(59, 0, -1):
            fields += [f"Ref($low, {index})/$close"]
            names += [f"LOW{index}"]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for index in range(59, 0, -1):
            fields += [f"Ref($vwap, {index})/$close"]
            names += [f"VWAP{index}"]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for index in range(59, 0, -1):
            fields += [f"Ref($volume, {index})/($volume+1e-12)"]
            names += [f"VOLUME{index}"]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]
        return fields, names


class Alpha158DL(QlibDataLoader):
    def __init__(self, config=None, **kwargs):
        cfg = {"feature": self.get_feature_config()}
        if config is not None:
            cfg.update(config)
        super().__init__(config=cfg, **kwargs)

    @staticmethod
    def get_feature_config(config=None):
        if config is None:
            config = {
                "kbar": {},
                "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
                "rolling": {},
            }
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            features = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in features:
                field = field.lower()
                fields += [f"Ref(${field}, {day})/$close" if day != 0 else f"${field}/$close" for day in windows]
                names += [field.upper() + str(day) for day in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += [f"Ref($volume, {day})/($volume+1e-12)" if day != 0 else "$volume/($volume+1e-12)" for day in windows]
            names += ["VOLUME" + str(day) for day in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])

            def use(name):
                return name not in exclude and (include is None or name in include)

            if use("ROC"):
                fields += [f"Ref($close, {day})/$close" for day in windows]
                names += [f"ROC{day}" for day in windows]
            if use("MA"):
                fields += [f"Mean($close, {day})/$close" for day in windows]
                names += [f"MA{day}" for day in windows]
            if use("STD"):
                fields += [f"Std($close, {day})/$close" for day in windows]
                names += [f"STD{day}" for day in windows]
            if use("BETA"):
                fields += [f"Slope($close, {day})/$close" for day in windows]
                names += [f"BETA{day}" for day in windows]
            if use("RSQR"):
                fields += [f"Rsquare($close, {day})" for day in windows]
                names += [f"RSQR{day}" for day in windows]
            if use("RESI"):
                fields += [f"Resi($close, {day})/$close" for day in windows]
                names += [f"RESI{day}" for day in windows]
            if use("MAX"):
                fields += [f"Max($high, {day})/$close" for day in windows]
                names += [f"MAX{day}" for day in windows]
            if use("LOW"):
                fields += [f"Min($low, {day})/$close" for day in windows]
                names += [f"MIN{day}" for day in windows]
            if use("QTLU"):
                fields += [f"Quantile($close, {day}, 0.8)/$close" for day in windows]
                names += [f"QTLU{day}" for day in windows]
            if use("QTLD"):
                fields += [f"Quantile($close, {day}, 0.2)/$close" for day in windows]
                names += [f"QTLD{day}" for day in windows]
            if use("RANK"):
                fields += [f"Rank($close, {day})" for day in windows]
                names += [f"RANK{day}" for day in windows]
            if use("RSV"):
                fields += [f"($close-Min($low, {day}))/(Max($high, {day})-Min($low, {day})+1e-12)" for day in windows]
                names += [f"RSV{day}" for day in windows]
            if use("IMAX"):
                fields += [f"IdxMax($high, {day})/{day}" for day in windows]
                names += [f"IMAX{day}" for day in windows]
            if use("IMIN"):
                fields += [f"IdxMin($low, {day})/{day}" for day in windows]
                names += [f"IMIN{day}" for day in windows]
            if use("IMXD"):
                fields += [f"(IdxMax($high, {day})-IdxMin($low, {day}))/{day}" for day in windows]
                names += [f"IMXD{day}" for day in windows]
            if use("CORR"):
                fields += [f"Corr($close, Log($volume+1), {day})" for day in windows]
                names += [f"CORR{day}" for day in windows]
            if use("CORD"):
                fields += [f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {day})" for day in windows]
                names += [f"CORD{day}" for day in windows]
            if use("CNTP"):
                fields += [f"Mean($close>Ref($close, 1), {day})" for day in windows]
                names += [f"CNTP{day}" for day in windows]
            if use("CNTN"):
                fields += [f"Mean($close<Ref($close, 1), {day})" for day in windows]
                names += [f"CNTN{day}" for day in windows]
            if use("CNTD"):
                fields += [f"Mean($close>Ref($close, 1), {day})-Mean($close<Ref($close, 1), {day})" for day in windows]
                names += [f"CNTD{day}" for day in windows]
            if use("SUMP"):
                fields += [f"Sum(Greater($close-Ref($close, 1), 0), {day})/(Sum(Abs($close-Ref($close, 1)), {day})+1e-12)" for day in windows]
                names += [f"SUMP{day}" for day in windows]
            if use("SUMN"):
                fields += [f"Sum(Greater(Ref($close, 1)-$close, 0), {day})/(Sum(Abs($close-Ref($close, 1)), {day})+1e-12)" for day in windows]
                names += [f"SUMN{day}" for day in windows]
            if use("SUMD"):
                fields += [f"(Sum(Greater($close-Ref($close, 1), 0), {day})-Sum(Greater(Ref($close, 1)-$close, 0), {day}))/(Sum(Abs($close-Ref($close, 1)), {day})+1e-12)" for day in windows]
                names += [f"SUMD{day}" for day in windows]
            if use("VMA"):
                fields += [f"Mean($volume, {day})/($volume+1e-12)" for day in windows]
                names += [f"VMA{day}" for day in windows]
            if use("VSTD"):
                fields += [f"Std($volume, {day})/($volume+1e-12)" for day in windows]
                names += [f"VSTD{day}" for day in windows]
            if use("WVMA"):
                fields += [f"Std(Abs($close/Ref($close, 1)-1)*$volume, {day})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {day})+1e-12)" for day in windows]
                names += [f"WVMA{day}" for day in windows]
            if use("VSUMP"):
                fields += [f"Sum(Greater($volume-Ref($volume, 1), 0), {day})/(Sum(Abs($volume-Ref($volume, 1)), {day})+1e-12)" for day in windows]
                names += [f"VSUMP{day}" for day in windows]
            if use("VSUMN"):
                fields += [f"Sum(Greater(Ref($volume, 1)-$volume, 0), {day})/(Sum(Abs($volume-Ref($volume, 1)), {day})+1e-12)" for day in windows]
                names += [f"VSUMN{day}" for day in windows]
            if use("VSUMD"):
                fields += [f"(Sum(Greater($volume-Ref($volume, 1), 0), {day})-Sum(Greater(Ref($volume, 1)-$volume, 0), {day}))/(Sum(Abs($volume-Ref($volume, 1)), {day})+1e-12)" for day in windows]
                names += [f"VSUMD{day}" for day in windows]
        return fields, names
