from inspect import getfullargspec

import pandas as pd

from .loader import Alpha158DL, Alpha360DL, QlibDataLoader
from .processor import Processor
from . import processor as processor_module
from .log import TimeInspector
from .utils import fetch_df_by_col, fetch_df_by_index, get_callable_kwargs, init_instance_by_config, lazy_sort_index


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
	new_l = []
	for proc in proc_l:
		if not isinstance(proc, Processor):
			klass, proc_kwargs = get_callable_kwargs(proc, processor_module)
			args = getfullargspec(klass).args
			if "fit_start_time" in args and "fit_end_time" in args:
				assert fit_start_time is not None and fit_end_time is not None
				proc_kwargs.update({"fit_start_time": fit_start_time, "fit_end_time": fit_end_time})
			proc_config = {"class": klass.__name__, "kwargs": proc_kwargs}
			new_l.append(proc_config)
		else:
			new_l.append(proc)
	return new_l


_DEFAULT_LEARN_PROCESSORS = [
	{"class": "DropnaLabel"},
	{"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
	{"class": "ProcessInf", "kwargs": {}},
	{"class": "ZScoreNorm", "kwargs": {}},
	{"class": "Fillna", "kwargs": {}},
]


class DataHandler:
	CS_ALL = "__all"
	DK_R = "raw"
	DK_I = "infer"
	DK_L = "learn"

	def __init__(self, instruments=None, start_time=None, end_time=None, data_loader=None, init_data=True, fetch_orig=True):
		self.data_loader = data_loader
		self.instruments = instruments
		self.start_time = start_time
		self.end_time = end_time
		self.fetch_orig = fetch_orig
		if init_data:
			with TimeInspector.logt("Init data"):
				self.setup_data()

	def setup_data(self):
		with TimeInspector.logt("Loading data"):
			self._data = lazy_sort_index(self.data_loader.load(self.instruments, self.start_time, self.end_time))

	def _fetch_data(self, data_storage, selector=slice(None, None), level="datetime", col_set=CS_ALL):
		data_df = fetch_df_by_col(data_storage, col_set)
		return fetch_df_by_index(data_df, selector, level, fetch_orig=self.fetch_orig)

	def fetch(self, selector=slice(None, None), level="datetime", col_set=CS_ALL, data_key=DK_I):
		_ = data_key
		return self._fetch_data(self._data, selector=selector, level=level, col_set=col_set)


class DataHandlerLP(DataHandler):
	PTYPE_I = "independent"
	PTYPE_A = "append"
	ATTR_MAP = {DataHandler.DK_R: "_data", DataHandler.DK_I: "_infer", DataHandler.DK_L: "_learn"}

	def __init__(
		self,
		instruments=None,
		start_time=None,
		end_time=None,
		data_loader=None,
		infer_processors=None,
		learn_processors=None,
		shared_processors=None,
		process_type=PTYPE_A,
		drop_raw=False,
		**kwargs,
	):
		infer_processors = [] if infer_processors is None else infer_processors
		learn_processors = [] if learn_processors is None else learn_processors
		shared_processors = [] if shared_processors is None else shared_processors
		self.infer_processors = [init_instance_by_config(proc, processor_module, accept_types=Processor) for proc in infer_processors]
		self.learn_processors = [init_instance_by_config(proc, processor_module, accept_types=Processor) for proc in learn_processors]
		self.shared_processors = [init_instance_by_config(proc, processor_module, accept_types=Processor) for proc in shared_processors]
		self.process_type = process_type
		self.drop_raw = drop_raw
		super().__init__(instruments, start_time, end_time, data_loader, **kwargs)

	def get_all_processors(self):
		return self.shared_processors + self.infer_processors + self.learn_processors

	@staticmethod
	def _run_proc_l(df, proc_l, with_fit, check_for_infer):
		for proc in proc_l:
			if check_for_infer and not proc.is_for_infer():
				raise TypeError("Only processors usable for inference can be used in infer_processors")
			with TimeInspector.logt(proc.__class__.__name__):
				if with_fit:
					proc.fit(df)
				df = proc(df)
		return df

	@staticmethod
	def _is_proc_readonly(proc_l):
		return all(proc.readonly() for proc in proc_l)

	def process_data(self, with_fit=False):
		shared_df = self._data if self._is_proc_readonly(self.shared_processors) else self._data.copy()
		shared_df = self._run_proc_l(shared_df, self.shared_processors, with_fit=with_fit, check_for_infer=True)

		infer_df = shared_df if self._is_proc_readonly(self.infer_processors) else shared_df.copy()
		infer_df = self._run_proc_l(infer_df, self.infer_processors, with_fit=with_fit, check_for_infer=True)
		self._infer = infer_df

		learn_df = shared_df if self.process_type == self.PTYPE_I else infer_df
		if not self._is_proc_readonly(self.learn_processors):
			learn_df = learn_df.copy()
		learn_df = self._run_proc_l(learn_df, self.learn_processors, with_fit=with_fit, check_for_infer=False)
		self._learn = learn_df

		if self.drop_raw:
			del self._data

	def setup_data(self):
		super().setup_data()
		with TimeInspector.logt("fit & process data"):
			self.process_data(with_fit=True)

	def fetch(self, selector=slice(None, None), level="datetime", col_set=DataHandler.CS_ALL, data_key=DataHandler.DK_I):
		return self._fetch_data(getattr(self, self.ATTR_MAP[data_key]), selector=selector, level=level, col_set=col_set)


class Alpha360(DataHandlerLP):
	def __init__(
		self,
		instruments="csi500",
		start_time=None,
		end_time=None,
		freq="day",
		infer_processors=_DEFAULT_INFER_PROCESSORS,
		learn_processors=_DEFAULT_LEARN_PROCESSORS,
		fit_start_time=None,
		fit_end_time=None,
		filter_pipe=None,
		inst_processors=None,
		**kwargs,
	):
		infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
		learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
		data_loader = QlibDataLoader(
			config={
				"feature": Alpha360DL.get_feature_config(),
				"label": kwargs.pop("label", self.get_label_config()),
			},
			filter_pipe=filter_pipe,
			freq=freq,
			inst_processors=inst_processors,
		)
		super().__init__(
			instruments=instruments,
			start_time=start_time,
			end_time=end_time,
			data_loader=data_loader,
			learn_processors=learn_processors,
			infer_processors=infer_processors,
			**kwargs,
		)

	def get_label_config(self):
		return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
	def __init__(
		self,
		instruments="csi500",
		start_time=None,
		end_time=None,
		freq="day",
		infer_processors=None,
		learn_processors=_DEFAULT_LEARN_PROCESSORS,
		fit_start_time=None,
		fit_end_time=None,
		process_type=DataHandlerLP.PTYPE_A,
		filter_pipe=None,
		inst_processors=None,
		**kwargs,
	):
		infer_processors = [] if infer_processors is None else infer_processors
		infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
		learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
		data_loader = QlibDataLoader(
			config={
				"feature": self.get_feature_config(),
				"label": kwargs.pop("label", self.get_label_config()),
			},
			filter_pipe=filter_pipe,
			freq=freq,
			inst_processors=inst_processors,
		)
		super().__init__(
			instruments=instruments,
			start_time=start_time,
			end_time=end_time,
			data_loader=data_loader,
			infer_processors=infer_processors,
			learn_processors=learn_processors,
			process_type=process_type,
			**kwargs,
		)

	def get_feature_config(self):
		config = {
			"kbar": {},
			"price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
			"rolling": {},
		}
		return Alpha158DL.get_feature_config(config)

	def get_label_config(self):
		return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
