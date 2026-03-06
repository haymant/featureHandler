import logging
from contextlib import contextmanager
from time import time

from .config import C


class _LoggerManager:
    def __call__(self, module_name, level=None):
        logger = logging.getLogger(f"featureHandler.{module_name}")
        logger.setLevel(C.logging_level if level is None else level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.propagate = False
        return logger

    def setLevel(self, level):
        logging.getLogger("featureHandler").setLevel(level)


get_module_logger = _LoggerManager()


class TimeInspector:
    timer_logger = get_module_logger("timer")
    time_marks = []

    @classmethod
    def set_time_mark(cls):
        cls.time_marks.append(time())

    @classmethod
    def log_cost_time(cls, info="Done"):
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: %.3fs | %s", cost_time, info)

    @classmethod
    @contextmanager
    def logt(cls, name="", show_start=False):
        if show_start:
            cls.timer_logger.info("%s Begin", name)
        cls.set_time_mark()
        try:
            yield None
        finally:
            cls.log_cost_time(info=f"{name} Done")
