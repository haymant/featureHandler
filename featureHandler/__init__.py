from .config import C, REG_US
from .provider import D
from .Handler import Alpha158, Alpha360, DataHandlerLP
from .log import get_module_logger


def init(default_conf="client", **kwargs):
	_ = default_conf
	logger = get_module_logger("Initialization")
	C.set(**kwargs)
	D.reset()
	C.register()
	logger.info("featureHandler initialized.")
	logger.info("data_path=%s", {"__DEFAULT_FREQ": C.dpm.get_data_uri("day")})


__all__ = ["init", "C", "REG_US", "D", "Alpha158", "Alpha360", "DataHandlerLP"]
