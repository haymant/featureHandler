from pathlib import Path
import subprocess
import sys

from .config import C, REG_US
from .log import get_module_logger


def _ensure_extensions_built():
	libs_dir = Path(__file__).resolve().parent / "_libs"
	if list(libs_dir.glob("rolling*.so")) or list(libs_dir.glob("rolling*.pyd")):
		return
	build_script = Path(__file__).resolve().parent / "build_ext.py"
	if not build_script.exists():
		return
	subprocess.run(
		[sys.executable, str(build_script), "build_ext", "--inplace"],
		cwd=str(build_script.parent.parent),
		check=True,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)


_ensure_extensions_built()

from .provider import D
from .Handler import Alpha158, Alpha360, DataHandlerLP


def init(default_conf="client", **kwargs):
	_ = default_conf
	logger = get_module_logger("Initialization")
	C.set(**kwargs)
	D.reset()
	C.register()
	logger.info("featureHandler initialized.")
	logger.info("data_path=%s", {"__DEFAULT_FREQ": C.dpm.get_data_uri("day")})


__all__ = ["init", "C", "REG_US", "D", "Alpha158", "Alpha360", "DataHandlerLP"]
