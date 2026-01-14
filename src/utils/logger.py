import logging
from src.utils.config import Config


class Logger():
    def __init__(self, facility, level_to_show: str = None):
        self.logger = logging.getLogger(facility)
        if level_to_show is None:
            self._get_config()
        else:
            self.level_to_show = level_to_show
        self._setup_logger()

    def log(self, message: str, level: str):
        log_method = getattr(self.logger, level.lower())
        log_method(message)

    def _setup_logger(self):
        logging.basicConfig(
            level=getattr(logging, self.level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def _get_config(self):
        config = Config()
        logging_config = config.get_section("logging")
        self.level = logging_config.get("level", "INFO")
