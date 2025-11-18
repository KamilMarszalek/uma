import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from experiments.logger.csv_handler import CSVHandler


class ExactLevelFilter(logging.Filter):
    def __init__(self, level: int):
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


DATA_TRACE_LEVEL = 5
logging.addLevelName(DATA_TRACE_LEVEL, "DATA_TRACE")


def log_data_trace(
    self: logging.Logger, msg: Any, *args: object, exc_info: Any = None
) -> None:
    if self.isEnabledFor(DATA_TRACE_LEVEL):
        if is_dataclass(msg) and not isinstance(msg, type):
            msg = asdict(msg)
        self._log(DATA_TRACE_LEVEL, msg, args, exc_info=exc_info)


class TournamentLogger(logging.Logger):
    def data_trace(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(DATA_TRACE_LEVEL):
            if is_dataclass(msg) and not isinstance(msg, type):
                msg = asdict(msg)
            self._log(DATA_TRACE_LEVEL, msg, args, **kwargs)


logger = TournamentLogger("TournamentForestLogger", level=DATA_TRACE_LEVEL)
logger.setLevel(DATA_TRACE_LEVEL)

console_info_handler = logging.StreamHandler()
console_info_handler.setLevel(logging.INFO)
console_info_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

data_trace_handler = CSVHandler(
    "data_trace.csv",
    fieldnames=[
        "experiment",
        "forest_type",
        "eval_function",
        "num_trees",
        "sample_ratio",
        "feature_ratio",
        "tree_max_depth",
        "tree_tournament_size",
        "set_id",
        "train_size",
        "random_seed",
        "categorial_encoding",
        "time_of_building",
        "accuracy",
    ],
)
data_trace_handler.setLevel(DATA_TRACE_LEVEL)
data_trace_handler.addFilter(ExactLevelFilter(DATA_TRACE_LEVEL))


logger.addHandler(console_info_handler)
logger.addHandler(data_trace_handler)
