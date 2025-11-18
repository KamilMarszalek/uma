import logging
from dataclasses import asdict, is_dataclass

from experiments.logger.csv_handler import CSVHandler


class ExactLevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


DATA_TRACE_LEVEL = 5
logging.addLevelName(DATA_TRACE_LEVEL, "DATA_TRACE")


def data_trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(DATA_TRACE_LEVEL):
        if is_dataclass(msg):
            msg = asdict(msg)
        self._log(DATA_TRACE_LEVEL, msg, args, **kwargs)


logger = logging.getLogger("TournamentForestLogger")
logger.setLevel(DATA_TRACE_LEVEL)

logging.Logger.data_trace = data_trace

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
