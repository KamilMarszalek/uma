import csv
import logging
import os
from dataclasses import asdict, is_dataclass


class CSVHandler(logging.Handler):
    def __init__(self, filename: str, fieldnames: list[str]):
        super().__init__()
        self.filename = filename
        self.fieldnames = fieldnames

        file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
        self.file = open(filename, mode="a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg_dict = record.msg
            if not isinstance(msg_dict, dict):
                if is_dataclass(msg_dict) and not isinstance(msg_dict, type):
                    msg_dict = asdict(msg_dict)
                else:
                    # jeśli nie da się skonwertować, pomiń lub rzuć wyjątek
                    raise TypeError(f"record.msg must be a dict, got {type(msg_dict)}")
            self.writer.writerow(msg_dict)
            self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self.file.close()
        super().close()
