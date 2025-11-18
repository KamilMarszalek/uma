import csv
import logging
import os


class CSVHandler(logging.Handler):
    def __init__(self, filename, fieldnames):
        super().__init__()
        self.filename = filename
        self.fieldnames = fieldnames

        file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
        self.file = open(filename, mode="a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def emit(self, record):
        try:
            self.writer.writerow(record.msg)
            self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self.file.close()
        super().close()
