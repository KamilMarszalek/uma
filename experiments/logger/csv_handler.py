import csv
import logging


class CSVHandler(logging.Handler):
    def __init__(self, filename, fieldnames):
        super().__init__()
        self.filename = filename
        self.fieldnames = fieldnames
        self.file = open(filename, mode="w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
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


