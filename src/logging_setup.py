"""Structured logging: one JSON object per line, so tools can filter and average fields."""
import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {"level": record.levelname, "event": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(out)


def get_logger(name="energy_api"):
    log = logging.getLogger(name)
    if not log.handlers:  # set up once, even if this is called many times
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log
