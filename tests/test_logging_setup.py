import sys
sys.path.insert(0, "src")

import json
import logging

from logging_setup import JsonFormatter, get_logger


def make_record(msg, fields=None, exc_info=None):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, exc_info)
    if fields is not None:
        record.fields = fields
    return record


def test_line_is_json_with_level_event_and_fields():
    line = JsonFormatter().format(make_record("forecast", {"building": "b1", "ms": 55.1}))
    assert json.loads(line) == {"level": "INFO", "event": "forecast",
                                "building": "b1", "ms": 55.1}


def test_log_call_without_fields_still_works():
    line = JsonFormatter().format(make_record("startup"))
    assert json.loads(line) == {"level": "INFO", "event": "startup"}


def test_exception_adds_a_one_line_error():
    try:
        1 / 0
    except ZeroDivisionError:
        record = make_record("request_failed", exc_info=sys.exc_info())
    out = json.loads(JsonFormatter().format(record))
    assert out["error"] == "ZeroDivisionError: division by zero"


def test_get_logger_adds_its_handler_only_once():
    get_logger("energy_test_logger")
    log = get_logger("energy_test_logger")
    assert len(log.handlers) == 1


def test_every_request_writes_one_log_line_with_status_and_time():
    import io
    from fastapi.testclient import TestClient
    import api

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    api.log.addHandler(handler)
    try:
        TestClient(api.app).get("/no-such-route")
    finally:
        api.log.removeHandler(handler)
    line = json.loads(buffer.getvalue().splitlines()[-1])
    assert line["event"] == "request"
    assert line["method"] == "GET"
    assert line["path"] == "/no-such-route"
    assert line["status"] == 404
    assert line["ms"] >= 0
