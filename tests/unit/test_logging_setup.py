import json
import logging

import pytest
from loguru import logger

from vllm_nano.utils.logging_setup import (
    LOG_FILE_NAME,
    redact,
    request_context,
    setup_logging,
)

TOKEN = "A" * 24


@pytest.fixture
def log_lines(tmp_path):
    """Configure file-only logging and return a reader for the JSON records."""
    setup_logging(level="DEBUG", log_dir=tmp_path, console=False)

    def read() -> list[dict]:
        logger.complete()  # flush the enqueue=True background writer
        lines = (tmp_path / LOG_FILE_NAME).read_text().splitlines()
        return [json.loads(line)["record"] for line in lines]

    yield read
    logger.remove()


def test_redact_masks_bearer_and_hf_tokens():
    assert redact(f"Authorization: Bearer {TOKEN}") == "Authorization: Bearer <redacted>"
    assert redact("key hf_" + "b" * 30) == "key <redacted>"
    assert redact("nothing secret here") == "nothing secret here"


def test_request_id_attached_and_restored(log_lines):
    with request_context("abc-123"):
        logger.info("inside")
    logger.info("outside")

    inside, outside = log_lines()
    assert inside["extra"]["request_id"] == "abc-123"
    assert outside["extra"]["request_id"] == "-"


def test_secrets_scrubbed_in_message_and_extra(log_lines):
    logger.bind(header=f"Bearer {TOKEN}").info(f"sent Bearer {TOKEN}")

    (record,) = log_lines()
    assert TOKEN not in record["message"]
    assert TOKEN not in record["extra"]["header"]


def test_stdlib_logging_is_bridged(log_lines):
    logging.getLogger("some.third.party").warning("from stdlib")

    (record,) = log_lines()
    assert record["message"] == "from stdlib"
    assert record["level"]["name"] == "WARNING"
    assert record["file"]["name"] == "test_logging_setup.py"


def test_setup_is_idempotent(log_lines, tmp_path):
    setup_logging(level="DEBUG", log_dir=tmp_path, console=False)
    logger.info("once")

    assert len(log_lines()) == 1
