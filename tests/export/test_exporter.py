# tests/export/test_exporter.py
from unittest.mock import MagicMock, patch

import pytest

from hippofloop.export.exporter import GgufExporter


def test_exporter_stores_config():
    exporter = GgufExporter(quantization="Q4_K_M")
    assert exporter.quantization == "Q4_K_M"


def test_exporter_validates_quantization():
    with pytest.raises(ValueError, match="Unsupported quantization"):
        GgufExporter(quantization="INVALID")


def test_supported_quantizations():
    """All documented quantization levels are accepted."""
    for quant in ("Q4_K_M", "Q5_K_M", "Q8_0"):
        exporter = GgufExporter(quantization=quant)
        assert exporter.quantization == quant
