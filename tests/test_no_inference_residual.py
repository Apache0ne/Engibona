import tempfile

import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.export import export_packed
from engibona.modules import GroupQuantizedLinear


def test_export_payload_contains_only_packed_codes_scales_and_metadata() -> None:
    config = EngibonaConfig(
        mode=QuantMode.BINARY,
        export_strategy="trained",
    )
    model = nn.Sequential(
        GroupQuantizedLinear(nn.Linear(128, 2, bias=False), config)
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        item = export_packed(model, handle.name, config)["tensors"]["0"]
    assert set(item) == {
        "shape",
        "encoding",
        "padding_symbols",
        "packed_codes",
        "scales_fp16",
    }
    assert item["packed_codes"].dtype == torch.uint8
    assert item["scales_fp16"].dtype == torch.float16
