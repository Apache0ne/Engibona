import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import GroupQuantizedLinear


def test_load_hard_binary_codes_and_scales_roundtrip() -> None:
    module = GroupQuantizedLinear(
        nn.Linear(128, 3, bias=False),
        EngibonaConfig(mode=QuantMode.BINARY),
    )
    codes, scales, _ = module.hard_codes_and_scales()
    codes = codes.clone()
    codes[0, :8] *= -1
    scales = scales.clone()
    scales[0, 0] *= 1.1
    module.load_hard_codes_and_scales(codes, scales)
    loaded_codes, loaded_scales, _ = module.hard_codes_and_scales()
    assert torch.equal(loaded_codes, codes)
    assert torch.allclose(loaded_scales.float(), scales.float(), atol=2e-3)
