import pytest
import torch

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules
from engibona.modules_tied import TiedGroupQuantizedLMHead


transformers = pytest.importorskip("transformers")


def _build_official_text_model():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    rope = {
        "rope_type": "default",
        "rope_theta": 5_000_000.0,
        "mrope_section": [6, 5, 5],
        "mrope_interleaved": True,
    }
    kwargs = dict(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        rms_norm_eps=1.0e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        use_cache=False,
    )
    try:
        config = Qwen3VLTextConfig(
            **kwargs, rope_scaling=rope, rope_theta=5_000_000.0
        )
    except TypeError:
        config = Qwen3VLTextConfig(**kwargs, rope_parameters=rope)
    config._attn_implementation = "sdpa"
    text = Qwen3VLTextModel(config)
    head = torch.nn.Linear(128, 128, bias=False)
    head.weight = text.embed_tokens.weight
    return torch.nn.ModuleDict({"text": text, "lm_head": head})


def test_official_qwen3vl_text_modules_quantize_and_preserve_tie() -> None:
    model = _build_official_text_model()
    replaced = replace_linear_modules(
        model,
        EngibonaConfig(mode=QuantMode.BINARY),
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    assert isinstance(model["lm_head"], TiedGroupQuantizedLMHead)
    assert model["lm_head"].embedding is model["text"].embed_tokens
    assert any(name.endswith("q_proj") for name in replaced)
    assert any(name.endswith("gate_proj") for name in replaced)

    input_ids = torch.randint(0, 128, (2, 8))
    hidden = model["text"](
        input_ids=input_ids, use_cache=False, return_dict=True
    ).last_hidden_state
    logits = model["lm_head"](hidden)
    assert logits.shape == (2, 8, 128)
