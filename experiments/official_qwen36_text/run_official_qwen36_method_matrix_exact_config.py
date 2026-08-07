#!/usr/bin/env python3
"""Exact-config entry point for the official Qwen3.6 method matrix."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch.nn as nn
from transformers import AutoConfig, AutoModel


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen36_matrix_original",
    HERE / "run_official_qwen36_method_matrix.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load Qwen3.6 method matrix")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class TinyOfficialQwen36Exact(nn.Module):
    def __init__(
        self,
        layers: int,
        vocab_size: int = 256,
        hidden_size: int = 128,
        intermediate_size: int = 384,
        tied: bool = True,
    ) -> None:
        super().__init__()
        layer_types = [
            "linear_attention" if index % 4 != 3 else "full_attention"
            for index in range(layers)
        ]
        config = AutoConfig.for_model(
            "qwen3_5_text",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            full_attention_interval=4,
            layer_types=layer_types,
            hidden_act="silu",
            max_position_embeddings=1024,
            initializer_range=0.02,
            rms_norm_eps=1.0e-6,
            use_cache=False,
            attention_bias=False,
            use_qk_norm_in_attention=False,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "mrope_section": [3, 3, 2],
                "partial_rotary_factor": 0.25,
            },
            tie_word_embeddings=tied,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        self.config = config
        self.text = AutoModel.from_config(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tied:
            self.lm_head.weight = self.text.embed_tokens.weight

    def forward(self, input_ids, output_hidden_states: bool = False):
        output = self.text(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(output.last_hidden_state)
        if output_hidden_states:
            return logits, output.last_hidden_state, output.hidden_states
        return logits


experiment.TinyOfficialQwen36 = TinyOfficialQwen36Exact


if __name__ == "__main__":
    experiment.main()
