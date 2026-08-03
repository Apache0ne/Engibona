import pytest

from engibona.config import EngibonaConfig, QuantMode


def test_evidence_selected_defaults() -> None:
    binary = EngibonaConfig(mode=QuantMode.BINARY)
    ternary = EngibonaConfig(mode=QuantMode.TERNARY)
    assert binary.relaxation == "auto"
    assert binary.export_strategy == "trained"
    assert ternary.hard_recovery_start == 0.50
    assert ternary.kd_weight == 1.0
    assert ternary.ce_weight == 0.0
    assert ternary.hidden_mse_weight == 0.0
    binary.validate()
    ternary.validate()


def test_invalid_export_strategy_rejected() -> None:
    config = EngibonaConfig(export_strategy="unknown")
    with pytest.raises(ValueError):
        config.validate()


def test_group_size_required_positive() -> None:
    config = EngibonaConfig(group_size=0)
    with pytest.raises(ValueError):
        config.validate()


def test_negative_loss_weight_rejected() -> None:
    config = EngibonaConfig(ce_weight=-1.0)
    with pytest.raises(ValueError):
        config.validate()
