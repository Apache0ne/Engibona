import pytest

from engibona.config import EngibonaConfig, QuantMode


def test_default_binary_choice_is_hard_and_trained_export() -> None:
    config = EngibonaConfig(mode=QuantMode.BINARY)
    assert config.relaxation == "auto"
    assert config.export_strategy == "trained"
    config.validate()


def test_invalid_export_strategy_rejected() -> None:
    config = EngibonaConfig(export_strategy="unknown")
    with pytest.raises(ValueError):
        config.validate()


def test_group_size_required_positive() -> None:
    config = EngibonaConfig(group_size=0)
    with pytest.raises(ValueError):
        config.validate()
