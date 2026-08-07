import pytest

from engibona.config import EngibonaConfig, QuantMode


def test_evidence_selected_defaults() -> None:
    binary = EngibonaConfig(mode=QuantMode.BINARY)
    ternary = EngibonaConfig(mode=QuantMode.TERNARY)
    assert binary.relaxation == "auto"
    assert binary.export_strategy == "trained"
    assert binary.binary_embedding_strategy == "frozen_ptq"
    assert ternary.ternary_embedding_strategy == "sign_locked_recovery"
    assert ternary.hard_recovery_start == 0.50
    assert ternary.kd_weight == 1.0
    assert ternary.ce_weight == 0.0
    assert ternary.hidden_mse_weight == 0.0
    binary.validate()
    ternary.validate()


def test_release_matched_and_behavior_profiles_are_explicit() -> None:
    release_binary = EngibonaConfig.release_matched(QuantMode.BINARY)
    release_ternary = EngibonaConfig.release_matched(QuantMode.TERNARY)
    behavior_binary = EngibonaConfig.behavior_maximizing(QuantMode.BINARY)
    behavior_ternary = EngibonaConfig.behavior_maximizing(QuantMode.TERNARY)
    assert release_binary.binary_embedding_strategy == "frozen_ptq"
    assert release_ternary.ternary_embedding_strategy == "sign_locked_recovery"
    assert behavior_binary.binary_embedding_strategy == "train"
    assert behavior_ternary.ternary_embedding_strategy == "train"
    for config in (
        release_binary,
        release_ternary,
        behavior_binary,
        behavior_ternary,
    ):
        config.validate()


def test_profile_overrides_are_respected() -> None:
    config = EngibonaConfig.release_matched(
        QuantMode.BINARY,
        binary_embedding_strategy="train",
        kd_temperature=1.5,
    )
    assert config.binary_embedding_strategy == "train"
    assert config.kd_temperature == 1.5


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
