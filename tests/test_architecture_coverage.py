import torch.nn as nn

from engibona.architecture_checks import low_bit_coverage_report
from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules


class SmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 128)
        self.proj = nn.Linear(128, 128, bias=False)
        self.head = nn.Linear(128, 32, bias=False)
        self.head.weight = self.embed.weight
        self.norm = nn.LayerNorm(128)


def test_coverage_report_detects_complete_matrix_replacement() -> None:
    model = SmallModel()
    replace_linear_modules(
        model,
        EngibonaConfig(mode=QuantMode.BINARY),
        preserve_tied_weights=True,
    )
    report = low_bit_coverage_report(model)
    assert report.complete
    assert report.embeddings == 1
    assert report.tied_heads == 1
    assert report.linears == 1


def test_coverage_report_detects_unconverted_dense_matrix() -> None:
    model = SmallModel()
    report = low_bit_coverage_report(model)
    assert not report.complete
    assert set(report.unsupported_dense_matrices) == {"embed", "head", "proj"}
