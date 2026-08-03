import torch.nn as nn

from engibona.architecture_checks import low_bit_coverage_report


def test_norm_parameters_are_not_reported_as_dense_matrices() -> None:
    model = nn.Sequential(nn.RMSNorm(128))
    report = low_bit_coverage_report(model)
    assert report.complete
    assert report.unsupported_dense_matrices == ()
