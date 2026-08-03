from __future__ import annotations

import torch

from engibona.config import QuantMode
from engibona.metrics import activation_covariance
from engibona.projection import metric_project


def main() -> None:
    torch.manual_seed(7)
    weight = torch.randn(16, 256)
    calibration = torch.randn(512, 256)
    covariance = activation_covariance(calibration, group_size=128)

    result = metric_project(
        weight,
        QuantMode.TERNARY,
        group_size=128,
        metric=covariance,
        refine_steps=16,
    )
    print("initial weighted error:", float(result.initial_error.mean()))
    print("final weighted error:  ", float(result.final_error.mean()))
    print("codes:", sorted(result.codes.unique().tolist()))
    print("scale shape:", tuple(result.scales.shape))


if __name__ == "__main__":
    main()
