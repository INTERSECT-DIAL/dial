from copy import deepcopy

import pytest
from dial_service.service_specific_dataclasses import AVAILABLE_DIAL_BACKENDS


def generate_pytest_parameters(
    old_params: tuple[tuple[str], tuple[list[object]]], backend_idx: int
) -> tuple[tuple[str], tuple[list[object]]]:
    """
    Generates test safeguards from generic parameters.

    This should only really be used if you simultaneously want to use generic parameters in contexts other than Pytest.

    Params:
      - old_params = your normal pytest.mark.parametrize parameters (don't use pytest.param)
      - backend_idx = the index of the "backend" parameter in your args list"""
    new_params = deepcopy(old_params)
    for i, test in enumerate(old_params[1]):
        backend_name = test[backend_idx]
        if backend_name != 'sklearn':
            new_params[1][i] = pytest.param(
                *test,
                marks=pytest.mark.skipif(
                    backend_name not in AVAILABLE_DIAL_BACKENDS,
                    reason=f'{backend_name} not installed',
                ),
            )
    return new_params
