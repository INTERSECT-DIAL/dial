"""Simple mockup of a Rosenbrock microservice, meant to supplement automated_client"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from intersect_sdk import (
    IntersectBaseCapabilityImplementation,
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
    intersect_message,
)

logger = logging.getLogger(__name__)


@dataclass
class RosenbrockInputs:
    """Inputs necessary for the Rosenbrock function."""

    x: float
    y: float
    a: float = 1.0
    """Defaults to 1.0 if not set by user."""
    b: float = 100.0
    """Defaults to 100.0 if not set by user."""


class RosenbrockCapabilityImplementation(IntersectBaseCapabilityImplementation):
    intersect_sdk_capability_name = 'Rosenbrock'

    @intersect_message
    def rosenbrock(self, r: RosenbrockInputs) -> float:
        """
        Represents simulation error (vs experimental data) as a function of 2 simulation parameters.
        """
        return (r.a - r.x) ** 2 + r.b * (r.y - r.x**2) ** 2

    @intersect_message
    def rosenbrock_bulk(self, inputs: list[RosenbrockInputs]) -> list[float]:
        return [self.rosenbrock(i) for i in inputs]


"""
This launches the service.  Separate file due to module/import structure, plus we possibly want the capability to be a separate unit
"""

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # boilerplate config file setup
    parser = argparse.ArgumentParser(description='Mock rosenbrock service')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get('DIAL_CONFIG_FILE', Path(__file__).parents[1] / 'local-conf.json'),
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open('rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)

    config = IntersectServiceConfig(
        hierarchy=from_config_file['rosenbrock-hierarchy'],
        **from_config_file['intersect'],
    )

    capability = RosenbrockCapabilityImplementation()
    service = IntersectService([capability], config)

    default_intersect_lifecycle_loop(
        service,
    )
