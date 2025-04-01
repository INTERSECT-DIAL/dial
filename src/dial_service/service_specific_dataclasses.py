from typing import Annotated, Literal

from pydantic import Field

from dial_dataclass.dial_dataclass import _POSSIBLE_BACKENDS, _DialWorkflowCreationParams

from .logger import logger


def _get_permitted_backends() -> tuple[str, ...]:
    """private method which should only be executed ONCE per application

    This manually checks imports for the available backends, and generates a type definition based off of the available imports.
    """
    import importlib.util

    available_backends = [
        bkend for bkend in _POSSIBLE_BACKENDS if importlib.util.find_spec(bkend) is not None
    ]
    if not available_backends:
        # TODO - provide explicit installation instructions in this message
        possible_backends = '","'.join(_POSSIBLE_BACKENDS)
        msg = f'No backends were configured, please install at least one backend ("{possible_backends}")'
        raise RuntimeError(msg)

    logger.info('Available backends: %s', available_backends)
    return tuple(available_backends)


AVAILABLE_DIAL_BACKENDS = _get_permitted_backends()
BackendType = Literal[AVAILABLE_DIAL_BACKENDS]
"""'sklearn' or 'gpax' depending on installed dependencies. This is dynamically generated at runtime, so type hints may not be complete."""


# this class is specific to the microservice
class DialWorkflowCreationParamsService(_DialWorkflowCreationParams):
    """This comprises the information needed to create a DIAL workflow."""

    backend: Annotated[
        BackendType,
        Field(description='Backend implementations supported by this instance of DIAL.'),
    ]
    """'sklearn' or 'gpax' depending on installed dependencies. This is dynamically generated at runtime, so type hints may not be complete."""
