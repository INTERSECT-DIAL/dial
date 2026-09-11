import importlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..serverside_data import ServersideInputBase, ServersideInputPrediction

# tuple of module and class name
_BACKENDS = {
    'gpax': ('dial_service.backends.gpax_backend', 'GpaxBackend'),
    'sable': ('dial_service.backends.sable_backend', 'SABLEBackend'),
    'sklearn': ('dial_service.backends.sklearn_backend', 'SklearnBackend'),
}

_MODEL = TypeVar('_MODEL')
_KERNEL = TypeVar('_KERNEL')
_PREDICT = TypeVar('_PREDICT')


class AbstractBackend(ABC, Generic[_MODEL, _KERNEL, _PREDICT]):
    """Template for creating a usable backend. Note that all functions in the subclass should be static methods.

    GENERICS:
    - first generic = MODEL type
    - second generic = KERNEL type
    - third generic = return type of calling predict()
    """

    @staticmethod
    @abstractmethod
    def train_model(data: ServersideInputBase) -> _MODEL: ...

    @staticmethod
    @abstractmethod
    def initialize_model(data: ServersideInputBase) -> _MODEL: ...

    @staticmethod
    @abstractmethod
    def update_model(model: _MODEL, data: ServersideInputBase) -> _MODEL:
        """A default implementation of update_model could simply train a new model (like train_model).

        Backends can also implement this method to use the provided model and update it
        with new data (re-training, continual learning), which may be more efficient.
        The method returns either a new model, or the provided model with updated state.
        """

    @staticmethod
    @abstractmethod
    def predict(
        model: _MODEL, data: ServersideInputPrediction | ServersideInputBase
    ) -> _PREDICT: ...

    @staticmethod
    @abstractmethod
    def get_kernel(data: ServersideInputBase) -> _KERNEL: ...

    @staticmethod
    @abstractmethod
    def sample(module, model: _MODEL, data: ServersideInputBase): ...

    @staticmethod
    @abstractmethod
    def samples(module, model: _MODEL, data: ServersideInputBase): ...


def get_backend_module(backend: str) -> AbstractBackend:
    """Get the backend module; to be more precise, get the class which follows the AbstractBackend interface.

    We must dynamically import the backend module and its class.

    When using INTERSECT, the "backend" param will have already been validated from the dial_dataclass object.

    If not using INTERSECT, this value is not necessarily validated and may throw an ImportError later on.
    """

    if backend not in _BACKENDS:
        msg = f'Unknown backend {backend}'
        raise ValueError(msg)
    module_name, backend_name = _BACKENDS[backend]

    module = importlib.import_module(module_name)
    return getattr(module, backend_name)
