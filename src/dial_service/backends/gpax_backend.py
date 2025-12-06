"""NOTE: This file should not be imported in application code except dynamically via the get_backend_module function in __init__.py ."""

import gpax
import jax.numpy as jnp

from . import AbstractBackend

_KERNELS_GPAX = {'rbf': 'RBF', 'matern': 'matern'}

gpax.utils.enable_x64()


class GpaxBackend(AbstractBackend[gpax.viGP, str, tuple[jnp.ndarray, jnp.ndarray]]):
    @staticmethod
    def train_model(data):
        """Generate a trained model."""
        rng_key_train, rng_key_predict = gpax.utils.get_keys(
            seed=data.seed if data.seed != -1 else None
        )
        gp_model = gpax.viGP(len(data.bounds), GpaxBackend.get_kernel(data), guide='delta')
        gp_model.fit(
            rng_key_train,
            data.X_train,
            data.Y_train,
            num_steps=250,
            step_size=0.05,
            print_summary=False,
            progress_bar=False,
        )
        return gp_model

    @staticmethod
    def initialize_model(data):
        """Generate an untrained model."""
        rng_key_train, rng_key_predict = gpax.utils.get_keys(
            seed=data.seed if data.seed != -1 else None
        )
        return gpax.viGP(len(data.bounds), GpaxBackend.get_kernel(data), guide='delta')

    @staticmethod
    def predict(model, data):
        rng_key_train, rng_key_predict = gpax.utils.get_keys(
            seed=data.seed if data.seed != -1 else None
        )
        x = data.x_predict
        # mean, y_var = model.predict(rng_key_predict, data.x_predict)
        # TODO check why model.predict().reshape() fails
        mean, y_var = model.predict(rng_key_predict, x.reshape(1, -1))
        return mean[0], data.stddev * y_var[0]

    @staticmethod
    def get_kernel(data):
        kernel_name = data.kernel.lower()
        if kernel_name not in _KERNELS_GPAX:
            msg = f'Unknown kernel {kernel_name}'
            raise ValueError(msg)
        return _KERNELS_GPAX[kernel_name]

    @staticmethod
    def sample(module, model, data):
        raise NotImplementedError
