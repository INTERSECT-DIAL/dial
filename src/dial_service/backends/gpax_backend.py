import gpax

_KERNELS_GPAX = {'rbf': 'RBF', 'matern': 'matern'}


def train_model(data):
    rng_key_train, rng_key_predict = gpax.utils.get_keys(
        seed=data.seed if data.seed != -1 else None
    )
    gp_model = gpax.viGP(len(data.bounds), get_kernel(data), guide='delta')
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


def predict(model, x, data):
    rng_key_train, rng_key_predict = gpax.utils.get_keys(
        seed=data.seed if data.seed != -1 else None
    )
    mean, y_var = model.predict(rng_key_predict, x.reshape(1, -1))
    return mean[0], data.stddev * y_var[0]


def get_kernel(data):
    kernel_name = data.kernel.lower()
    if kernel_name not in _KERNELS_GPAX:
        raise ValueError(f'Unknown kernel {kernel_name}')
    return _KERNELS_GPAX[kernel_name]
