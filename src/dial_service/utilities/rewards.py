from scipy.stats import norm


def get_negative_value_function(data):
    match data.strategy:
        case 'uncertainty':
            return lambda mean, stddev: -stddev
        case 'expected_improvement':
            return lambda mean, stddev: _expected_improvement(mean, stddev, data)
        case 'upper_confidence_bound':
            return lambda mean, stddev: -(mean + 1 * stddev)
        case 'confidence_bound':
            z_value = norm.ppf(0.5 + data.confidence_bound / 2)
            return lambda mean, stddev: _confidence_bound(mean, stddev, z_value, data)
        case _:
            raise ValueError(f'Unknown strategy {data.strategy}')


def _expected_improvement(mean, stddev, data):
    if stddev == 0:
        return 0
    z = (mean - data.Y_best) / stddev * (1 if data.y_is_good else -1)
    return -stddev * (z * norm.cdf(z) + norm.pdf(z))


def _confidence_bound(mean, stddev, z_value, data):
    return -z_value * stddev + mean * (-1 if data.y_is_good else 1)
