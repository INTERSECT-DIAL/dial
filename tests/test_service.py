from math import e as E_CONSTANT

import numpy as np
import pytest
from dial_dataclass import DialInputMultiple, DialInputPredictions, DialInputSingle
from dial_service import DialCapabilityImplementation as Service
from dial_service.serverside_data import ServersideInputPrediction, ServersideInputSingle


# Test data:
@pytest.fixture()
def single_1D_A():  # alpha because I may add more
    return DialInputSingle(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[1, 2]],
        strategy='expected_improvement',
        backend='sklearn',
        seed=42,
    )


@pytest.fixture()
def single_1D_GPax():
    return DialInputSingle(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[1, 2]],
        strategy='expected_improvement',
        backend='gpax',
        seed=42,
    )


@pytest.fixture()
def single_2D_A():
    return DialInputSingle(
        dataset_x=[
            [0.9317758694133622, -0.23597335497782845],
            [-0.7569874398003542, -0.76891211613756],
            [-0.38457336507729645, -1.1327391183311766],
            [-0.9293590899359039, 0.25039725076881014],
            [1.984696498789749, -1.7147926093003538],
            [1.2001856430453541, 1.572387611848939],
            [0.5080666898409634, -1.566722183270571],
            [-1.871124738716507, 1.9022651997285078],
            [-1.572941300813352, 1.0014173171150125],
            [0.033053333077524005, 0.44682040004191537],
        ],
        dataset_y=[
            2.08609604,
            2.26284928,
            2.21989834,
            1.61634392,
            3.50481457,
            0.25065034,
            2.52277171,
            2.42139515,
            2.34930184,
            1.31811177,
        ],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[-2, 2], [-2, 2]],
        strategy='expected_improvement',
        backend='sklearn',
        seed=42,
    )


@pytest.fixture()
def single_2D_GPax():
    return DialInputSingle(
        dataset_x=[
            [0.9317758694133622, -0.23597335497782845],
            [-0.7569874398003542, -0.76891211613756],
            [-0.38457336507729645, -1.1327391183311766],
            [-0.9293590899359039, 0.25039725076881014],
            [1.984696498789749, -1.7147926093003538],
            [1.2001856430453541, 1.572387611848939],
            [0.5080666898409634, -1.566722183270571],
            [-1.871124738716507, 1.9022651997285078],
            [-1.572941300813352, 1.0014173171150125],
            [0.033053333077524005, 0.44682040004191537],
        ],
        dataset_y=[
            2.08609604,
            2.26284928,
            2.21989834,
            1.61634392,
            3.50481457,
            0.25065034,
            2.52277171,
            2.42139515,
            2.34930184,
            1.31811177,
        ],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[-2, 2], [-2, 2]],
        strategy='expected_improvement',
        backend='gpax',
        seed=42,
    )


@pytest.fixture()
def single_3D_A():
    return DialInputSingle(
        dataset_x=[
            [-0.3666976630219634, -0.7643946670537294, -1.1506370018439385],
            [1.4762726361543423, 0.8181375702328815, 1.5299621681784998],
            [-0.7986512557541572, -0.300813391209126, 0.3398475262499705],
            [-1.7533368192621634, 0.5682809815567915, 0.7298166365187178],
            [0.300254731182263, 0.028841529606553618, 1.819824246857562],
            [1.1678007977360392, -1.9715897117057408, 1.194164838762112],
            [0.5019723334777071, 1.3967281950149926, -0.4644564435701244],
            [-1.3708919800265436, -1.5239595463615325, -1.5405347832484588],
            [1.9722151349341495, -1.0312670259072774, -1.6539808151893483],
            [-1.0234541059672568, 1.7103830579558839, -0.310540710518709],
        ],
        dataset_y=[
            386.7765772276362,
            259.6237369279032,
            99.25298200142575,
            652.2798400277729,
            332.6820405390801,
            1846.538928158297,
            714.8214853543617,
            2662.506790954967,
            3165.0730457087557,
            1095.6837658755999,
        ],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[-2, 2], [-2, 2], [-2, 2]],
        strategy='expected_improvement',
        backend='sklearn',
        seed=42,
    )


@pytest.fixture()
def single_3D_GPax():
    return DialInputSingle(
        dataset_x=[
            [-0.3666976630219634, -0.7643946670537294, -1.1506370018439385],
            [1.4762726361543423, 0.8181375702328815, 1.5299621681784998],
            [-0.7986512557541572, -0.300813391209126, 0.3398475262499705],
            [-1.7533368192621634, 0.5682809815567915, 0.7298166365187178],
            [0.300254731182263, 0.028841529606553618, 1.819824246857562],
            [1.1678007977360392, -1.9715897117057408, 1.194164838762112],
            [0.5019723334777071, 1.3967281950149926, -0.4644564435701244],
            [-1.3708919800265436, -1.5239595463615325, -1.5405347832484588],
            [1.9722151349341495, -1.0312670259072774, -1.6539808151893483],
            [-1.0234541059672568, 1.7103830579558839, -0.310540710518709],
        ],
        dataset_y=[
            386.7765772276362,
            259.6237369279032,
            99.25298200142575,
            652.2798400277729,
            332.6820405390801,
            1846.538928158297,
            714.8214853543617,
            2662.506790954967,
            3165.0730457087557,
            1095.6837658755999,
        ],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[-2, 2], [-2, 2], [-2, 2]],
        strategy='expected_improvement',
        backend='gpax',
        seed=42,
    )


@pytest.fixture()
def multiple_2D_A():
    return DialInputMultiple(
        dataset_x=[],
        dataset_y=[],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[0, 100], [-1, 1]],
        points='10',
        strategy='random',
        backend='sklearn',
        seed=42,
    )


@pytest.fixture()
def multiple_2D_GPax():
    return DialInputMultiple(
        dataset_x=[],
        dataset_y=[],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[0, 100], [-1, 1]],
        points='10',
        strategy='random',
        backend='gpax',
        seed=42,
    )


@pytest.fixture()
def prediction_1D_A():
    return DialInputPredictions(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[1, 2]],
        points_to_predict=[[1], [1.25], [1.5], [1.75], [2]],
        backend='sklearn',
        seed=42,
    )


@pytest.fixture()
def prediction_1D_GPax():
    return DialInputPredictions(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[1, 2]],
        points_to_predict=[[1], [1.25], [1.5], [1.75], [2]],
        backend='gpax',
        seed=42,
    )


# random.seed(42)


def test_n_grid(single_1D_A, single_1D_GPax):
    # sklearn
    data = ServersideInputSingle(single_1D_A)
    assert Service()._create_n_dim_grid(data, 11) == pytest.approx(
        np.array([[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0]])
    )
    # gpax
    data = ServersideInputSingle(single_1D_GPax)
    assert Service()._create_n_dim_grid(data, 11) == pytest.approx(
        np.array([[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0]])
    )


def test_EI_1D(single_1D_A, single_1D_GPax):
    assert Service().get_next_point(single_1D_A) == pytest.approx([1.85742], abs=0.00001)
    assert Service().get_next_point(single_1D_GPax) == pytest.approx([2.0], abs=0.00001)


def test_EI_2D(single_2D_A, single_2D_GPax):
    assert Service().get_next_point(single_2D_A) == pytest.approx([2.0, 2.0])
    assert Service().get_next_point(single_2D_GPax) == pytest.approx([2.0, 2.0])


def test_EI_3D(single_3D_A, single_3D_GPax):
    assert Service().get_next_point(single_3D_A) == pytest.approx([-2.0, -2.0, 2.0])
    assert Service().get_next_point(single_3D_GPax) == pytest.approx([2.0, 2.0, 2.0])


def test_uncertainty(single_1D_A, single_1D_GPax):
    # sklearn
    single_1D_A = single_1D_A.model_copy(update={'strategy': 'uncertainty'})
    assert Service().get_next_point(single_1D_A) == pytest.approx([1.5])
    # gpax
    single_1D_GPax = single_1D_GPax.model_copy(update={'strategy': 'uncertainty'})
    assert Service().get_next_point(single_1D_GPax) == pytest.approx([2.0])


def test_preprocessing_standardize(single_1D_A, single_1D_GPax):
    # sklearn
    single_1D_A = single_1D_A.model_copy(update={'preprocess_standardize': True})
    data = ServersideInputSingle(single_1D_A)
    assert data.Y_best == 1
    assert data.Y_train == pytest.approx([-1, 1])
    assert Service().get_next_point(single_1D_A) == pytest.approx([1.96832802])
    # gpax
    single_1D_GPax = single_1D_GPax.model_copy(update={'preprocess_standardize': True})
    data = ServersideInputSingle(single_1D_GPax)
    assert data.Y_best == 1
    assert data.Y_train == pytest.approx([-1, 1])
    assert Service().get_next_point(single_1D_GPax) == pytest.approx([2.0])


def test_random(single_1D_A, single_1D_GPax):
    # sklearn
    single_1D_A = single_1D_A.model_copy(update={'strategy': 'random'})
    for _ in range(100):
        output = Service().get_next_point(single_1D_A)
        assert len(output) == 1
        assert 1 <= output[0] <= 2
    # gpax
    single_1D_GPax = single_1D_GPax.model_copy(update={'strategy': 'random'})
    for _ in range(100):
        output = Service().get_next_point(single_1D_GPax)
        assert len(output) == 1
        assert 1 <= output[0] <= 2


def test_random_points(multiple_2D_A, multiple_2D_GPax):
    # sklearn
    for pt in Service().get_next_points(multiple_2D_A):
        assert 0 <= pt[0] <= 100
        assert -1 <= pt[1] <= 1
    # gpax
    for pt in Service().get_next_points(multiple_2D_GPax):
        assert 0 <= pt[0] <= 100
        assert -1 <= pt[1] <= 1


def test_hypercube(multiple_2D_A, multiple_2D_GPax):
    # sklearn
    multiple_2D_A = multiple_2D_A.model_copy(update={'strategy': 'hypercube'})
    points = Service().get_next_points(multiple_2D_A)
    for i in range(10):
        assert sum(1 for pt in points if 10 * i <= pt[0] <= 10 * (i + 1)) == 1
        assert (
            sum(1 for pt in points if -1 + 0.2 * i <= pt[1] <= -1 + 0.2 * (i + 1)) == 1
        ), f'Need exactly one in [{-1+.2*i}, {-1+.2*(i+1)}]'
    # gpax
    multiple_2D_GPax = multiple_2D_GPax.model_copy(update={'strategy': 'hypercube'})
    points = Service().get_next_points(multiple_2D_GPax)
    for i in range(10):
        assert sum(1 for pt in points if 10 * i <= pt[0] <= 10 * (i + 1)) == 1
        assert (
            sum(1 for pt in points if -1 + 0.2 * i <= pt[1] <= -1 + 0.2 * (i + 1)) == 1
        ), f'Need exactly one in [{-1+.2*i}, {-1+.2*(i+1)}]'


def test_surrogate(prediction_1D_A, prediction_1D_GPax):
    # sklearn
    means, stddevs, raw_stddevs = Service().get_surrogate_values(prediction_1D_A)
    assert means == pytest.approx(
        [100.0, 135.4253956249114, 168.17893262605446, 191.52029424913434, 200.0]
    )
    assert stddevs[1:4] == pytest.approx([8.739217244027204, 11.956904892760956, 8.739217244027222])
    assert raw_stddevs[1:4] == pytest.approx(
        [8.739217244027204, 11.956904892760956, 8.739217244027222]
    )
    # gpax
    means, stddevs, raw_stddevs = Service().get_surrogate_values(prediction_1D_GPax)
    assert means == pytest.approx(
        [
            76.99987768175089,
            79.70037093884447,
            81.5157895856116,
            82.38145329572436,
            82.26569221517353,
        ]
    )
    assert stddevs[1:4] == pytest.approx(
        [3335.7290084812175, 3327.202331393974, 3335.7290084812175]
    )
    assert raw_stddevs[1:4] == pytest.approx(
        [3335.7290084812175, 3327.202331393974, 3335.7290084812175]
    )


def test_inverse_transform(prediction_1D_A, prediction_1D_GPax):
    # sklearn
    data = ServersideInputPrediction(prediction_1D_A)
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx([-1, 0, 1])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, 0, 1])
    data = ServersideInputPrediction(prediction_1D_A.model_copy(update={'preprocess_log': True}))
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx(
        [1 / E_CONSTANT, 1, E_CONSTANT]
    )
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, -1, -1])
    data = ServersideInputPrediction(
        prediction_1D_A.model_copy(update={'preprocess_standardize': True})
    )
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx([100, 150, 200])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx(
        [-50, 0, 50]
    )  # technically improper, as uncertainties can't be negative
    data = ServersideInputPrediction(
        prediction_1D_A.model_copy(update={'preprocess_standardize': True, 'preprocess_log': True})
    )
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx(
        [100, 141.42135623730945, 200]
    )  # TODO
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, -1, -1])
    # gpax
    data = ServersideInputPrediction(prediction_1D_GPax)
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx([-1, 0, 1])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, 0, 1])
    data = ServersideInputPrediction(prediction_1D_GPax.model_copy(update={'preprocess_log': True}))
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx(
        [1 / E_CONSTANT, 1, E_CONSTANT]
    )
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, -1, -1])
    data = ServersideInputPrediction(
        prediction_1D_GPax.model_copy(update={'preprocess_standardize': True})
    )
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx([100, 150, 200])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx(
        [-50, 0, 50]
    )  # technically improper, as uncertainties can't be negative
    data = ServersideInputPrediction(
        prediction_1D_GPax.model_copy(
            update={'preprocess_standardize': True, 'preprocess_log': True}
        )
    )
    assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx(
        [100, 141.42135623730945, 200]
    )  # TODO
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, -1, -1])
