import numpy as np
from pytest import approx, fixture

from boalaas_dataclass import BOALaaSInputSingle, BOALaaSInputMultiple, BOALaaSInputPredictions
from boalaas_service import BOALaaSCapabilityImplementation as Service
from boalaas_service.serverside_data import ServersideInputSingle, ServersideInputPrediction

from math import e as E_CONSTANT

#Test data:
@fixture
def single_1D_A(): #alpha because I may add more
    return BOALaaSInputSingle(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[1,2]],
        strategy="expected_improvement"
    )

@fixture
def single_2D_A():
    return BOALaaSInputSingle(
        dataset_x=[[0.9317758694133622, -0.23597335497782845], [-0.7569874398003542, -0.76891211613756], [-0.38457336507729645, -1.1327391183311766], [-0.9293590899359039, 0.25039725076881014], [1.984696498789749, -1.7147926093003538], [1.2001856430453541, 1.572387611848939], [0.5080666898409634, -1.566722183270571], [-1.871124738716507, 1.9022651997285078], [-1.572941300813352, 1.0014173171150125], [0.033053333077524005, 0.44682040004191537]],
        dataset_y=[2.08609604, 2.26284928, 2.21989834, 1.61634392, 3.50481457, 0.25065034, 2.52277171, 2.42139515, 2.34930184, 1.31811177],
        y_is_good=False,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[-2,2], [-2,2]],
        strategy="expected_improvement"
    )

@fixture
def multiple_2D_A():
    return BOALaaSInputMultiple(
        dataset_x=[],
        dataset_y=[],
        y_is_good=False,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[0,100], [-1,1]],
        points="10",
        strategy="random"
    )

@fixture
def prediction_1D_A():
    return BOALaaSInputPredictions(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[1,2]],
        points_to_predict=[[1], [1.25], [1.5], [1.75], [2]],
    )

def test_n_grid(single_1D_A):
    data = ServersideInputSingle(single_1D_A)
    assert (Service()._create_n_dim_grid(data, 11)
            == approx(np.array([[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0]])))

def test_EI_1D(single_1D_A):
    assert Service().get_next_point(single_1D_A) == approx([1.85742], abs=.00001)

def test_EI_2D(single_2D_A):
    assert Service().get_next_point(single_2D_A) == approx([2., 2.])

def test_uncertainty(single_1D_A):
    single_1D_A = single_1D_A.model_copy(update={"strategy": "uncertainty"})
    assert Service().get_next_point(single_1D_A) == approx([1.5])

def test_preprocessing_standardize(single_1D_A):
    single_1D_A = single_1D_A.model_copy(update={"preprocess_standardize": True})
    data = ServersideInputSingle(single_1D_A)
    assert data.Y_best == 1
    assert data.Y_train == approx([-1, 1])
    assert Service().get_next_point(single_1D_A) == approx([1.96832802])

def test_random(single_1D_A):
    single_1D_A = single_1D_A.model_copy(update={"strategy": "random"})
    for _ in range(100):
        output = Service().get_next_point(single_1D_A)
        assert len(output)==1
        assert 1 <= output[0] <= 2

def test_random_points(multiple_2D_A):
    for pt in Service().get_next_points(multiple_2D_A):
        assert 0 <= pt[0] <= 100
        assert -1 <= pt[1] <= 1

def test_hypercube(multiple_2D_A):
    multiple_2D_A = multiple_2D_A.model_copy(update={"strategy": "hypercube"})
    points = Service().get_next_points(multiple_2D_A)
    for i in range(10):
        assert 1 == sum(1 for pt in points if 10*i <= pt[0] <= 10*(i+1))
        assert 1 == sum(1 for pt in points if -1+.2*i <= pt[1] <= -1+.2*(i+1)), f"Need exactly one in [{-1+.2*i}, {-1+.2*(i+1)}]"

def test_surrogate(prediction_1D_A):
    means, stddevs, raw_stddevs = Service().get_surrogate_values(prediction_1D_A)
    assert means == approx([100., 135.4253956249114, 168.17893262605446, 191.52029424913434, 200.])
    assert stddevs[1:4] == approx([8.739217244027204, 11.956904892760956, 8.739217244027222])
    assert raw_stddevs[1:4] == approx([8.739217244027204, 11.956904892760956, 8.739217244027222])

def test_inverse_transform(prediction_1D_A):
    data = ServersideInputPrediction(prediction_1D_A)
    assert data.inverse_transform(np.array([-1, 0, 1])) == approx([-1, 0, 1])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == approx([-1, 0, 1])
    data = ServersideInputPrediction(prediction_1D_A.model_copy(update={"preprocess_log": True}))
    assert data.inverse_transform(np.array([-1, 0, 1])) == approx([1/E_CONSTANT, 1, E_CONSTANT])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == approx([-1, -1, -1])
    data = ServersideInputPrediction(prediction_1D_A.model_copy(update={"preprocess_standardize": True}))
    assert data.inverse_transform(np.array([-1, 0, 1])) == approx([100, 150, 200])
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == approx([-50, 0, 50]) #technically improper, as uncertainties can't be negative
    data = ServersideInputPrediction(prediction_1D_A.model_copy(update={"preprocess_standardize": True, "preprocess_log": True}))
    assert data.inverse_transform(np.array([-1, 0, 1])) == approx([100, 141.42135623730945, 200]) #TODO
    assert data.inverse_transform(np.array([-1, 0, 1]), True) == approx([-1, -1, -1])
