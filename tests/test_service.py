import numpy as np
from pytest import approx, fixture

from neeter_active_learning.data_class import BoalasInputSingle, BoalasInputMultiple
from neeter_active_learning.active_learning_service import BoalasCapabilityImplementation as Service
from neeter_active_learning.serverside_data import ServersideInputSingle

#Test data:
@fixture
def single_1D_A(): #alpha because I may add more
    return BoalasInputSingle(
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
    return BoalasInputSingle(
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
    return BoalasInputMultiple(
        dataset_x=[],
        dataset_y=[],
        y_is_good=False,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[0,100], [-1,1]],
        points="10",
        strategy="random"
    )

def test_n_grid(single_1D_A):
    data = ServersideInputSingle(single_1D_A)
    assert (Service()._create_n_dim_grid(data, 11)
            == approx(np.array([[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0]])))

def test_EI_1D(single_1D_A):
    assert Service().next_point(single_1D_A) == approx([1.85742], abs=.00001)

def test_EI_2D(single_2D_A):
    assert Service().next_point(single_2D_A) == approx([2., 2.])

def test_uncertainty(single_1D_A):
    single_1D_A = single_1D_A.model_copy(update={"strategy": "uncertainty"})
    assert Service().next_point(single_1D_A) == approx([1.5])

def test_preprocessing_standardize(single_1D_A):
    single_1D_A = single_1D_A.model_copy(update={"preprocess_standardize": True})
    data = ServersideInputSingle(single_1D_A)
    assert data.Y_best == 1
    assert data.Y_train == approx([-1, 1])
    assert Service().next_point(single_1D_A) == approx([1.96832802])

def test_random_points(multiple_2D_A):
    for pt in Service().next_points(multiple_2D_A):
        assert 0 <= pt[0] <= 100
        assert -1 <= pt[1] <= 1

def test_hypercube(multiple_2D_A):
    multiple_2D_A = multiple_2D_A.model_copy(update={"strategy": "hypercube"})
    points = Service().next_points(multiple_2D_A)
    for i in range(10):
        assert 1 == sum(1 for pt in points if 10*i <= pt[0] <= 10*(i+1))
        assert 1 == sum(1 for pt in points if -1+.2*i <= pt[1] <= -1+.2*(i+1)), f"Need exactly one in [{-1+.2*i}, {-1+.2*(i+1)}]"
