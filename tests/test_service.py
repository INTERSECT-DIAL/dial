import numpy as np
from pytest import approx, fixture

from src.neeter_active_learning.data_class import ActiveLearningInputData
from src.neeter_active_learning.active_learning_service import ActiveLearningServiceCapabilityImplementation as Service

@fixture
def one_D_alpha(): #alpha because I may add more
    return ActiveLearningInputData(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        y_is_good=True,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[1,2]]
    )

@fixture
def two_D_alpha():
    return ActiveLearningInputData(
        dataset_x=[[0.9317758694133622, -0.23597335497782845], [-0.7569874398003542, -0.76891211613756], [-0.38457336507729645, -1.1327391183311766], [-0.9293590899359039, 0.25039725076881014], [1.984696498789749, -1.7147926093003538], [1.2001856430453541, 1.572387611848939], [0.5080666898409634, -1.566722183270571], [-1.871124738716507, 1.9022651997285078], [-1.572941300813352, 1.0014173171150125], [0.033053333077524005, 0.44682040004191537]],
        dataset_y=[2.08609604, 2.26284928, 2.21989834, 1.61634392, 3.50481457, 0.25065034, 2.52277171, 2.42139515, 2.34930184, 1.31811177],
        y_is_good=False,
        kernel="rbf",
        length_per_dimension=False,
        bounds=[[-2,2], [-2,2]]
    )

def test_n_grid(one_D_alpha):
    assert (Service()._create_n_dim_grid(one_D_alpha, 11)
            == approx(np.array([[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0]])))

def test_EI_1D(one_D_alpha):
    assert Service().next_point_by_EI(one_D_alpha) == approx([1.85742], abs=.00001)

def test_EI_2D(two_D_alpha):
    assert Service().next_point_by_EI(two_D_alpha) == approx([2., 2.])

def test_uncertainty(one_D_alpha):
    assert Service().next_point_by_uncertainty(one_D_alpha) == approx([1.5])
