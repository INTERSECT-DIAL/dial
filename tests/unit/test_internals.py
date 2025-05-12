from math import e as E_CONSTANT

import numpy as np
import pytest
from bson import ObjectId

from dial_dataclass import (
    DialInputMultiple,
    DialInputPredictions,
    DialInputSingleOtherStrategy,
)
from dial_service import core
from dial_service.serverside_data import (
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from dial_service.service_specific_dataclasses import DialWorkflowCreationParamsService

DUMMY_WORKFLOW_ID = str(ObjectId())
"""This is used so that we can run the tests without connecting to a backend database or skipping validation for the rest of the data."""


######### HELPERS ####################


def single_1D(backend, strategy, strategy_args):

    workflow_state = DialWorkflowCreationParamsService(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        bounds=[[1, 2]],
        kernel='rbf',
        backend=backend,
        preprocess_standardize=True,
        y_is_good=True,
        seed=42,
    )
    params = DialInputSingleOtherStrategy(
        workflow_id=DUMMY_WORKFLOW_ID,
        strategy=strategy,
        strategy_args = strategy_args,
        bounds=[[1, 2]],
        kernel_args={'length_scale': .5, 'length_scale_bounds': "fixed"},
        seed=42,
    )
    return ServersideInputSingle(workflow_state, params)


def single_2D(backend, strategy, strategy_args):
    workflow_state = DialWorkflowCreationParamsService(
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
        bounds=[[-2, 2], [-2, 2]],
        kernel='rbf',
        backend=backend,
        preprocess_standardize=True,
        y_is_good=True,
        seed=42,
    )
    params = DialInputSingleOtherStrategy(
        workflow_id=DUMMY_WORKFLOW_ID,
        strategy=strategy,
        strategy_args = strategy_args,
        bounds=[[-2, 2], [-2, 2]],
        kernel_args={'length_scale': .5, 'length_scale_bounds': "fixed"},
        seed=42,
    )
    return ServersideInputSingle(workflow_state, params)


def single_3D(backend, strategy, strategy_args):
    workflow_state = DialWorkflowCreationParamsService(
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
        bounds=[[-2, 2], [-2, 2], [-2, 2]],
        kernel='rbf',
        backend=backend,
        preprocess_standardize=True,
        y_is_good=True,
        seed=42,
    )
    params = DialInputSingleOtherStrategy(
        workflow_id=DUMMY_WORKFLOW_ID,
        strategy=strategy,
        strategy_args = strategy_args,
        bounds=[[-2, 2], [-2, 2], [-2, 2]],
        kernel_args={'length_scale': .5, 'length_scale_bounds': "fixed"},
        extra_args={'length_per_dimension': True},
        seed=42
    )
    return ServersideInputSingle(workflow_state, params)


def multiple_2D(backend, strategy):
    workflow_state = DialWorkflowCreationParamsService(
        dataset_x=[],
        dataset_y=[],
        y_is_good=False,
        kernel='rbf',
        length_per_dimension=False,
        bounds=[[0, 100], [-1, 1]],
        backend=backend,
        seed=42,
    )
    params = DialInputMultiple(
        workflow_id=DUMMY_WORKFLOW_ID,
        points=10,
        strategy=strategy,
    )
    return ServersideInputMultiple(workflow_state, params)


def prediction_1D(backend):
    workflow_state = DialWorkflowCreationParamsService(
        dataset_x=[[1], [2]],
        dataset_y=[100, 200],
        bounds=[[1, 2]],
        kernel='rbf',
        backend=backend,
        preprocess_standardize=True,
        y_is_good=True,
        seed=42,
    )
    params = DialInputPredictions(
        workflow_id=DUMMY_WORKFLOW_ID,
        points_to_predict=[[1], [1.25], [1.5], [1.75], [2]],
        kernel_args={'length_scale': .5, 'length_scale_bounds': "fixed"}
    )
    return ServersideInputPrediction(workflow_state, params)


####### TESTS ###################


@pytest.mark.parametrize(
    ('backend', 'approx'),
    [
        ('sklearn', 1.842309),
        # ('gpax', 2.0),
    ],
)
def test_EI_1D(backend, approx):
    data = single_1D(backend,
                     strategy='upper_confidence_bound',
                     strategy_args = {'exploit': 1, 'explore': 1})
    assert core.get_next_point(data) == pytest.approx([approx], abs=0.00001)


@pytest.mark.parametrize(
    ('backend', 'approx'),
    [
        ('sklearn', [1.705352, -1.682829]),
        # ('gpax', [2.0, 2.0]),
    ],
)
def test_EI_2D(backend, approx):
    data = single_2D(backend,
                     strategy='upper_confidence_bound',
                     strategy_args = {'exploit': 1, 'explore': 1})
    assert core.get_next_point(data) == pytest.approx(approx)


@pytest.mark.parametrize(
    ('backend', 'approx'),
    [
        ('sklearn', [2.000000, -1.143727, -1.859496]),
        # ('gpax', [2.0, 2.0, -2.0],  # WAS: [2.0, 2.0, 2.0]
        # ),
    ],
)
def test_EI_3D(backend, approx):
    data = single_3D(backend,
                     strategy='upper_confidence_bound',
                     strategy_args = {'exploit': 1, 'explore': 1})
    assert core.get_next_point(data) == pytest.approx(approx)

@pytest.mark.parametrize(
    ('backend', 'approx'),
    [
        ('sklearn', [2.0]),
        # (
        #     'gpax',
        #     [1.0],  # WAS: [2.0]
        # ),
    ],
)
def test_uncertainty(backend, approx):
    data = single_1D(backend, strategy='uncertainty', strategy_args = None)
    assert core.get_next_point(data) == pytest.approx(approx)


@pytest.mark.parametrize(
    ('backend', 'approx'),
    [
        ('sklearn', [1.037454]),
        # ('gpax', [2.0]),
    ],
)
def test_preprocessing_standardize(backend, approx):
    data = single_1D(backend,
                     strategy='expected_improvement',
                     strategy_args = None)
    data.preprocess_standardize = True
    assert data.Y_best == 1
    assert data.Y_train == pytest.approx([-1, 1])
    assert core.get_next_point(data) == pytest.approx(approx)


@pytest.mark.parametrize(
    ('backend'),
    [
        ('sklearn'),
        ('gpax'),
    ],
)
def test_random(backend):
    data = single_1D(backend,
                     strategy='random',
                     strategy_args = None)
    for _ in range(100):
        output = core.get_next_point(data)
        assert len(output) == 1
        assert 1 <= output[0] <= 2


@pytest.mark.parametrize(
    ('backend'),
    [
        ('sklearn'),
        ('gpax'),
    ],
)
def test_random_points(backend):
    data = multiple_2D(backend, strategy='random')
    for pt in core.get_next_points(data):
        assert 0 <= pt[0] <= 100
        assert -1 <= pt[1] <= 1


@pytest.mark.parametrize(
    ('backend'),
    [
        ('sklearn'),
        # ('gpax'),
    ],
)
def test_hypercube(backend):
    data = multiple_2D(backend, strategy='hypercube')
    points = core.get_next_points(data)
    for i in range(10):
        assert sum(1 for pt in points if 10 * i <= pt[0] <= 10 * (i + 1)) == 1
        assert sum(1 for pt in points if -1 + 0.2 * i <= pt[1] <= -1 + 0.2 * (i + 1)) == 1, (
            f'Need exactly one in [{-1 + 0.2 * i}, {-1 + 0.2 * (i + 1)}]'
        )


@pytest.mark.parametrize(
    ('backend', 'expected_means', 'expected_stddevs', 'expected_raw_stddevs'),
    [
        (
            'sklearn',
            [
            100.00000001,
            116.68792615,
            148.85898558,
            181.20361927,
            200.
            ],
            [4.71404540e-04,
             1.99052432e+01,
             2.79660797e+01,
             1.99052432e+01,
             3.33333717e-04],
            [1.0000000413701844e-05,
             0.4222539741268675,
             0.5932501381196003,
             0.42225397408625437,
             7.071075954854466e-06],
        ),
        # (
            # 'gpax',
            # [
            #     76.99987768175089,
            #     79.70037093884447,
            #     81.5157895856116,
            #     82.38145329572436,
            #     82.26569221517353,
            # ],
            # [3335.7290084812175, 3327.202331393974, 3335.7290084812175],
            # [3335.7290084812175, 3327.202331393974, 3335.7290084812175],
        # ),
    ],
)
def test_surrogate(backend, expected_means, expected_stddevs, expected_raw_stddevs):
    data = prediction_1D(backend)
    means, stddevs, raw_stddevs = core.get_surrogate_values(data)
    assert means == pytest.approx(expected_means)
    assert stddevs[1:4] == pytest.approx(expected_stddevs)
    assert raw_stddevs[1:4] == pytest.approx(expected_raw_stddevs)


# @pytest.mark.parametrize(
#     ('backend'),
#     [
#         ('sklearn'),
#         ('gpax'),
#     ],
# )
# def test_inverse_transform(backend):
#     data = prediction_1D(backend)
#     assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx([-1, 0, 1])
#     assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, 0, 1])

#     data.preprocess_log = True
#     assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx(
#         [1 / E_CONSTANT, 1, E_CONSTANT]
#     )
#     assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, -1, -1])

#     data.preprocess_log = False
#     data.preprocess_standardize = True
#     assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx([100, 150, 200])
#     assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx(
#         [-50, 0, 50]
#     )  # technically improper, as uncertainties can't be negative

#     data.preprocess_log = True
#     assert data.inverse_transform(np.array([-1, 0, 1])) == pytest.approx(
#         [100, 141.42135623730945, 200]
#     )  # TODO
#     assert data.inverse_transform(np.array([-1, 0, 1]), True) == pytest.approx([-1, -1, -1])
