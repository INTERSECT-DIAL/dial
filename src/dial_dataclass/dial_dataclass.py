from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

from .pydantic_helpers import ValidatedObjectId

PositiveIntType = Annotated[int, Field(ge=0)]

_POSSIBLE_BACKENDS = ('sklearn', 'gpax')


class _DialWorkflowCreationParams(BaseModel):
    """This comprises the information needed to create a DIAL workflow.

    This is a base class which should not be directly imported, clients should use "DialWorkflowCreationParamsClient" (in this file) and services should use "DialWorkflowCreationParamsService" (exported from the service)
    """

    dataset_x: Annotated[
        list[
            Annotated[
                list[float],
                Field(description='Field lengths of all subarrays should be equal', min_length=1),
            ]
        ],
        Field(description='The input vectors of the training data'),
    ]
    dataset_y: Annotated[
        list[float],
        Field(
            description='The output values of the training data. Length should equal dataset_x',
        ),
    ]
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description='If true, treat higher y values as better (e.g. y represents yield or profit).  If false, opposite (e.g. y represents error or waste)'
        ),
    ]
    kernel: Literal['rbf', 'matern']
    bounds: list[
        Annotated[
            Annotated[list[float], Field(min_length=2, max_length=2)],
            Field(min_length=2, max_length=2),
        ]
    ]
    seed: Annotated[
        int,
        Field(
            default=-1,
            ge=-1,
            le=4294967295,
            description='Specific RNG seed - use -1 to use system default',
        ),
    ]

    preprocess_log: bool = Field(default=False)
    preprocess_standardize: bool = Field(default=False)

    @field_validator('dataset_x')
    @classmethod
    def ensure_consistent_dataset_x_lengths(cls, x):
        if len(x) < 2:
            return x
        target_length = len(x[0])
        for row in x[1:]:
            if len(row) != target_length:
                msg = 'Unequal vector lengths in dataset_x'
                raise ValueError(msg)
        return x

    # order rows as [low, high] - do NOT error out here, we can efficiently handle normalization
    @field_validator('bounds')
    @classmethod
    def order_bounds(cls, bounds: list[list[float]]):
        for row in bounds:
            row.sort()
        return bounds


# this class is specific to clients; they have no way of knowing which backends the Service supports, so we allow all of them
class DialWorkflowCreationParamsClient(_DialWorkflowCreationParams):
    """Dataclass which clients can use to help verify requests to the DIAL microservice."""

    backend: Literal[_POSSIBLE_BACKENDS]


class DialWorkflowDatasetUpdate(BaseModel):
    workflow_id: ValidatedObjectId
    next_x: list[float] = Field(
        description='The next collection of X values you want to append to your overall data',
        min_length=1,
    )
    """the next collection of X values you want to append"""
    next_y: float = Field(description='The next Y value you want to append to your overall data')
    """the next Y value you want to append"""

class DialInputSingleConfidenceBound(BaseModel):
    workflow_id: ValidatedObjectId
    strategy: Literal['confidence_bound']
    strategy_args: dict[str, Union[float, int, bool]] | None = Field(default=None)
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description='If true, treat higher y values as better (e.g. y represents yield or profit).  If false, opposite (e.g. y represents error or waste)'
        ),
    ]
    backend_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    bounds: list[
        Annotated[
            Annotated[list[float], Field(min_length=2, max_length=2)],
            Field(min_length=2, max_length=2),
        ]
    ]
    extra_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    optimization_points: PositiveIntType = Field(default=1000)
    confidence_bound: float = Field(gt=0.5, lt=1)
    discrete_measurements: bool = Field(default=False)
    discrete_measurement_grid_size: list[PositiveIntType] = Field(default=[20, 20])


class DialInputSingleOtherStrategy(BaseModel):
    workflow_id: ValidatedObjectId
    strategy: Literal['random', 'uncertainty', 'expected_improvement', 'upper_confidence_bound']
    strategy_args: dict[str, Union[float, int, bool]] | None = Field(default=None)
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description='If true, treat higher y values as better (e.g. y represents yield or profit).  If false, opposite (e.g. y represents error or waste)'
        ),
    ]
    kernel_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    backend_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    bounds: list[
        Annotated[
            Annotated[list[float], Field(min_length=2, max_length=2)],
            Field(min_length=2, max_length=2),
        ]
    ]
    seed: Annotated[
        int,
        Field(
            default=-1,
            ge=-1,
            le=4294967295,
            description='Specific RNG seed - use -1 to use system default',
        ),
    ]
    extra_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    optimization_points: PositiveIntType = Field(default=1000)
    discrete_measurements: bool = Field(default=False)
    discrete_measurement_grid_size: list[PositiveIntType] = Field(default=[20, 20])


DialInputSingle = Annotated[
    DialInputSingleConfidenceBound | DialInputSingleOtherStrategy,
    Field(
        discriminator='strategy',
        description='This is the input dataclass for Dial for selecting a single new point to measure.',
    ),
]


class DialInputMultiple(BaseModel):
    """This is the input dataclass for Dial for selecting a multiple new point to measures (i.e., a batch of measurements)."""

    workflow_id: ValidatedObjectId
    points: int
    strategy: Literal['random', 'hypercube']


class DialInputPredictions(BaseModel):
    """This is the input dataclass for Dial for requesting a surrogate evaluation at a given number of points."""

    workflow_id: ValidatedObjectId
    points_to_predict: list[list[float]]

    kernel_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    backend_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)
    extra_args: dict[str, Union[float, int, bool, str, list[float], tuple]] | None = Field(default=None)