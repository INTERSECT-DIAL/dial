from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from .pydantic_helpers import ValidatedObjectId

PositiveIntType = Annotated[int, Field(ge=0)]

_POSSIBLE_BACKENDS = ('sklearn', 'gpax', 'sable')

BackendType = Literal[_POSSIBLE_BACKENDS]


def validate_dims_and_length(data, x_name, y_name, dim_x=None):
    """validate the lengths of datasets"""

    #print(data)

    dataset_x = data[x_name]
    dataset_y = data[y_name]

    len_x = len(dataset_x)
    len_y = len(dataset_y)

    if len_y != len_x:
        msg = (f'Unequal number of points in {x_name} {len_x=}'
               f' and {y_name} {len_y=}.')
        raise ValueError(msg)

    if dim_x is None and len_x == 0:
        msg = ('Can not infer dim_x from empty dataset.'
               'Set dim_x to the correct dimension.')
        raise ValueError(msg)

    if dim_x is not None and len_x > 0 and len(dataset_x[0]) != dim_x:
        msg = (f'Vectors in {x_name} must be of length {dim_x=}.'
               'Set dim_x to the correct dimension.')
        raise ValueError(msg)


class _DialWorkflowCreationParams(BaseModel):
    """This comprises the information needed to create a DIAL workflow.

    This is a base class which should not be directly imported, clients should use
    "DialWorkflowCreationParamsClient" (in this file) and services should use
    "DialWorkflowCreationParamsService" (exported from the service)
    """

    dataset_x: Annotated[
        list[
            Annotated[
                list[float],
                Field(description='Field lengths of all subarrays should be equal'),
            ]
        ],
        Field(description='The input vectors of the training data'),
    ]
    dataset_y: Annotated[
        list[float],
        Field(
            description=('The output values of the training data.'
                         ' Length should equal dataset_x'),
        ),
    ]
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description=('If true, treat higher y values as better'
                         ' (e.g. y represents yield or profit).'
                         ' If false, opposite (e.g. y represents error or waste)'),
        ),
    ]
    kernel: Literal['rbf', 'matern', 'linear']
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
    dim_x: Annotated[int | None, Field(
        default=None,
        description=('Provide the dimension of x explicitly,'
                     ' e.g. if the initial dataset is empty.'
                     ' If None, it will be inferred from dataset_x if possible.'),
    )]

    preprocess_log: bool = Field(default=False)
    preprocess_standardize: bool = Field(default=False)

    kernel_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """Additional arguments to provide alongside the kernel type."""
    backend_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """Additional arguments to provide alongside the backend type."""
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """Miscellaneous additional arguments."""

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

    @model_validator(mode='after')
    def validate_dims_and_length(self, values):
        validate_dims_and_length(vars(self),
                                 'dataset_x', 'dataset_y', dim_x=self.dim_x)
        return self


# this class is specific to clients; they have no way of knowing which backends the Service supports, so we allow all of them
class DialWorkflowCreationParamsClient(_DialWorkflowCreationParams):
    """Dataclass which clients can use to help verify requests to the DIAL microservice."""

    backend: BackendType


class DialWorkflowDatasetUpdate(BaseModel):
    """This class is used to send a single update to the dataset."""
    workflow_id: ValidatedObjectId
    next_x: list[float] = Field(
        description='The next collection of X values you want to append to your overall data',
        min_length=1,
    )
    """the next collection of X values you want to append"""
    next_y: float = Field(description='The next Y value you want to append to your overall data')
    """the next Y value you want to append"""
    kernel_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """Additional arguments to provide alongside the kernel type. These arguments will OVERRIDE prior saved arguments."""
    backend_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """Additional arguments to provide alongside the backend type. These arguments will OVERRIDE prior saved arguments."""
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """Miscellaneous additional arguments. These arguments will OVERRIDE prior saved arguments."""


class DialWorkflowDatasetUpdates(BaseModel):
    """This class is used to send multiple updates to the dataset."""
    workflow_id: ValidatedObjectId
    next_x_list: list[list[float]] = Field(min_length=1)
    next_y_list: list[float] = Field(min_length=1)
    kernel_args: dict[str, float | int | bool | str | list[float] | tuple] | None = None
    backend_args: dict[str, float | int | bool | str | list[float] | tuple] | None = None
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = None

    @model_validator(mode='after')
    def validate_dims_and_length(self, values):
        validate_dims_and_length(vars(self),
                                 'next_x_list', 'next_y_list')
        return self

class DialInputSingleConfidenceBound(BaseModel):
    workflow_id: ValidatedObjectId
    strategy: Literal['confidence_bound']
    strategy_args: dict[str, float | int | bool] | None = Field(default=None)
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description='If true, treat higher y values as better (e.g. y represents yield or profit).  If false, opposite (e.g. y represents error or waste)',
        ),
    ]
    bounds: list[
        Annotated[
            Annotated[list[float], Field(min_length=2, max_length=2)],
            Field(min_length=2, max_length=2),
        ]
    ]
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """These extra arguments will be MERGED with the saved extra_args, with these arguments taking place over the saved values when applicable."""
    optimization_points: PositiveIntType = Field(default=1000)
    confidence_bound: float = Field(gt=0.5, lt=1)
    discrete_measurements: bool = Field(default=False)
    discrete_measurement_grid_size: list[PositiveIntType] = Field(default=[20, 20])


class DialInputSingleOtherStrategy(BaseModel):
    workflow_id: ValidatedObjectId
    strategy: Literal[
        'random',
        'uncertainty',
        'expected_improvement',
        'upper_confidence_bound',
        'upper_confidence_bound_nomad',
        'polymer_acl_sampler',
    ]
    strategy_args: dict[str, float | int | bool] | None = Field(default=None)
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description='If true, treat higher y values as better (e.g. y represents yield or profit).  If false, opposite (e.g. y represents error or waste)',
        ),
    ]
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
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """These extra arguments will be MERGED with the saved extra_args, with these arguments taking place over the saved values when applicable."""
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


class DialInputMultipleOtherStrategy(BaseModel):
    """TODO: document this"""
    workflow_id: ValidatedObjectId
    points: PositiveIntType
    strategy: Literal[
        'uncertainty',
        'expected_improvement',
        'upper_confidence_bound',
        'upper_confidence_bound_nomad',
        'polymer_acl_sampler',
        'hypercube',
    ]
    strategy_args: dict[str, float | int | bool] | None = Field(default=None)
    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description='If true, treat higher y values as better (e.g. y represents yield or profit).  If false, opposite (e.g. y represents error or waste)',
        ),
    ]
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
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """These extra arguments will be MERGED with the saved extra_args, with these arguments taking place over the saved values when applicable."""
    optimization_points: PositiveIntType = Field(default=1000)
    discrete_measurements: bool = Field(default=False)
    discrete_measurement_grid_size: list[PositiveIntType] = Field(default=[20, 20])


DialInputMultiple = Annotated[
    DialInputMultipleOtherStrategy,
    Field(discriminator='strategy'),
]


class DialInputPredictions(BaseModel):
    """This is the input dataclass for Dial for requesting a surrogate evaluation at a given number of points."""

    workflow_id: ValidatedObjectId

    points_to_predict: list[list[float]]
    extra_args: dict[str, float | int | bool | str | list[float] | tuple] | None = Field(
        default=None
    )
    """These extra arguments will be MERGED with the saved extra_args, with these arguments taking place over the saved values when applicable."""
