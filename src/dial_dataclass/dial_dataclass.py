from typing import Annotated, Literal, Optional

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


def _validate_dataset_lengths(dataset: list[any]) -> bool:
    """validate the lengths of dataset entries"""
    if len(dataset) > 1:
        data = dataset[0]
        if data is list:
            target_length = len(dataset[0])
            for row in dataset[1:]:
                if len(row) != target_length:
                    return False
    return True


def _validate_dims_and_length(data: dict,
                              x_name: str,
                              y_name: str) -> tuple[int, int]:
    """validate the lengths of datasets"""

    print(data)

    dim_x = data.get("dim_x")
    dim_y = data.get("dim_y")
    dataset_x = data[x_name]
    dataset_y = data[y_name]

    len_x = len(dataset_x)
    len_y = len(dataset_y)

    if len_y != len_x:
        msg = (f'Unequal number of points in {x_name} {len_x=}'
               f' and {y_name} {len_y=}.')
        raise ValueError(msg)

    def compute_dim(dim, dataset, name):
        lenn = len(dataset)
        if dim is None and lenn == 0:
            msg = (f'Can not infer dim from empty dataset {name}.'
                   'Set dim to the correct dimension.')
            raise ValueError(msg)
        else:
            inferred_len = len(dataset[0]) if dataset[0] is list else 1
            if dim is not None and inferred_len != dim_x:
                msg = (f'Vectors in {name} must be of length {dim=}.'
                       'Set dim to the correct dimension.')
                raise ValueError(msg)
            else:
                dim = inferred_len
        return dim

    dim_x = compute_dim(dim_x, dataset_x, x_name)
    dim_y = compute_dim(dim_y, dataset_y, y_name)

    return dim_x, dim_y


def _validate_labels(data: dict,
                     x_name: str,
                     y_name: str) -> tuple[str, str]:
    """validate the lengths of datasets"""

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
        list[float | Annotated[
            # TODO: this could be the default
                         list[float],
                         Field(description='Field lengths of all subarrays should be equal'),
                      ]
             ],
        Field(
            description=('The output values of the training data.'
                         ' Length should equal dataset_x'),
        ),
    ]
    x_labels: Annotated[
        str | list[str],
        Field(default='x',
              description='Labels for input variables x.')
    ]
    y_labels: Annotated[
        str | list[str],
        Field(default='y',
              description='Labels for output variables y.')
    ]
    dim_x: Annotated[int | None, Field(
        default=None,
        description=('Provide the dimension of entries in dataset_x explicitly,'
                     ' e.g. if the initial dataset is empty.'
                     ' If None, it will be inferred from dataset_x if possible.'),
    )]
    dim_y: Annotated[int | None, Field(
        default=1,
        description=('Provide the dimension of entries in dataset_y explicitly,'
                     ' e.g. if the initial dataset is empty.'
                     ' If None, it will be inferred from dataset_y if possible.'),
    )]
    # maximize_y: TODO, provide a language to determine which y to maximize

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

    @field_validator('dataset_x', 'dataset_y')
    @classmethod
    def ensure_consistent_dataset_x_lengths(cls, dataset, ctx):
        is_valid = _validate_dataset_lengths(dataset)
        if not is_valid:
            msg = f'Unequal vector lengths in {ctx.field_name}'
            raise ValueError(msg)
        return dataset

    # order rows as [low, high] - do NOT error out here, we can efficiently handle normalization
    @field_validator('bounds')
    @classmethod
    def order_bounds(cls, bounds: list[list[float]]):
        for row in bounds:
            row.sort()
        return bounds

    @model_validator(mode='after')
    def validate_dims_and_length(self, values):
        _validate_dims_and_length(vars(self),
                                  'dataset_x', 'dataset_y')
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

    @field_validator('next_x_list', 'next_y_list')
    @classmethod
    def ensure_consistent_dataset_x_lengths(cls, dataset, ctx):
        is_valid = _validate_dataset_lengths(dataset)
        if not is_valid:
            msg = f'Unequal vector lengths in {ctx.field_name}'
            raise ValueError(msg)
        return dataset

    @model_validator(mode='after')
    def validate_dims_and_length(self, values):
        _validate_dims_and_length(vars(self),
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
