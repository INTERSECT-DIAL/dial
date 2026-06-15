from abc import ABC
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .pydantic_helpers import ValidatedObjectId

_POSSIBLE_BACKENDS = ('sklearn', 'gpax', 'sable')

BackendType = Literal[_POSSIBLE_BACKENDS]

PositiveIntType = Annotated[int, Field(ge=0)]

Label = Annotated[str, Field(max_length=50, description='Label for a dataset entry.')]
FloatOrLabel = Annotated[
    float | Label,
    Field(
        description='A constant float, or the label of a dataset entry.',
    ),
]


class BaseDistribution(BaseModel, ABC):
    """Base class for a statistical distribution."""

    name: str
    loc: Annotated[
        Label,
        Field(
            description='The location (mean, or center) of the distribution',
        ),
    ]
    scale: Annotated[
        FloatOrLabel,
        Field(
            description='The scale or standard deviation of the distribution',
        ),
    ]


class Delta(BaseDistribution):
    """The Delta distribution is deterministic and equal to its mean/loc."""

    name: str = Field(default='Delta', frozen=True)
    scale: float = Field(gt=0.0, lt=0.0, default=0.0, frozen=True)


class Normal(BaseDistribution):
    """The normal distribution is determined by loc (mean) and scale (standard deviation)."""

    name: str = Field(default='Normal', frozen=True)


Distribution = Annotated[Delta | Normal, Field(description='Union of all supported Distributions.')]


def _validate_dataset_lengths(dataset: list[any]) -> bool:
    """validate the lengths of dataset entries"""
    if len(dataset) > 1:
        data = dataset[0]
        if isinstance(data, list):
            target_length = len(dataset[0])
            for row in dataset[1:]:
                if len(row) != target_length:
                    return False
    return True


def _validate_dims_and_length(data: dict, x_name: str, y_name: str) -> tuple[int, int]:
    """validate the lengths of datasets, and compute dim_x and dim_y"""

    # print('\n'.join(f'{k}: {v}' for k, v in data.items()))

    dim_x = data.get('dim_x')
    dim_y = data.get('dim_y')
    dataset_x = data[x_name]
    dataset_y = data[y_name]

    len_x = len(dataset_x)
    len_y = len(dataset_y)

    if len_y != len_x:
        msg = f'Unequal number of points in {x_name} {len_x=} and {y_name} {len_y=}.'
        raise ValueError(msg)

    def compute_dim(dim, dataset, name):
        lenn = len(dataset)
        if dim is None and lenn == 0:
            msg = f'Can not infer dim from empty dataset {name}.Set dim to the correct dimension.'
            raise ValueError(msg)

        if lenn > 0:
            inferred_dim = len(dataset[0]) if isinstance(dataset[0], list) else 1
            if dim is not None and inferred_dim != dim:
                msg = (
                    f'Vectors in {name} must be of length {dim=}.Set dim to the correct dimension.'
                )
                raise ValueError(msg)
            dim = inferred_dim

        return dim

    # compute dimensions and validate consistency
    dim_x = compute_dim(dim_x, dataset_x, x_name)
    dim_y = compute_dim(dim_y, dataset_y, y_name)

    # print(dim_x, dataset_x, x_name)
    # print(dim_y, dataset_y, y_name)

    # validate bounds, if they exist
    bounds = data.get('bounds')
    if bounds is not None and len(bounds) != dim_x:
        msg = f'Bounds have incorrect length {len(bounds)} != {dim_x=}'
        raise ValueError(msg)

    return dim_x, dim_y


def _validate_labels(cls) -> tuple[str, str]:
    """validate the lengths of labels"""
    labels_x, labels_y = cls.labels_x, cls.labels_y
    dim_x, dim_y = cls.dim_x, cls.dim_y

    def compute_labels(dim, labels):
        if isinstance(labels, list):
            if dim is not None and dim != len(labels):
                msg = f'Labels {labels} ar not consistent with data dimension {dim=}'
                raise ValueError(msg)
        elif dim > 1:
            # give each parameter a unique label by appending a number
            labels = [f'{labels}_{i + 1}' for i in range(dim)]
        else:
            # normalize to single element list
            labels = [labels]
        return labels

    labels_x = compute_labels(dim_x, labels_x)
    labels_y = compute_labels(dim_y, labels_y)

    return labels_x, labels_y


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
        list[
            float
            | Annotated[
                # TODO: this could be the default
                list[float],
                Field(description='Field lengths of all subarrays should be equal'),
            ]
        ],
        Field(
            description=('The output values of the training data. Length should equal dataset_x'),
        ),
    ]
    labels_x: Annotated[
        Label | list[Label], Field(default='x', description='Labels for input variables x.')
    ]
    labels_y: Annotated[
        Label | list[Label], Field(default='y', description='Labels for output variables y.')
    ]
    dim_x: Annotated[
        PositiveIntType | None,
        Field(
            default=None,
            description=(
                'Provide the dimension of entries in dataset_x explicitly,'
                ' e.g. if the initial dataset is empty.'
                ' If None, it will be inferred from dataset_x if possible.'
            ),
        ),
    ]
    dim_y: Annotated[
        PositiveIntType | None,
        Field(
            default=1,
            description=(
                'Provide the dimension of entries in dataset_y explicitly,'
                ' e.g. if the initial dataset is empty.'
                ' If None, it will be inferred from dataset_y if possible.'
            ),
        ),
    ]
    statistics_y: Annotated[
        Distribution,
        Field(
            default=Delta(loc='y'),
            description=(
                'Provide the statistical model underlying the y data: For example:'
                " Delta(loc='y') means that the y data is without error,"
                " Normal(loc='y', scale=0.1) is a standard error with mean y and standard deviation 0.1"
                " Normal(loc='y', scale='yerr') takes the std.dev. from the data column yerr."
            ),
        ),
    ]

    y_is_good: Annotated[
        bool,
        Field(
            default=True,  # <-- Set default here
            description=(
                'If true, treat higher y values as better'
                ' (e.g. y represents yield or profit).'
                ' If false, opposite (e.g. y represents error or waste)'
            ),
        ),
    ]
    kernel: Literal['rbf', 'matern', 'linear']
    bounds: list[
        Annotated[
            list[float],
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
    def ensure_consistent_dataset_lengths(cls, dataset, info: ValidationInfo):
        is_valid = _validate_dataset_lengths(dataset)
        if not is_valid:
            msg = f'Unequal vector lengths in {info.field_name}'
            raise ValueError(msg)
        return dataset

    # order rows as [low, high] - do NOT error out here, we can efficiently handle normalization
    @field_validator('bounds', mode='after')
    @classmethod
    def order_bounds(cls, bounds: list[list[float]]):
        for row in bounds:
            row.sort()
        return bounds

    @model_validator(mode='after')
    def validate_dims_and_length(self):
        # compute the dimensions and validate consistency
        self.dim_x, self.dim_y = _validate_dims_and_length(vars(self), 'dataset_x', 'dataset_y')
        # compute or validate labels
        self.labels_x, self.labels_y = _validate_labels(self)
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
    def validate_dims_and_length(self):
        _validate_dims_and_length(vars(self), 'next_x_list', 'next_y_list')
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
        'random',
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
