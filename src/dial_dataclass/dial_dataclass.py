from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PositiveIntType = Annotated[int, Field(ge=0)]


class DialInputBase(BaseModel):
    """This is the base input dataclass for BOALaaS."""

    dataset_x: list[list[float]]  # the input vectors of the training data
    dataset_y: list[float]  # the output values of the training data
    y_is_good: bool  # if True, treat higher y values as better (e.g. y represents yield or profit).  If False, opposite (e.g. y represents error or waste)
    kernel: Literal['rbf', 'matern']
    length_per_dimension: bool  # if True, will have the kernel use a separate length scale for each dimension (useful if scales differ).  If False, all dimensions are forced to the same length scale
    bounds: list[list[float]]
    backend: Literal['sklearn', 'gpax']
    seed: int

    preprocess_log: bool = Field(default=False)
    preprocess_standardize: bool = Field(default=False)

    @field_validator('dataset_x')
    @classmethod
    def validate_dataset_x(cls, x):
        if len({len(row) for row in x}) > 1:
            msg = 'Unequal vector lengths in dataset_x'
            raise ValueError(msg)
        return x

    @field_validator('bounds')
    @classmethod
    def validate_bounds(cls, bounds):
        for row in bounds:
            if len(row) != 2 or row[0] > row[1]:
                msg = f'Bounds entries must be [low,high], not {row}'
                raise ValueError(msg)
        return bounds


class DialInputSingle(DialInputBase):
    """This is the input dataclass for Dial for selecting a single new point to measure."""

    strategy: Literal['random', 'uncertainty', 'expected_improvement', 'confidence_bound']
    optimization_points: PositiveIntType | None = Field(default=1000)
    confidence_bound: float | None = Field(default=None)
    discrete_measurements: bool | None = Field(default=False)
    discrete_measurement_grid_size: list[PositiveIntType] | None = Field(default=[20, 20])

    @model_validator(mode='after')
    def validate_confidence_bound(self):
        if self.strategy == 'confidence_bound':
            if self.confidence_bound is None:
                msg = 'confidence_bound value must be specified for confidence_bound strategy'
                raise ValueError(msg)
            if not (0.5 < self.confidence_bound < 1):
                msg = 'confidence_bound value must be in (.5, 1)'
                raise ValueError(msg)
        return self


class DialInputMultiple(DialInputBase):
    """This is the input dataclass for Dial for selecting a multiple new point to measures (i.e., a batch of measurements)."""

    points: int
    strategy: Literal['random', 'hypercube']


class DialInputPredictions(DialInputBase):
    """This is the input dataclass for Dial for requesting a surrogate evaluation at a given number of points."""

    points_to_predict: list[list[float]]
