from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional

class BOALaaSInputBase(BaseModel):
    dataset_x: list[list[float]] #the input vectors of the training data
    dataset_y: list[float] #the output values of the training data
    y_is_good: bool  #if True, treat higher y values as better (e.g. y represents yield or profit).  If False, opposite (e.g. y represents error or waste)
    kernel: Literal["rbf", "matern"]
    length_per_dimension: bool #if True, will have the kernel use a separate length scale for each dimension (useful if scales differ).  If False, all dimensions are forced to the same length scale
    bounds: list[list[float]]

    preprocess_log: bool = Field(default=False)
    preprocess_standardize: bool = Field(default=False)

    @field_validator("dataset_x")
    @classmethod
    def validate_dataset_x(cls, x):
        if len(set(len(row) for row in x))>1:
            raise ValueError("Unequal vector lengths in dataset_x")
        return x
    
    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, bounds):
        for row in bounds:
            if len(row)!=2 or row[0]>row[1]:
                raise ValueError(f"Bounds entries must be [low,high], not {row}")
        return bounds

class BOALaaSInputSingle(BOALaaSInputBase):
    strategy: Literal["uncertainty", "expected_improvement", "confidence_bound"]
    confidence_bound: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def validate_confidence_bound(self):
        if self.strategy == "confidence_bound":
            if self.confidence_bound is None:
                raise ValueError("confidence_bound value must be specified for confidence_bound strategy")
            if not (.5 < self.confidence_bound < 1):
                raise ValueError("confidence_bound value must in (.5, 1)")
        return self

class BOALaaSInputMultiple(BOALaaSInputBase):
    points: int
    strategy: Literal["random", "hypercube"]

class BOALaaSInputPredictions(BOALaaSInputBase):
    points_per_dimension: list[int]
