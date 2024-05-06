from pydantic import BaseModel, field_validator
from typing import Literal

Vector = list[float]
Matrix = list[Vector]

class ActiveLearningInputData(BaseModel):
    dataset_x: Matrix #the input vectors of the training data
    dataset_y: Vector #the output values of the training data
    y_is_good: bool  #if True, treat higher y values as better (e.g. y represents yield or profit).  If False, opposite (e.g. y represents error or waste)
    kernel: Literal["rbf", "matern"]
    length_per_dimension: bool #if True, will have the kernel use a separate length scale for each dimension (useful if scales differ).  If False, all dimensions are forced to the same length scale
    bounds: Matrix

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
