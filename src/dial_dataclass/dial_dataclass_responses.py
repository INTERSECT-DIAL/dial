from typing import Annotated

from pydantic import BaseModel, Field

from .dial_dataclass import DialWorkflowCreationParamsClient
from .pydantic_helpers import ValidatedObjectId

PositiveIntType = Annotated[int, Field(ge=0)]


class DialWorkflowFullState(DialWorkflowCreationParamsClient):
    """Full state of the workflow."""

    workflow_id: ValidatedObjectId


class DialDataResponse1D(BaseModel):
    """Possible response from DIAL"""

    data: list[float]
    """Raw data"""
    workflow_id: ValidatedObjectId
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""
    dataset_x_size: int
    """Current length of dataset_x"""


class DialDataResponse2D(BaseModel):
    """Possible response from DIAL"""

    data: list[list[float]]
    """Raw data"""
    workflow_id: ValidatedObjectId
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""
    dataset_x_size: int
    """Current length of dataset_x"""


class DialSurrogateValuesResponse(BaseModel):
    """Response structure from calling get_surrogate_values()"""

    values: list[float]
    """The computed values (for example, from Gaussian backends, the means) from calling get_surrogate_values()"""
    transformed_stddevs: list[float]
    """The computed uncertainties from calling get_surrogate_values(), with an inverse transform. If inverse-transforming is not possible (due to log-preprocessing), this will be all -1"""
    stddevs: list[float]  # TODO will probably remove in future
    """The computed raw uncertainties from calling get_surrogate_values(), without an inverse transform"""
    dim_x: int
    """Number of dimensions of the associated data, derived from workflow"""
    bounds: list[list[float]]
    """Bounding box of the data, derived from workflow"""
    points_to_predict: list[list[float]]
    """Original list of points provided from the get_surrogate_values() input"""
    workflow_id: ValidatedObjectId
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""
    dataset_x_size: int
    """Current length of dataset_x"""
    transformed_stddevs_avg: float
    """the average of the transformed stddevs being returned"""
