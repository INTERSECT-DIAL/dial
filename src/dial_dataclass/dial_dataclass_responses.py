from pydantic import BaseModel

from .pydantic_helpers import ValidatedObjectId


class DialDataResponse1D(BaseModel):
    """Possible response from DIAL"""

    data: list[float]
    """Raw data"""
    workflow_id: ValidatedObjectId
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""


class DialDataResponse2D(BaseModel):
    """Possible response from DIAL"""

    data: list[list[float]]
    """Raw data"""
    workflow_id: ValidatedObjectId
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""
