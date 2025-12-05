from pydantic import BaseModel


class DialDataResponse1D(BaseModel):
    """Possible response from DIAL"""

    data: list[float]
    """Raw data"""
    workflow_id: str
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""


class DialDataResponse2D(BaseModel):
    """Possible response from DIAL"""

    data: list[list[float]]
    """Raw data"""
    workflow_id: str
    """The same workflow ID that was used to get the data, to facilitate possible load balancing."""
