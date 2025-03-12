from typing import Any

import gpax
from intersect_sdk import IntersectBaseCapabilityImplementation, intersect_message, intersect_status

from dial_dataclass import (
    DialInputMultiple,
    DialInputPredictions,
    DialInputSingle,
    DialWorkflowCreationParams,
    DialWorkflowDatasetUpdate,
)
from dial_dataclass.pydantic_helpers import ValidatedObjectId

from .dial_service_implementations import (
    internal_get_next_point,
    internal_get_next_points,
    internal_get_surrogate_values,
)
from .mongo_handler import MongoDBCredentials, MongoDBHandler
from .serverside_data import (
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)

gpax.utils.enable_x64()

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


class DialCapabilityImplementation(IntersectBaseCapabilityImplementation):
    """Internal guts for GP usage."""

    intersect_sdk_capability_name = 'dial'

    def __init__(self, credentials: dict[str, Any]):
        super().__init__()
        self.mongo_handler = MongoDBHandler(MongoDBCredentials(**credentials))

    @intersect_message()
    def initialize_workflow(self, client_data: DialWorkflowCreationParams) -> str:
        """Initializes a stateful workflow for DIALED.

        Takes in initial data points, and returns the ID of the associated workflow.
        """

        workflow_id = self.mongo_handler.create_workflow(client_data.model_dump())
        if not workflow_id:
            raise Exception  # noqa: TRY002 (Expected, we don't need to provide a detailed error message at this point)
        return workflow_id

    @intersect_message()
    def get_workflow_data(self, uuid: ValidatedObjectId) -> DialWorkflowCreationParams:
        """Returns the current state of the workflow associated with the id"""
        db_result = self.mongo_handler.get_workflow(uuid)
        if not db_result:
            raise Exception  # noqa: TRY002 (workflow does not exist - TODO the former should realistically be a Pydantic ValidationError that can propogate to the client)
        return DialWorkflowCreationParams(**db_result)

    @intersect_message()
    def update_workflow_with_data(self, update_params: DialWorkflowDatasetUpdate) -> None:
        """Updates the DB with the provided params. Success of operation is based off whether or not the INTERSECT response is an error."""
        db_result = self.mongo_handler.update_workflow_dataset(update_params)
        if not db_result:
            raise Exception  # noqa: TRY002 (workflow does not exist OR the length of the x value didn't match the rest) - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)

    @intersect_message()
    # trains a model and then recommends a point to measure based on user's requested strategy:
    def get_next_point(self, client_data: DialInputSingle) -> list[float]:
        """Trains a model, and then recommends a point to measure based on user's requested strategy."""
        workflow_id = ValidatedObjectId(client_data.workflow_id)
        workflow_state = self.mongo_handler.get_workflow(workflow_id)
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)

        data = ServersideInputSingle(DialWorkflowCreationParams(**workflow_state), client_data)

        return internal_get_next_point(data)

    @intersect_message
    def get_next_points(self, client_data: DialInputMultiple) -> list[list[float]]:
        workflow_id = ValidatedObjectId(client_data.workflow_id)
        workflow_state = self.mongo_handler.get_workflow(workflow_id)
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)

        data = ServersideInputMultiple(DialWorkflowCreationParams(**workflow_state), client_data)

        return internal_get_next_points(data)

    @intersect_message
    def get_surrogate_values(self, client_data: DialInputPredictions) -> list[list[float]]:
        """Trains a model then returns 3 lists based on user-supplied points:
        -Index 0: Predicted values.  These are inverse transformed (undoing the preprocessing to put them on the same scale as dataset_y)
        -Index 1: Inverse-transformed uncertainties.  If inverse-transforming is not possible (due to log-preprocessing), this will be all -1
        -Index 2: Uncertainties without inverse transformation
        """
        workflow_id = ValidatedObjectId(client_data.workflow_id)
        workflow_state = self.mongo_handler.get_workflow(workflow_id)
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)
        data = ServersideInputPrediction(DialWorkflowCreationParams(**workflow_state), client_data)

        return internal_get_surrogate_values(data)

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
