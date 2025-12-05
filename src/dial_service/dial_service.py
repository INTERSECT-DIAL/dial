import logging
import pickle
from typing import Any

from intersect_sdk import IntersectBaseCapabilityImplementation, intersect_message, intersect_status

from dial_dataclass import (
    DialDataResponse1D,
    DialDataResponse2D,
    DialInputMultiple,
    DialInputPredictions,
    DialInputSingle,
    DialWorkflowDatasetUpdate,
    DialWorkflowDatasetUpdates,
)
from dial_dataclass.pydantic_helpers import ValidatedObjectId

from . import core
from .mongo_handler import MongoDBCredentials, MongoDBHandler
from .serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from .service_specific_dataclasses import DialWorkflowCreationParamsService

logger = logging.getLogger(__name__)


class DialCapabilityImplementation(IntersectBaseCapabilityImplementation):
    """Internal guts for GP usage."""

    intersect_sdk_capability_name = 'dial'

    def __init__(self, credentials: dict[str, Any]):
        super().__init__()
        self.mongo_handler = MongoDBHandler(MongoDBCredentials(**credentials))

    ### STATEFUL + WORKFLOW FUNCTIONS ###

    @intersect_message()
    def initialize_workflow(self, client_data: DialWorkflowCreationParamsService) -> str:
        """Initializes a stateful workflow for DIALED.

        Takes in initial data points, and returns the ID of the associated workflow.
        """

        try:
            server_data = ServersideInputBase(client_data)
            model = pickle.dumps(core.train_model(server_data), protocol=5)
            workflow_id = self.mongo_handler.create_workflow(client_data.model_dump(), model)
        except Exception:
            logger.exception('initialize_workflow exception')
            workflow_id = None
        if not workflow_id:
            msg = "Couldn't initialize workflow"
            raise Exception(msg)  # noqa: TRY002 (Expected, we don't need to provide a detailed error message at this point)
        return workflow_id

    @intersect_message()
    def get_workflow_data(self, uuid: ValidatedObjectId) -> DialWorkflowCreationParamsService:
        """Returns the current state of the workflow associated with the id"""
        try:
            db_result = self.mongo_handler.get_workflow(uuid)
        except Exception:
            logger.exception('get_workflow_data exception for %s', uuid)
            db_result = None
        if not db_result:
            msg = f"Couldn't get workflow data with id {uuid}"
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO the former should realistically be a Pydantic ValidationError that can propogate to the client)
        return DialWorkflowCreationParamsService(**db_result)

    @intersect_message()
    def update_workflow_with_data(
        self, update_params: DialWorkflowDatasetUpdate
    ) -> ValidatedObjectId:
        """Updates the DB with the provided params. Success of operation is based off whether or not the INTERSECT response is an error."""

        # TODO - all exceptions should realistically provide error information to the client. INTERSECT-SDK v0.9 will introduce a specific exception we can throw which will allow us to do this.
        try:
            db_get_result = self.mongo_handler.get_workflow(update_params.workflow_id)
        except Exception:
            logger.exception('update_workflow exception for %s', update_params.workflow_id)
            db_get_result = None
        if not db_get_result:
            msg = f'Could not get workflow with id {update_params.workflow_id}'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist)

        try:
            pretrain_result = DialWorkflowCreationParamsService(**db_get_result)
        except Exception:
            logger.exception(
                'update_workflow validation exception for %s', update_params.workflow_id
            )
            pretrain_result = None
        if not pretrain_result or (
            len(pretrain_result.dataset_x) > 0
            and len(pretrain_result.dataset_x[0]) != len(update_params.next_x)
        ):
            msg = f'Length mismatch in update function for workflow ID {update_params.workflow_id}'
            raise Exception(msg)  # noqa: TRY002 (data structure mismatch)

        try:
            pretrain_result.dataset_x.append(update_params.next_x)
            pretrain_result.dataset_y.append(update_params.next_y)
            server_data = ServersideInputBase(pretrain_result)

            if update_params.backend_args is not None:
                server_data.backend_args = update_params.backend_args

            if update_params.kernel_args is not None:
                server_data.kernel_args = update_params.kernel_args

            if update_params.extra_args is not None:
                server_data.extra_args = update_params.extra_args

            model = pickle.dumps(core.train_model(server_data), protocol=5)

            db_update_result = self.mongo_handler.update_workflow_dataset(update_params, model)
        except Exception:
            logger.exception('update_workflow exception for %s', update_params.workflow_id)
            db_update_result = None
        if not db_update_result:
            msg = f"Couldn't update workflow with new data for workflow {update_params.workflow_id}"
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist OR the length of the x value didn't match the rest) - TODO this should realistically be a Pydantic ValidationError that can propogate to the client) a Pydantic ValidationError that can propogate to the client)

        return update_params.workflow_id

    @intersect_message()
    def update_workflow_with_batch_data(
        self, update_params: DialWorkflowDatasetUpdates
    ) -> ValidatedObjectId:
        try:
            db_get_result = self.mongo_handler.get_workflow(
                update_params.workflow_id, include_model=True
            )
        except Exception:
            logger.exception('update_workflow_with_batch_data init %s', update_params.workflow_id)
            db_get_result = None
        if not db_get_result:
            exc = f'Could not get workflow with id {update_params.workflow_id}'
            raise Exception(exc)  # noqa: TRY002

        try:
            pretrain = DialWorkflowCreationParamsService(**db_get_result)
        except Exception:
            logger.exception(
                'update_workflow_with_batch_data validation %s', update_params.workflow_id
            )
            pretrain = None
        if not pretrain:
            exc = f'Workflow validation failed for {update_params.workflow_id}'
            raise Exception(exc)  # noqa: TRY002

        # shape check
        expected_dim = (
            len(pretrain.dataset_x[0]) if pretrain.dataset_x else len(update_params.next_x_list[0])
        )
        for row in update_params.next_x_list:
            if len(row) != expected_dim:
                exc = 'Length mismatch in update function'
                raise Exception(exc)  # noqa: TRY002

        try:
            pretrain.dataset_x.extend(update_params.next_x_list)
            pretrain.dataset_y.extend(update_params.next_y_list)
            server_data = ServersideInputBase(pretrain)

            if update_params.backend_args is not None:
                server_data.backend_args = update_params.backend_args
            if update_params.kernel_args is not None:
                server_data.kernel_args = update_params.kernel_args
            if update_params.extra_args is not None:
                server_data.extra_args = update_params.extra_args

            model = pickle.dumps(core.train_model(server_data), protocol=5)
            db_update_result = self.mongo_handler.update_workflow_dataset_batch(
                update_params, model
            )
        except Exception:
            logger.exception(
                'update_workflow_with_batch_data training %s', update_params.workflow_id
            )
            db_update_result = None
        if not db_update_result:
            exc = f"Couldn't update workflow with new batch data for {update_params.workflow_id}"
            raise Exception(exc)  # noqa: TRY002

        return update_params.workflow_id

    ### STATELESS FUNCTIONS ###

    @intersect_message()
    # trains a model and then recommends a point to measure based on user's requested strategy:
    def get_next_point(self, client_data: DialInputSingle) -> DialDataResponse1D:
        """Trains a model, and then gets the next point for optimization based on the provided strategy.

        Args:
            client_data (DialInputSingle): Input data containing bounds, strategy, and other parameters.

        Returns:
            list[float]: The selected point for the next iteration.
        """
        try:
            workflow_id = ValidatedObjectId(client_data.workflow_id)
            workflow_state = self.mongo_handler.get_workflow(workflow_id)
        except Exception:
            logger.exception(
                'get_next_point exception (state initialization) for %s', client_data.workflow_id
            )
            workflow_state = None
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)

        try:
            model = pickle.loads(workflow_state['model'])  # noqa: S301 (XXX - this is technically trusted data as long as the DB hasn't been modified)
            validated_state = DialWorkflowCreationParamsService(**workflow_state)
            if client_data.extra_args:
                if validated_state.extra_args:
                    validated_state.extra_args.update(client_data.extra_args)
                else:
                    validated_state.extra_args = client_data.extra_args
            data = ServersideInputSingle(validated_state, client_data)

            return_data = core.get_next_point(data, model)
            return DialDataResponse1D(
                data=return_data,
                workflow_id=workflow_id,
            )
        except Exception as err:
            logger.exception(
                'get_next_point exception (primary logic) for %s', client_data.workflow_id
            )
            msg = f'get_next_point exception (primary logic) for {client_data.workflow_id}'
            raise Exception(msg) from err  # noqa: TRY002 (for INTERSECT)

    @intersect_message
    def get_next_points(self, client_data: DialInputMultiple) -> DialDataResponse2D:
        """
        Get multiple next points for optimization based on the provided strategy.

        Args:
            client_data: Input data containing bounds, strategy, and other parameters.

        Returns:
            list[list[float]]: A list of selected points for the next iteration.
        """
        try:
            workflow_id = ValidatedObjectId(client_data.workflow_id)
            workflow_state = self.mongo_handler.get_workflow(workflow_id)
        except Exception:
            logger.exception(
                'get_next_pointS exception (state initialization) for %s', client_data.workflow_id
            )
            workflow_state = None
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)

        try:
            model = pickle.loads(workflow_state['model'])  # noqa: S301 (XXX - this is technically trusted data as long as the DB hasn't been modified)
            validated_state = DialWorkflowCreationParamsService(**workflow_state)
            if client_data.extra_args:
                if validated_state.extra_args:
                    validated_state.extra_args.update(client_data.extra_args)
                else:
                    validated_state.extra_args = client_data.extra_args
            data = ServersideInputMultiple(validated_state, client_data)

            return_data = core.get_next_points(data, model)
            return DialDataResponse2D(
                data=return_data,
                workflow_id=workflow_id,
            )
        except Exception as err:
            logger.exception(
                'get_next_pointS exception (primary logic) for %s', client_data.workflow_id
            )
            msg = f'get_next_pointS exception (primary logic) for {client_data.workflow_id}'
            raise Exception(msg) from err  # noqa: TRY002 (for INTERSECT)

    @intersect_message
    def get_surrogate_values(self, client_data: DialInputPredictions) -> DialDataResponse2D:
        """Trains a model then returns 3 lists based on user-supplied points:
        -Index 0: Predicted values.  These are inverse transformed (undoing the preprocessing to put them on the same scale as dataset_y)
        -Index 1: Inverse-transformed uncertainties.  If inverse-transforming is not possible (due to log-preprocessing), this will be all -1
        -Index 2: Uncertainties without inverse transformation
        """
        try:
            workflow_id = ValidatedObjectId(client_data.workflow_id)
            workflow_state = self.mongo_handler.get_workflow(workflow_id, include_model=True)
        except Exception:
            logger.exception(
                'get_surrogate_values exception (state initialization) for %s',
                client_data.workflow_id,
            )
            workflow_state = None
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise Exception(msg)  # noqa: TRY002 (workflow does not exist - TODO this should realistically be a Pydantic ValidationError that can propogate to the client)

        try:
            model = pickle.loads(workflow_state['model'])  # noqa: S301 (XXX - this is technically trusted data as long as the DB hasn't been modified)
            validated_state = DialWorkflowCreationParamsService(**workflow_state)
            if client_data.extra_args:
                if validated_state.extra_args:
                    validated_state.extra_args.update(client_data.extra_args)
                else:
                    validated_state.extra_args = client_data.extra_args
            data = ServersideInputPrediction(validated_state, client_data)

            return_data = core.get_surrogate_values(data, model)
            return DialDataResponse2D(
                data=return_data,
                workflow_id=workflow_id,
            )
        except Exception as err:
            logger.exception(
                'get_surrogate_values exception (primary logic) for %s', client_data.workflow_id
            )
            msg = f'get_surrogate_values exception (primary logic) for {client_data.workflow_id}'
            raise Exception(msg) from err  # noqa: TRY002 (for INTERSECT)

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
