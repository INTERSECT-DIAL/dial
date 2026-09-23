import logging
import traceback
from pathlib import Path
from typing import Any

from intersect_dial_dataclass import (
    DialDataResponse1D,
    DialDataResponse2D,
    DialInputMultiple,
    DialInputPredictions,
    DialInputSingle,
    DialSurrogateValuesResponse,
    DialWorkflowDatasetUpdate,
    DialWorkflowDatasetUpdates,
    DialWorkflowFullState,
)
from intersect_dial_dataclass.pydantic_helpers import ValidatedObjectId
from intersect_sdk import (
    IntersectBaseCapabilityImplementation,
    IntersectCapabilityError,
    intersect_message,
    intersect_status,
)

from . import core
from .model_storage import FileModelStorage
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

    def __init__(self, credentials: dict[str, Any], base_directory: Path):
        super().__init__()
        self.mongo_handler = MongoDBHandler(MongoDBCredentials(**credentials))
        self.file_storage = FileModelStorage(base_directory)

    def _load_full_state(
        self, workflow_id: ValidatedObjectId, *, include_model: bool = False
    ) -> dict[str, Any] | None:
        """Merge a workflow's Mongo metadata with its filesystem-backed model/dataset.

        The model and dataset live on disk (see FileModelStorage), not in Mongo, so
        every read path needs to stitch them back together before validating the
        combined dict as a DialWorkflowCreationParamsService.
        """
        db_result = self.mongo_handler.get_workflow(workflow_id)
        if not db_result:
            return None
        dataset_x, dataset_y = self.file_storage.read_dataset(str(workflow_id))
        full_state = {**db_result, 'dataset_x': dataset_x, 'dataset_y': dataset_y}
        if include_model:
            full_state['model'] = self.file_storage.read_model(str(workflow_id))
        return full_state

    ### STATEFUL + WORKFLOW FUNCTIONS ###

    @intersect_message()
    def initialize_workflow(self, client_data: DialWorkflowCreationParamsService) -> str:
        """Initializes a stateful workflow for DIALED.

        Takes in initial data points, and returns the ID of the associated workflow.
        """
        try:
            server_data = ServersideInputBase(client_data)
            has_initial_data = client_data.dataset_x and len(client_data.dataset_y) > 0
            if has_initial_data:
                # the user provided some initial data, so train a model
                model = core.train_model(server_data)
            else:
                # no initial data was provided, so just initialize a workflow ID and some common settings for the user
                model = core.initialize_model(server_data)
            initial_data = client_data.model_dump(exclude={'dataset_x', 'dataset_y'})
            workflow_id = self.mongo_handler.create_workflow(initial_data)
            if workflow_id:
                self.file_storage.write_model(workflow_id, model)
                # unconditionally touch dataset.tsv into existence (even empty) so a workflow
                # always has one - a missing file later means broken storage, not "no data yet"
                self.file_storage.append_dataset_batch(
                    workflow_id, client_data.dataset_x, client_data.dataset_y
                )
        except Exception:
            logger.exception('initialize_workflow exception')
            workflow_id = None
        if not workflow_id:
            msg = "Couldn't initialize workflow"
            raise IntersectCapabilityError(msg)
        return workflow_id

    @intersect_message()
    def get_workflow_data(self, uuid: ValidatedObjectId) -> DialWorkflowFullState:
        """Returns the current state of the workflow associated with the id"""
        try:
            full_state = self._load_full_state(uuid)
        except Exception:
            logger.exception('get_workflow_data exception for %s', uuid)
            full_state = None
        if not full_state:
            msg = f"Couldn't get workflow data with id {uuid}"
            raise IntersectCapabilityError(msg)
        return DialWorkflowFullState(
            workflow_id=uuid,
            dataset_x_size=len(full_state['dataset_x']),
            **full_state,
        )

    @intersect_message()
    def update_workflow_with_data(
        self, update_params: DialWorkflowDatasetUpdate
    ) -> ValidatedObjectId:
        """Updates the DB with the provided params. Success of operation is based off whether or not the INTERSECT response is an error."""

        try:
            db_get_result = self._load_full_state(update_params.workflow_id)
        except Exception:
            logger.exception('update_workflow exception for %s', update_params.workflow_id)
            db_get_result = None
        if not db_get_result:
            msg = f'Could not get workflow with id {update_params.workflow_id}'
            raise IntersectCapabilityError(msg)

        try:
            pretrain_result = DialWorkflowCreationParamsService(**db_get_result)
        except Exception:
            logger.exception(
                'update_workflow validation exception for %s',
                update_params.workflow_id,
            )
            pretrain_result = None
        if not pretrain_result or (
            len(pretrain_result.dataset_x) > 0
            and len(pretrain_result.dataset_x[0]) != len(update_params.next_x)
        ):
            msg = f'Length mismatch in update function for workflow ID {update_params.workflow_id}'
            raise IntersectCapabilityError(msg)

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

            model = core.train_model(server_data)

            self.file_storage.write_model(str(update_params.workflow_id), model)
            self.file_storage.append_dataset(
                str(update_params.workflow_id), update_params.next_x, update_params.next_y
            )
            db_update_result = self.mongo_handler.update_workflow_dataset(update_params)
        except Exception:
            logger.exception('update_workflow exception for %s', update_params.workflow_id)
            db_update_result = None
        if not db_update_result:
            msg = f"Couldn't update workflow with new data for workflow {update_params.workflow_id}"
            raise IntersectCapabilityError(msg)

        return update_params.workflow_id

    @intersect_message()
    def update_workflow_with_batch_data(
        self, update_params: DialWorkflowDatasetUpdates
    ) -> ValidatedObjectId:
        try:
            db_get_result = self._load_full_state(update_params.workflow_id)
        except Exception:
            logger.exception(
                'update_workflow_with_batch_data init %s',
                update_params.workflow_id,
            )
            db_get_result = None
        if not db_get_result:
            exc = f'Could not get workflow with id {update_params.workflow_id}'
            raise IntersectCapabilityError(exc)

        try:
            pretrain = DialWorkflowCreationParamsService(**db_get_result)
        except Exception:
            logger.exception(
                'update_workflow_with_batch_data validation %s',
                update_params.workflow_id,
            )
            pretrain = None
        if not pretrain:
            exc = f'Workflow validation failed for {update_params.workflow_id}'
            raise IntersectCapabilityError(exc)

        # shape check
        expected_dim = (
            len(pretrain.dataset_x[0]) if pretrain.dataset_x else len(update_params.next_x_list[0])
        )
        for row in update_params.next_x_list:
            if len(row) != expected_dim:
                exc = 'Length mismatch in update function'
                raise IntersectCapabilityError(exc)

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

            model = core.train_model(server_data)

            self.file_storage.write_model(str(update_params.workflow_id), model)
            self.file_storage.append_dataset_batch(
                str(update_params.workflow_id),
                update_params.next_x_list,
                update_params.next_y_list,
            )
            db_update_result = self.mongo_handler.update_workflow_dataset_batch(update_params)
        except Exception:
            logger.exception(
                'update_workflow_with_batch_data training %s',
                update_params.workflow_id,
            )
            db_update_result = None
        if not db_update_result:
            exc = f"Couldn't update workflow with new batch data for {update_params.workflow_id}"
            raise IntersectCapabilityError(exc)

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
            workflow_state = self._load_full_state(client_data.workflow_id, include_model=True)
        except Exception:
            logger.exception(
                'get_next_point exception (state initialization) for %s',
                client_data.workflow_id,
            )
            workflow_state = None
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise IntersectCapabilityError(msg)

        try:
            model = workflow_state['model']
            validated_state = DialWorkflowCreationParamsService(**workflow_state)
            data = ServersideInputSingle(validated_state, client_data)
            return_data = core.get_next_point(data, model)
            return DialDataResponse1D(
                data=return_data,
                workflow_id=client_data.workflow_id,
                dataset_x_size=len(validated_state.dataset_x),
            )
        except Exception as err:
            logger.exception(
                'get_next_point exception (primary logic) for %s',
                client_data.workflow_id,
            )
            raise IntersectCapabilityError(traceback.format_exc()) from err

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
            workflow_state = self._load_full_state(client_data.workflow_id, include_model=True)
        except Exception:
            logger.exception(
                'get_next_pointS exception (state initialization) for %s',
                client_data.workflow_id,
            )
            workflow_state = None
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise IntersectCapabilityError(msg)

        try:
            model = workflow_state['model']
            validated_state = DialWorkflowCreationParamsService(**workflow_state)
            data = ServersideInputMultiple(validated_state, client_data)
            return_data = core.get_next_points(data, model)
            return DialDataResponse2D(
                data=return_data,
                workflow_id=client_data.workflow_id,
                dataset_x_size=len(validated_state.dataset_x),
            )
        except Exception as err:
            logger.exception(
                'get_next_pointS exception (primary logic) for %s',
                client_data.workflow_id,
            )
            raise IntersectCapabilityError(traceback.format_exc()) from err

    @intersect_message
    def get_surrogate_values(
        self, client_data: DialInputPredictions
    ) -> DialSurrogateValuesResponse:
        """Trains a model then returns two lists based on user-supplied points:
        - Predicted values.
          These are inverse transformed (undoing the preprocessing to put them on the same scale as dataset_y)
        - Uncertainties.
          These are inverse transformed standard errors, transformed according to the differential of the transform.

        Additional metadata is also returned in the response.
        """
        try:
            workflow_state = self._load_full_state(client_data.workflow_id, include_model=True)
        except Exception:
            logger.exception(
                'get_surrogate_values exception (state initialization) for %s',
                client_data.workflow_id,
            )
            workflow_state = None
        if not workflow_state:
            msg = f'No workflow with id {client_data.workflow_id} exists'
            raise IntersectCapabilityError(msg)

        try:
            model = workflow_state['model']
            validated_state = DialWorkflowCreationParamsService(**workflow_state)
            if client_data.extra_args:
                if validated_state.extra_args:
                    validated_state.extra_args.update(client_data.extra_args)
                else:
                    validated_state.extra_args = client_data.extra_args
            data = ServersideInputPrediction(validated_state, client_data)

            means, stddevs, average_stddev = core.get_surrogate_values(data, model)
            return DialSurrogateValuesResponse(
                values=means,
                stddevs=stddevs,
                dim_x=validated_state.dim_x,
                points_to_predict=client_data.points_to_predict,
                bounds=validated_state.bounds,
                workflow_id=client_data.workflow_id,
                dataset_x_size=len(validated_state.dataset_x),
                stddevs_avg=average_stddev,
            )
        except Exception as err:
            logger.exception(
                'get_surrogate_values exception (primary logic) for %s',
                client_data.workflow_id,
            )
            raise IntersectCapabilityError(traceback.format_exc()) from err

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
