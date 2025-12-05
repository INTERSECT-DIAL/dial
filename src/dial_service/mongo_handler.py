from dataclasses import dataclass, field
from typing import Any

from bson import Binary
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from dial_dataclass import DialWorkflowDatasetUpdate, DialWorkflowDatasetUpdates
from dial_dataclass.pydantic_helpers import ValidatedObjectId

from .logger import logger


@dataclass
class MongoDBCredentials:
    username: str | None = field(default=None)
    password: str | None = field(default=None)
    host: str | None = field(default='127.0.0.1')
    port: int | None = field(default=27017)
    db_name: str = field(default='dial')


class MongoDBHandler:
    """Meant to be an abstract of the data service we plan to use eventually.

    NOTE: This initial implementation directly exposes the ObjectId strings externally and as input/output

    this can potentially be changed to use a field "workflow_id" , a UUID4, directly if desired (make sure to create it as an index)
    if we do not want to expose the DB implementation directly
    """

    def __init__(self, creds: MongoDBCredentials) -> None:
        client: MongoClient = MongoClient(
            host=creds.host,
            port=creds.port,
            username=creds.username,
            password=creds.password,
        )
        logger.info(client.admin.command('ping'))
        self._mongo_collection = client.get_database(creds.db_name).get_collection('workflows')
        # DO THE BELOW INSTEAD IF USING CUSTOM INDEXES INSTEAD
        # self._mongo_collection.create_index([('workflow_id', ASCENDING)], unique=True)

    def create_workflow(self, initial_data: dict[str, Any], model: bytes) -> str | None:
        """Initialize the workflow from initial data provided by the user

        Parameters are meant to be as generic as possible, validation should occur on the INTERSECT layer

        Returns:
          - the stringified DB ID if data was inserted successfully
          - None if we couldn't insert the data successfully
        """
        try:
            result = self._mongo_collection.insert_one({**initial_data, 'model': Binary(model)})
        except PyMongoError as e:
            logger.debug(e)
            return None

        return str(result.inserted_id)

    def get_workflow(
        self, workflow_id: ValidatedObjectId, include_model: bool = False
    ) -> dict[str, Any] | None:
        """Get a basic workflow by its MongoDB ObjectID, and maybe include a trained model along with it.

        Note that the value returned is meant to be a generic dictionary, we do not perform any Python validation here.

        However, the parameter to this argument SHOULD be assumed to have already been validated by Pydantic.
        """
        try:
            result = self._mongo_collection.find_one(
                {'_id': workflow_id}, {'fields': {'model': 0}} if not include_model else None
            )
        except PyMongoError as e:
            logger.debug(e)
            return None
        return result

    def update_workflow_dataset(self, params: DialWorkflowDatasetUpdate, model: bytes) -> bool:
        """Update the dataset of a workflow"""
        set_args = {
            'model': Binary(model),
        }
        if params.backend_args is not None:
            set_args['backend_args'] = params.backend_args
        if params.extra_args is not None:
            set_args['extra_args'] = params.extra_args
        if params.kernel_args is not None:
            set_args['kernel_args'] = params.kernel_args
        try:
            self._mongo_collection.update_one(
                {'_id': params.workflow_id},
                {
                    '$set': set_args,
                    '$push': {
                        'dataset_x': params.next_x,
                        'dataset_y': params.next_y,
                    },
                },
            )
        except (TypeError, IndexError, PyMongoError) as e:
            logger.warning(e)
            return False
        return True

    def update_workflow_dataset_batch(self, params: DialWorkflowDatasetUpdates, model: bytes) -> bool:
        set_args = {'model': Binary(model)}
        if params.backend_args is not None:
            set_args['backend_args'] = params.backend_args
        if params.extra_args is not None:
            set_args['extra_args'] = params.extra_args
        if params.kernel_args is not None:
            set_args['kernel_args'] = params.kernel_args
        try:
            result = self._mongo_collection.update_one(
                {'_id': params.workflow_id},
                {
                    '$set': set_args,
                    '$push': {
                        'dataset_x': {'$each': params.next_x_list},
                        'dataset_y': {'$each': params.next_y_list},
                    },
                },
            )
        except (TypeError, IndexError, PyMongoError) as e:
            logger.warning(e)
            return False
        return result.matched_count == 1 and result.modified_count >= 1
