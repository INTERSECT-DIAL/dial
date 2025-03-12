from dataclasses import dataclass, field
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from dial_dataclass import DialWorkflowDatasetUpdate
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

    def create_workflow(self, initial_data: dict[str, Any]) -> str | None:
        """Initialize the workflow from initial data provided by the user

        Parameters are meant to be as generic as possible, validation should occur on the INTERSECT layer

        Returns:
          - the stringified DB ID if data was inserted successfully
          - None if we couldn't insert the data successfully
        """
        try:
            result = self._mongo_collection.insert_one(initial_data)
        except PyMongoError as e:
            logger.debug(e)
            return None

        return str(result.inserted_id)

    def get_workflow(self, workflow_id: ValidatedObjectId) -> dict[str, Any] | None:
        """Get a basic workflow by its MongoDB ObjectID

        Note that the value returned is meant to be a generic dictionary, we do not perform any Python validation here.

        However, the parameter to this argument SHOULD be assumed to have already been validated by Pydantic.
        """
        try:
            result = self._mongo_collection.find_one({'_id': workflow_id})
        except PyMongoError as e:
            logger.debug(e)
            return None
        return result

    def update_workflow_dataset(self, params: DialWorkflowDatasetUpdate) -> bool:
        """Update the dataset of a workflow"""
        try:
            collection = self._mongo_collection.find_one({'_id': params.workflow_id})
        except PyMongoError as e:
            logger.debug(e)
            return False

        try:
            dataset_x_len = len(collection['dataset_x'][0])
            if dataset_x_len != len(params.next_x):
                return False
            self._mongo_collection.update_one(
                {'_id': params.workflow_id},
                {
                    '$push': {
                        'dataset_x': params.next_x,
                        'dataset_y': params.next_y,
                    }
                },
            )
        except (TypeError, IndexError, PyMongoError) as e:
            logger.warning(e)
            return False
        return True
