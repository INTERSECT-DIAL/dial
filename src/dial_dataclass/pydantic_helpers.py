from typing import Annotated, Any

from bson import ObjectId
from bson.errors import InvalidId
from pydantic_core import core_schema


# based on an answer by Pydantic's creator, Samuel Colvin: https://stackoverflow.com/a/76719893
class _ObjectIdPydanticAnnotation:
    """Internal class used for the ValidatedObjectId type"""

    @classmethod
    def validate_object_id(cls, v: Any, handler) -> ObjectId:
        if isinstance(v, ObjectId):
            return v

        s = handler(v)
        try:
            return ObjectId(s)
        except (InvalidId, TypeError) as e:
            msg = 'Invalid ObjectId - use the 24-character hex representation'
            raise ValueError(msg) from e

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, _handler) -> core_schema.CoreSchema:
        if source_type is not ObjectId:
            raise AssertionError
        return core_schema.no_info_wrap_validator_function(
            cls.validate_object_id,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        # add some standard information to the JSON schema, which can be used by client validation tools
        json_schema['minLength'] = 24
        json_schema['maxLength'] = 24
        json_schema['pattern'] = r'^[0-9a-f]{24}$'
        json_schema['description'] = '24-character hex string representation of a MongoDB ObjectID'
        return json_schema


ValidatedObjectId = Annotated[
    ObjectId,
    _ObjectIdPydanticAnnotation,
]
"""This is the actual annotation you can use on a BaseModel property.

By validating this through Pydantic, INTERSECT knows that it can tell clients directly that their workflow_id is in an invalid format.

(We explicitly need to raise validation errors through Pydantic if we want clients to understand the error, raising generic Exceptions means that INTERSECT will not reveal the reason for failure in the error message)
"""
