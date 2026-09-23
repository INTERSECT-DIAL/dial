from typing import Annotated, Any

try:
    from bson import ObjectId
    from bson.errors import InvalidId
except ImportError:
    # limited fixtures for Clients to use so they don't have to install pymongo, not for use by the Service
    class InvalidId(Exception):  # type: ignore[no-redef] # noqa: N818
        pass

    class ObjectId:  # type: ignore[no-redef]
        def __init__(self, oid: str):
            # this is a very limited constructor which assumes that any ObjectID instances will be created by the Service, which will not use this class
            if len(oid) != 24:
                raise InvalidId
            # enforce string with 24 hex characters, catch exception in
            bytes.fromhex(oid)
            self._id = oid

        def __str__(self):
            return self._id

        def __repr__(self):
            return f"ObjectId('{self!s}')"

        def __hash__(self):
            return hash(self._id)


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
        except (InvalidId, TypeError, ValueError) as e:
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
"""
Basic representation of the DIAL id structure, useful for double-checking Client code.

## Usage

If you are a Client, you don't actually need to import this class. You can initialize associated properties with a simple string or an existing variable, like this:

```
valid_hex_string = 'deadbeef12345678deadbeef'
value = DialWorkflowDatasetUpdate(workflow_id=valid_hex_string)
```
"""
