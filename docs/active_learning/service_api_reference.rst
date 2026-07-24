=======
Service
=======

All public members except for ``intersect_sdk_capability_name`` and ``intersect_sdk_events`` represent the public request API. (If looking at source code, these are the functions decorated with ``@intersect_message``.) You send a request message according to the parameter's type, DIAL sends a response message according to its return type.

Note that ``status`` is meant to represent the status of the DIAL Service as a whole, and will not be representative of a specific workflow.

DIAL may also send events on its own, these will be defined under the `intersect_sdk_events` object on the Capability class.

For a more in-depth look into how INTERSECT functions, please see the `INTERSECT-SDK documentation <https://intersect-python-sdk.readthedocs.io/>`_.

.. automodule:: dial_service.dial_service
   :members:
   :undoc-members:
