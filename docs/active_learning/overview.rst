==================
Dataclass Overview
==================

DIAL uses the INTERSECT-SDK as its externally-facing API; all public functionality is marked in ``src/dial_service/dial_service.py`` with ``@intersect_message`` decorators. For all decorated functions, the function has a single input parameter (excluding ``self``); the typing of this parameter represents the structure of the input you need to provide to DIAL. The structure of the return value is the output you will be given.

There are a few ways to interact with DIAL:

- Using the iHub web interface to construct a campaign which calls into DIAL. This is the primary option to use in production, and should be what is used by any user without an extensive programming background. The capability name should be ``dial`` (the exact value is the value of ``intersect_sdk_capability_name`` in the DIAL service). iHub is able to directly display the DIAL API (including descriptions) in the browser, and can automatically help your campaign adhere to the input/output schemas.
- Using python scripting to create an INTERSECT-SDK Client (your own custom orchestrator) and using it to make requests to DIAL directly. See the `<client_guide.html>`_ for an in-depth guide on how to do this. This option is more useful for development, but it requires you are able to spin up an entire INTERSECT ecosystem yourself.

Creating a workflow
-------------------

1. You need to call ``initialize_workflow`` with your workflow parameters to create a workflow. This will return a workflow_id, which you will need to provide in all subsequent calls.
2. To do active learning, you can call the ``get_next_point``, ``get_next_points``, or ``get_surrogate_values`` endpoints. This only returns **suggestions**, it does not update the workflow directly.
3. To update the workflow, you can then call the various ``update_workflow`` endpoints, which will actually update the workflow state.

Active learning models are encapsulated from external view, but are only trained when calling ``initialize_workflow`` and the various ``update_workflow`` endpoints. This means that calling the active learning endpoints should generally be cheap, and you can call active learning multiple times before updating the workflow with your selected data.

References
==========

- `The API reference for the Service <service_api_reference.html>`_
- `The reference for the dataclass input/output objects <dataclass_reference.html>`_
- `INTERSECT-SDK reference <https://intersect-python-sdk.readthedocs.io/>`_
