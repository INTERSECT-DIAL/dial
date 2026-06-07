============
Client Guide
============

For general information on writing an INTERSECT-SDK Client, please see the `INTERSECT-SDK documentation <https://intersect-python-sdk.readthedocs.io/en/latest/core_concepts.html#client>`. This page will cover DIAL-specific concepts.

Basic capability setup: sending message
---------------------------------------

When you return the ``IntersectClientCallback`` object from the Client callback function, you will always have some common functionality:

.. codeblock:: python
  from intersect_sdk import IntersectClientCallback, IntersectDirectMessageParams, INTERSECT_RESPONSE_VALUE

  from dial_dataclass import (
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdate,
    DialWorkflowDatasetUpdates,
    DialInputSingle,
    DialInputMultiple,
    DialInputPredictions,
  )

  class CustomOrchestrator:
    # ...

    def message_callback_function(self, source: str, operation: str, has_error: bool, payload: INTERSECT_RESPONSE_VALUE):
      # ... assume we are ready to pass a message to DIAL
      # self.dial_SDK_destination is defined at configuration time
      # assume that dial_function_to_call is defined based on custom workflow logic
      # and the structure of dial_payload will be based off of dial_function_to_call

      match dial_function_to_call:
        case 'initialize_workflow':
          payload = DialWorkflowCreationParamsClient(
            dataset_x=[[1.0,2.0],[3.0,.40]
          )
        case 'update_workflow':
          payload = DialWorkflowDatasetUpdate(

          )
        case 'update_workflow_with_batch_data':
          payload = DialWorkflowDatasetUpdates(

          )
        case 'get_next_point':
          payload = DialInputSingle(

          )
        case 'get_next_points':
          payload = DialInputMultiple(

          )
        case 'get_surrogate_values':
          payload = DialInputPredictions(

          )

      return IntersectClientCallback(
        messages_to_send=[
          IntersectDirectMessageParams(
            destination=self.dial_SDK_destination,
            operation=f'dial.{dial_function_to_call}',
            payload=payload,
          )
        ]
      )

Now ``payload`` will be constructed based off of `dial_function_to_call`:

Basic capability setup: receiving message
-----------------------------------------

To validate the response from a message, you can use the dial_dataclasses types as well (be sure to first check the source and operation parameters provided to you in the callback function):

.. codeblock:: python
  from intersect_sdk import IntersectClientCallback, IntersectDirectMessageParams, INTERSECT_RESPONSE_VALUE

  from dial_dataclass import

  class CustomOrchestrator:
    # ...

    def message_callback_function(self, source: str, operation: str, has_error: bool, payload: INTERSECT_RESPONSE_VALUE):
      # ... assume we are ready to pass a message to DIAL
      # self.dial_SDK_destination is defined at configuration time
      # assume that dial_function_to_call is defined based on custom workflow logic
      # and the structure of dial_payload will be based off of dial_function_to_call

      return IntersectClientCallback(
        messages_to_send=[
          IntersectDirectMessageParams(
            destination=self.dial_SDK_destination,
            operation=f'dial.{dial_function_to_call}',
            payload=payload,
          )
        ]
      )


When writing an SDK Client, you can import the various input/output formats from `dial_dataclass` inside of your script to provide you type hints and runtime validation of your input parameters.

See the various ``scripts/_client.py`` files in the source code as quick examples on how to construct Clients.
