from intersect_sdk import (
    HierarchyConfig,
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
)

from neeter_active_learning.active_learning_service import ActiveLearningServiceCapabilityImplementation

'''''
This launches the service.  Separate file due to module/import structure, plus we possibly want the capability to be a separate unit
'''''

if __name__ == '__main__':
    """
    step one: create configuration class, which handles validation - see the IntersectServiceConfig class documentation for more info

    In most cases, everything under from_config_file should come from a configuration file, command line arguments, or environment variables.
    """
    from_config_file = {
        'data_stores': {
            'minio': [
                {
                    'host': '',
                    'username': '',
                    'password': '',
                    'port': 0,
                },
            ],
        },
        'brokers': [
            {
                'host': '',
                'username': '',
                'password': '',
                'port': 0,
                'protocol': '',
            },
        ],
    }
    config = IntersectServiceConfig(
        hierarchy=HierarchyConfig(
            organization='hello-organization',
            facility='hello-facility',
            system='hello-system',
            subsystem='hello-subsystem',
            service='hello-service',
        ),
        schema_version='0.0.1',
        **from_config_file,
    )

    """
    step two - create your own capability implementation class.

    You have complete control over how you construct this class, as long as it has decorated functions with
    @intersect_message and @intersect_status, and that these functions are appropriately type-annotated.
    """
    capability = ActiveLearningServiceCapabilityImplementation()

    """
    step three - create service from both the configuration and your own capability
    """
    service = IntersectService(capability, config)

    """
    step four - start lifecycle loop. The only necessary parameter is your service.
    with certain applications (i.e. REST APIs) you'll want to integrate the service in the existing lifecycle,
    instead of using this one.
    In that case, just be sure to call service.startup() and service.shutdown() at appropriate stages.
    """
    default_intersect_lifecycle_loop(
        service,
    )

    """
    Note that the service will run forever until you explicitly kill the application (i.e. Ctrl+C)
    """
