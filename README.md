# neeter-active-learning

## Requirements

- Python >= 3.9
- a Gitlab Personal Access Token (see https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html for instructions)

## Installing

`pip install -i https://user:${REGISTRY_PASSWORD}@code.ornl.gov/api/v4/groups/5609/-/packages/pypi/simple -e .`

## Docker

To build:

`docker build --build-arg REGISTRY_PASSWORD=<Gitlab PAT> -t neeter-image .`

To run the service:

`docker run --rm -it neeter-image python -m neeter_active_learning.active_learning_service`

To run the client:

`docker run --rm -it neeter-image python -m neeter_active_learning.active_learning_client`
