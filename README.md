# neeter-active-learning

## Requirements

- Python >= 3.9
- a Gitlab Personal Access Token (see https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html for instructions)

## Installing

`pip install -i https://user:${REGISTRY_PASSWORD}@code.ornl.gov/api/v4/groups/5609/-/packages/pypi/simple -e .`

## CLI arguments / environment variables

NOTE: this applies to any Service and Client in this repository.

CLI arg `--config` or environment variable `NEETER_CONFIG_FILE` should be a path to a valid JSON configuration. If neither value is set, it will default to `local-conf.json` .

- `local-conf.json` - If you set up the infrastructure locally via `docker compose up`, use this config file.

## Docker

To build:

`docker build --build-arg REGISTRY_PASSWORD=<Gitlab PAT> -t neeter-image .`

To run the service:

`docker run --rm -it neeter-image python scripts/launch_service.py`

To run the client, select one of the following:

- Automatic run: `docker run --rm -it neeter-image python scripts/automated_client.py`
- Manual run: `docker run --rm -it neeter-image python scripts/manual_client.py`

## running infrastructure locally

You can use `docker compose up -d` to automatically spin up a broker instance locally.

To remove the infrastructure containers: `docker compose down -v`

Note that if you are also running the Client/Service scripts in Docker, you will need to make sure that you add in appropriate `host` properties inside of `local-conf.json` (the host names are the names of the services in `docker-compose.yml`, instead of `127.0.0.1`):

```json
{
  "data_stores": {
    "minio": [
      {
        "username": "AKIAIOSFODNN7EXAMPLE",
        "password": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "host": "minio",
        "port": 9000
      }
    ]
  },
  "brokers": [
    {
      "username": "intersect_username",
      "password": "intersect_password",
      "host": "broker",
      "port": 1883,
      "protocol": "mqtt3.1.1"
    }
  ]
}

```
