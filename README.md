[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14872254.svg)](https://doi.org/10.5281/zenodo.14872254)

# Dial

## Requirements

- Python >= 3.10

## Installing (Non-developers)

To install intersect-sdk from PyPI:

`pip install intersect-sdk`

To install Dial from source, git clone this repository and run the following from the root directory (that is, within the neeter-active-learning directory):

`pip install .`

Alternatively, both intersect-sdk and Dial may be installed with the following:

`pip install -e .`

## Installing (developers)

Install PDM ([link](https://pdm-project.org/en/latest/#installation)) and then run:

`pdm install`

We use PDM as a dependency manager and intend on using the dependencies listed in `pdm.lock` in a production environment.

We use ruff to lint/format; if using the PDM workflow, `pre-commit` will automatically fail the commit if there are linting/formatting errors.

To format:

`ruff format`

To run linter and automatically fix errors:

`ruff check --fix`

## running infrastructure locally

You can use `docker compose up -d` to automatically spin up a broker instance locally.

To remove the infrastructure containers: `docker compose down -v`

Note that if you are also running the Client/Service scripts in Docker, you will need to make sure that you add in appropriate `host` properties inside of `local-conf.json` (the host names are the names of the services in `docker-compose.yml`, instead of `127.0.0.1`):

```json
{
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

## Running

To run the service: `python scripts/launch_service.py`

In a separate terminal, you can run one of the following clients:
  - Automatic: `python scripts/automated_client.py`
  - Manual: `python scripts/manual_client.py`

CLI arg `--config` or environment variable `DIAL_CONFIG_FILE` should be a path to a valid JSON configuration. If neither value is set, it will default to `local-conf.json` .

- `local-conf.json` - If you set up the infrastructure locally via `docker compose up`, use this config file.

## Docker

To build:

`docker build -t dial-image .`

To run the service:

`docker run --rm -it dial-image -e DIAL_CONFIG_FILE=/app/config.json -v path-to-your-config.json:/app/config.json python scripts/launch_service.py`

To run the client, select one of the following:

- Automatic run: `docker run --rm -it -e DIAL_CONFIG_FILE=/app/config.json -v path-to-your-config.json:/app/config.json dial-image python scripts/automated_client.py`
- Manual run: `docker run --rm -it -e DIAL_CONFIG_FILE=/app/config.json -v path-to-your-config.json:/app/config.json dial-image python scripts/manual_client.py`

## Testing

You will need `pytest` installed to run the tests, it should be automatically included in your virtual environment if using the PDM workflow.

`pdm run test-all`
