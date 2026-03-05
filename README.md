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

You can use `docker compose up -d -f docker-compose-dev.yml` or to automatically spin up both a broker instance and a database instance locally.

If you also want to run DIAL inside the container, you can instead run `docker compose up -d`

To remove the infrastructure containers: `docker compose down -v`; leave off the `-v` flag if you would like to persist the DB data.

## Running

To run the service: `python scripts/launch_service.py`

In a separate terminal, you can run one of the following clients:
  - Automatic: `python scripts/automated_client.py`
  - Manual: `python scripts/manual_client.py`

CLI arg `--config` or environment variable `DIAL_CONFIG_FILE` should be a path to a valid JSON configuration. If neither value is set, it will default to `local-conf.json` .

- `local-conf.json` - If you set up the infrastructure locally via `docker compose up`, use this config file.
- `local-conf-docker.json` - This config file should only be used if you are running DIAL in the Docker image as well.

## Docker

To build:

`docker build -t dial-image .`

To run the service:

`docker run --rm -it dial-image -e DIAL_CONFIG_FILE=/app/config.json -v path-to-your-config.json:/app/config.json python scripts/launch_service.py`

To run the client, select one of the following:

- Automatic run: `docker run --rm -it -e DIAL_CONFIG_FILE=/app/config.json -v path-to-your-config.json:/app/config.json dial-image python scripts/automated_client.py`
- Manual run: `docker run --rm -it -e DIAL_CONFIG_FILE=/app/config.json -v path-to-your-config.json:/app/config.json dial-image python scripts/manual_client.py`

## Kubernetes Deployment (Helm)

A Helm chart is available for deploying Dial to Kubernetes with MongoDB database support. The chart includes:

- Dial service deployment
- MongoDB subchart from bitnami

### Quick Start

```bash
cd charts/dial
helm dependency update
helm install dial . -n dial --create-namespace
```

For detailed Helm chart documentation, see:
- [charts/dial/README.md](charts/dial/README.md) - Comprehensive Helm documentation

### Common Helm Commands

With custom INTERSECT configuration:
```bash
helm install dial . -n dial --create-namespace -f values.yaml -f values.config.yaml
```

With NodePort service:
```bash
helm install dial . -n dial --create-namespace -f values.yaml -f values.nodePort.yaml
```

With external MongoDB:
```bash
helm install dial . -n dial --create-namespace \
  --set mongodb.enabled=false \
  --set externalMongoDB.connectionString="mongodb://user:pass@host:27017/dial"
```

## Testing

You will need `pytest` installed to run the tests, it should be automatically included in your virtual environment if using the PDM workflow.

`pdm run test-all`
