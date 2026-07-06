[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14872254.svg)](https://doi.org/10.5281/zenodo.14872254)

# Dial

This repository is a uv workspace with two members:

- `packages/intersect_dial_dataclass` (package of the core DIAL types, available on PyPI)
- `services/dial_service` (DIAL service runtime package, available as a Docker container and a Helm chart)

## Requirements

- Python >= 3.10

## Installing (Non-developers)

When writing a Client, most users will only need to install the `intersect-sdk` and `intersect-dial-dataclass` .

To install all dependencies from PyPI:

```
pip install intersect-sdk intersect-dial-dataclass
```

To install the publishable dataclass package from source, run:

`pip install ./packages/dial_dataclass`

For more advanced users, to install the service package from source, run:

`pip install -e ./packages/dial_dataclass -e ./services/dial_service`

## Installing (developers)

Install UV ([link](https://docs.astral.sh/uv/getting-started/installation/)) and then run:

```
uv sync --all-groups --all-extras --all-packages
pre-commit install
```

We use UV as a dependency manager and intend on using the dependencies listed in `uv.lock` in a production environment.

We use ruff to lint/format; if using the UV workflow, `pre-commit` will automatically fail the commit if there are linting/formatting errors.

To format:

`ruff format`

To run linter and automatically fix errors:

`ruff check --fix`

## running infrastructure locally

You can use `docker compose -f docker-compose-dev.yml up` or to automatically spin up both a broker instance and a database instance locally.

If you also want to run DIAL inside the container, you can instead run `docker compose up`

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

You will need `pytest` installed to run the tests.

`pytest tests/`

## Building the documentation

You will need to make sure the `docs` optional dependency group is installed. Then you can run `sphinx build -W --keep-going docs/ docs/_build/` .

To quickly spin up the documentation web server on `http://localhost:8000` : `python -m http.server -d docs/_build/`

## Version management

- always bump `dial_service` whenever we want to tag a new version, and have the tag reference the `dial_service` version
- when bumping `intersect_dial_dataclass` version, bump it to match the `dial_service` version (okay to skip versions)
