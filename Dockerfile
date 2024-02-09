# container for CI/CD or development - NOT meant to be an extensible Docker image with the installed package

ARG REPO=code.ornl.gov:4567/rse/images/

# use this stage for development
FROM ${REPO}python:3.11-slim as minimal

ARG REGISTRY_PASSWORD=
ARG INTERSECT_REGISTRY=https://gitlab-ci-token:${REGISTRY_PASSWORD}@code.ornl.gov/api/v4/groups/5609/-/packages/pypi/simple

WORKDIR /app
# add minimal files needed for build
COPY pyproject.toml README.md ./
RUN pip install -i ${INTERSECT_REGISTRY} .

# use this stage in CI/CD, not useful in development
FROM minimal as complete
COPY . .
RUN pip install .

# set CMD at container runtime
