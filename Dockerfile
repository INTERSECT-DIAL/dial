# container for CI/CD or development - NOT meant to be an extensible Docker image with the installed package

ARG REPO=

# use this stage for development
FROM ${REPO}python:3.11-slim as minimal


WORKDIR /app
# add minimal files needed for build
COPY pyproject.toml README.md ./
RUN pip install .

# use this stage in CI/CD, not useful in development
FROM minimal as complete
COPY . .
RUN pip install .

# set CMD at container runtime
