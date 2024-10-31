# container for CI/CD or development - NOT meant to be an extensible Docker image with the installed package

ARG REPO=

# use this stage for development
FROM ${REPO}python:3.12-slim

WORKDIR /app
# add minimal files needed for build
COPY pyproject.toml README.md ./
COPY . .
RUN pip install .

# override CMD at container runtime if you want to execute the client
CMD ["python3", "scripts/launch_service.py"]
