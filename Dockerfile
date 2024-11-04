ARG REPO=
# common environment variables
FROM ${REPO}python:3.12-slim AS environment

# set to non-empty string to skip dev dependencies
ARG PRODUCTION=
ENV PDM_VERSION=2.19.3 \
  PDM_HOME=/usr/local \
  PDM_CHECK_UPDATE=false \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PRODUCTION=${PRODUCTION:-}

# intermediate PDM build stage
FROM environment AS builder

WORKDIR /app
RUN apt update \
  && apt-get install --no-install-recommends -y \
    curl \
    make \
  && rm -rf /var/lib/apt/lists/* \
  && curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python -

# add minimal files needed for package install
COPY pyproject.toml README.md pdm.lock ./
COPY src src
RUN if [ -n "${PRODUCTION}" ]; then pdm sync --no-editable --prod; else pdm sync --no-editable --dev; fi

# main execution environment
FROM environment AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY scripts scripts
ENV PATH="/app/.venv/bin:$PATH"

# override CMD at container runtime if you want to execute the client, make sure that "client" group is present
CMD ["python", "scripts/launch_service.py"]

# add additional files for testing
FROM runtime AS test
COPY tests tests
