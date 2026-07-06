# see https://docs.astral.sh/uv/guides/integration/docker/#optimizations and https://www.joshkasuboski.com/posts/distroless-python-uv/

FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:debian-slim AS builder

ARG PYTHON_VERSION=3.12
# set to "0" to include dev dependencies, "1" to exclude them (default: "1")
ARG UV_NO_DEV="1"

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_NO_DEV=${UV_NO_DEV}

# TODO - remove git once we install Sable from PyPI
RUN apt update -y
RUN apt install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install ${PYTHON_VERSION}

WORKDIR /app

# Install (required) dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/intersect_dial_dataclass/pyproject.toml,target=packages/intersect_dial_dataclass/pyproject.toml \
    --mount=type=bind,source=services/dial_service/pyproject.toml,target=services/dial_service/pyproject.toml \
    uv sync --locked --package dial_service --no-install-workspace --no-editable

# Sync the project
COPY packages packages
COPY services services
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/intersect_dial_dataclass/pyproject.toml,target=packages/intersect_dial_dataclass/pyproject.toml \
    --mount=type=bind,source=services/dial_service/pyproject.toml,target=services/dial_service/pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --package dial_service --no-editable

FROM --platform=$BUILDPLATFORM gcr.io/distroless/cc:nonroot AS runner

COPY --from=builder --chown=app:app /python /python

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app scripts scripts

ENV PATH="/app/.venv/bin:$PATH"

# override CMD at container runtime if you want to execute the client, make sure that "client" group is present
CMD ["python", "scripts/launch_service.py"]
