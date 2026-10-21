# The study's pinned environment: tests, the export and the figures run here
# with no network and no spend; episodes run here too, given API keys.
#
#   scripts/docker_build.sh                                          # builds policygate-rag
#   docker run --rm policygate-rag                                   # the test suite
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/docs/figures:/app/docs/figures" \
#       policygate-rag python scripts/figures.py                     # redraw the figures
#   docker run --rm --env-file .env -v "$PWD/runs:/app/runs" policygate-rag \
#       python -m policygate_rag.runner ...                          # episodes: costs money
#
# policygate is installed from an archive of its pinned tag (git archive, made by
# scripts/docker_build.sh), never its working tree, exactly as pyproject.toml pins
# it. tau2 comes from the vendored checkout at the same pin (third_party/PIN.md).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

WORKDIR /app

# Locked third-party dependencies first: this layer changes only with uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-hashes --no-emit-project --group dev \
        --no-emit-package policygate --no-emit-package tau2 > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt

# tau2 at the pin, without its dependencies (they are in the lock above).
COPY third_party/tau2-bench/ ./third_party/tau2-bench/
RUN uv pip install --system --no-cache --no-deps "./third_party/tau2-bench[knowledge,experiments]"
ENV TAU2_DATA_DIR=/app/third_party/tau2-bench/data

# The harness at the tag pyproject.toml pins, from git archive: uncommitted
# changes in PolicyGate never reach this image.
COPY vendor/policygate.tar /tmp/policygate.tar
RUN mkdir /opt/policygate && tar -xf /tmp/policygate.tar -C /opt/policygate \
    && uv pip install --system --no-cache --no-deps /opt/policygate \
    && rm /tmp/policygate.tar

COPY src/ ./src/
COPY tests/ ./tests/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY results/ ./results/
COPY index_cache/ ./index_cache/
COPY runs/ledger.json ./runs/ledger.json
RUN uv pip install --system --no-cache --no-deps -e .

RUN useradd --create-home --uid 10001 rag && chown -R rag /app
USER rag

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
