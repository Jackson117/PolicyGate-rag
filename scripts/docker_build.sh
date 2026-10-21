#!/usr/bin/env bash
# Build the image with policygate at the tag pyproject.toml pins.
#
#   scripts/docker_build.sh [path/to/PolicyGate]
#
# The tag is read from pyproject.toml and exported with git archive, so the image
# gets exactly that commit, never PolicyGate's working tree.
set -euo pipefail
cd "$(dirname "$0")/.."
policygate=${1:-../PolicyGate}
tag=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["tool"]["uv"]["sources"]["policygate"]["tag"])')
mkdir -p vendor
git -C "$policygate" archive --format=tar "$tag" > vendor/policygate.tar
echo "policygate $tag ($(git -C "$policygate" rev-parse --short "$tag^{commit}"))"
docker build -t policygate-rag .
