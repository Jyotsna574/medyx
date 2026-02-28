#!/bin/bash
# Resolve "local changes would be overwritten by merge" on Param Shakti.
# Run from project root: bash scripts/git_sync_cluster.sh
#
# Discards local llm_factory changes and pulls latest (remote has all fixes).

set -e
cd "$(dirname "$0")/.."

echo "Discarding local changes to infrastructure/llm_factory.py..."
git checkout -- infrastructure/llm_factory.py 2>/dev/null || true

echo "Pulling latest from master..."
git pull origin master

echo "Done. Run: python test_dummy_run.py  # optional verify"
