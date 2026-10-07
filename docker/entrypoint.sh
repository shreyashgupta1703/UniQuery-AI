#!/usr/bin/env sh
set -e

# Seed the sample corpus once, unless disabled or already populated. This makes
# `docker compose up` land on a working, pre-populated UI.
if [ "${LOCALRAG_SEED:-1}" = "1" ]; then
  echo "localrag: seeding sample documents..."
  python -m scripts.seed || echo "localrag: seed skipped (continuing)"
fi

exec "$@"
