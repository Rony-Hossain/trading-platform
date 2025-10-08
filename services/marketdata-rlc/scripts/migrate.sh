#!/usr/bin/env bash
set -euo pipefail

: "${PGURL:?Set PGURL, e.g. PGURL=postgres://user:pass@localhost:5432/db}"

echo "Applying migrations to ${PGURL}"
for file in "$(dirname "$0")/../migrations/"*.sql; do
  echo "==> Running ${file}"
  psql "${PGURL}" -v ON_ERROR_STOP=1 -f "${file}"
done
echo "Migrations complete."
