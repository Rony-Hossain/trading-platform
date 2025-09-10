#!/bin/bash

# Database initialization script for trading platform
# This script runs database migrations and sets up initial data

set -e

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-trading_db}
DB_USER=${DB_USER:-trading_user}
DB_PASSWORD=${DB_PASSWORD:-trading_pass}

MIGRATIONS_DIR=${MIGRATIONS_DIR:-./migrations}

echo "Initializing database: $DB_NAME"

# Wait for database to be ready
echo "Waiting for database to be ready..."
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; do
  echo "Database is unavailable - sleeping"
  sleep 2
done

echo "Database is ready!"

# Run migrations
echo "Running database migrations..."
for migration in "$MIGRATIONS_DIR"/*.sql; do
  if [ -f "$migration" ]; then
    echo "Applying migration: $(basename "$migration")"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration"
  fi
done

# Insert seed data if specified
if [ "$SEED_DATA" = "true" ]; then
  echo "Inserting seed data..."
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$MIGRATIONS_DIR/seed_data.sql"
fi

echo "Database initialization completed successfully!"