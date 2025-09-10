#!/bin/bash

# Production deployment script for trading platform
# Run this on the swarm manager after setting up the cluster

set -e

echo "🚀 Deploying Trading Platform to Production"

# Check if we're on a swarm manager
if ! docker node ls &> /dev/null; then
    echo "❌ Error: This script must be run on a Docker Swarm manager node"
    exit 1
fi

# Check if all required files exist
required_files=(
    "docker-compose.prod.yml"
    "nginx/nginx.conf"
    "migrations/001_initial_schema.sql"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file $file not found"
        exit 1
    fi
done

# Create secrets if they don't exist
echo "🔑 Setting up secrets..."

if ! docker secret ls | grep -q trading_postgres_password; then
    echo "Enter PostgreSQL password for production:"
    read -s postgres_password
    echo "$postgres_password" | docker secret create trading_postgres_password -
    echo "✅ PostgreSQL password secret created"
else
    echo "✅ PostgreSQL password secret already exists"
fi

# Check node labels
echo "🏷️  Checking node labels..."
database_nodes=$(docker node ls --filter "label=role=database" -q | wc -l)
api_nodes=$(docker node ls --filter "label=role=api" -q | wc -l)

if [ "$database_nodes" -eq 0 ]; then
    echo "⚠️  No nodes labeled with role=database found"
    echo "   Run: docker node update --label-add role=database <node-id>"
fi

if [ "$api_nodes" -eq 0 ]; then
    echo "⚠️  No nodes labeled with role=api found"
    echo "   Run: docker node update --label-add role=api <node-id>"
fi

if [ "$database_nodes" -eq 0 ] || [ "$api_nodes" -eq 0 ]; then
    echo "❌ Please label your nodes before deploying"
    echo ""
    echo "📋 Current nodes:"
    docker node ls
    exit 1
fi

# Create networks if they don't exist
echo "🌐 Creating networks..."
docker network create --driver overlay frontend || true
docker network create --driver overlay backend || true

# Deploy the stack
echo "🚀 Deploying trading platform stack..."
docker stack deploy -c docker-compose.prod.yml trading

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📊 Monitor deployment progress:"
echo "   docker stack services trading"
echo "   docker service logs trading_postgres -f"
echo ""
echo "🔍 Check service status:"
echo "   docker stack ps trading"
echo ""
echo "🌐 Access points (after services are ready):"
echo "   Frontend: https://yourdomain.com"
echo "   API: https://api.yourdomain.com"
echo "   Monitoring: http://<manager-ip>:9090 (Prometheus)"
echo "   Grafana: http://<manager-ip>:3001 (admin/admin123)"
echo ""
echo "⚠️  Don't forget to:"
echo "1. Update DNS records to point to your servers"
echo "2. Add SSL certificates to nginx/ssl/"
echo "3. Update domain names in nginx.conf and docker-compose.prod.yml"