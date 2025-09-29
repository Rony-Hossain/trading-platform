#!/bin/bash

# Start monitoring stack for trading platform
# Usage: ./scripts/start-monitoring.sh

set -e

echo "🚀 Starting Trading Platform Monitoring Stack..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create monitoring directories if they don't exist
mkdir -p monitoring/prometheus/rules
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards  
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/alertmanager

echo "📁 Monitoring directories created"

# Start the monitoring stack
echo "🐳 Starting containers..."
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=(
    "prometheus:9090"
    "grafana:3001" 
    "alertmanager:9093"
    "node-exporter:9100"
    "cadvisor:8080"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -f http://localhost:$port >/dev/null 2>&1; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
    fi
done

echo ""
echo "🎉 Monitoring stack started successfully!"
echo ""
echo "📊 Access URLs:"
echo "  • Prometheus: http://localhost:9090"
echo "  • Grafana:    http://localhost:3001 (admin/trading123)"
echo "  • Alertmanager: http://localhost:9093"
echo "  • Node Exporter: http://localhost:9100"
echo "  • cAdvisor:   http://localhost:8080"
echo ""
echo "📈 Next steps:"
echo "  1. Import Grafana dashboards from monitoring/grafana/dashboards/"
echo "  2. Configure Slack/email notifications in monitoring/alertmanager/alertmanager.yml"
echo "  3. Set up custom metrics in your services"
echo ""
echo "🛑 To stop: docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down"