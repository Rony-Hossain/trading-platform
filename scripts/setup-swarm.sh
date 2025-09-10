#!/bin/bash

# Docker Swarm setup script for trading platform
# Run this on your first Ubuntu server (will become the manager)

set -e

echo "🚀 Setting up Docker Swarm for Trading Platform"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    
    echo "⚠️  You need to log out and back in for Docker group changes to take effect"
    echo "   Then run this script again"
    exit 1
fi

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Initialize Swarm (this server becomes manager)
echo "🔧 Initializing Docker Swarm..."
MANAGER_IP=$(hostname -I | awk '{print $1}')
docker swarm init --advertise-addr $MANAGER_IP

# Get join tokens
WORKER_TOKEN=$(docker swarm join-token worker -q)
MANAGER_TOKEN=$(docker swarm join-token manager -q)

echo ""
echo "✅ Swarm initialized successfully!"
echo ""
echo "📋 To join your second server as a worker, run on Ubuntu Server 2:"
echo "   curl -fsSL https://raw.githubusercontent.com/your-repo/trading-platform/main/scripts/join-swarm.sh | bash -s worker $MANAGER_IP $WORKER_TOKEN"
echo ""
echo "🏷️  Run this command to label this node as 'database' role:"
echo "   docker node update --label-add role=database \$(docker node ls -q -f role=manager)"
echo ""

# Label this node as database role
NODE_ID=$(docker node ls -q -f role=manager)
docker node update --label-add role=database $NODE_ID

echo "🏷️  This node has been labeled as 'database' role"
echo ""
echo "🔑 Save these tokens securely:"
echo "   Worker Token: $WORKER_TOKEN"
echo "   Manager Token: $MANAGER_TOKEN"
echo "   Manager IP: $MANAGER_IP"
echo ""
echo "📝 Next steps:"
echo "1. Join your second server to the swarm"
echo "2. Label the second server: docker node update --label-add role=api <node-id>"
echo "3. Create secrets: docker secret create trading_postgres_password -"
echo "4. Deploy stack: docker stack deploy -c docker-compose.prod.yml trading"