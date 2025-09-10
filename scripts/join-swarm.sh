#!/bin/bash

# Script to join the second Ubuntu server to Docker Swarm
# Usage: ./join-swarm.sh worker <manager-ip> <token>
# Or: curl -fsSL <script-url> | bash -s worker <manager-ip> <token>

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <role> <manager-ip> <token>"
    echo "Example: $0 worker 192.168.1.100 SWMTKN-1-xxxx"
    exit 1
fi

ROLE=$1
MANAGER_IP=$2
TOKEN=$3

echo "🚀 Joining Docker Swarm as $ROLE"

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
    
    echo "⚠️  Docker installed. You need to log out and back in for group changes to take effect"
    echo "   Then run this script again with the same parameters"
    exit 1
fi

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Join the swarm
echo "🔗 Joining swarm cluster..."
docker swarm join --token $TOKEN $MANAGER_IP:2377

echo ""
echo "✅ Successfully joined the swarm!"
echo ""

if [ "$ROLE" = "worker" ]; then
    echo "📋 To label this node as 'api' role, run on the manager node:"
    echo "   NODE_ID=\$(docker node ls -q -f name=\$(hostname))"
    echo "   docker node update --label-add role=api \$NODE_ID"
fi

echo ""
echo "🔍 Verify the setup by running on the manager:"
echo "   docker node ls"