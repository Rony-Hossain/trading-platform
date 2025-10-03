#!/bin/bash
##
# Vault EC2 User Data Script
# Installs and configures HashiCorp Vault
##

set -e

# Variables from Terraform
VAULT_VERSION="${vault_version}"
KMS_KEY_ID="${kms_key_id}"
REGION="${region}"
ENVIRONMENT="${environment}"
STORAGE_TYPE="${storage_type}"
S3_BUCKET="${s3_bucket}"

# Logging
exec > >(tee /var/log/vault-bootstrap.log)
exec 2>&1

echo "=== Starting Vault Bootstrap ==="
echo "Vault Version: $VAULT_VERSION"
echo "Region: $REGION"
echo "Environment: $ENVIRONMENT"
echo "Storage Type: $STORAGE_TYPE"

# Update system
apt-get update
apt-get install -y wget unzip jq awscli

# Install Vault
echo "Installing Vault $VAULT_VERSION..."
cd /tmp
wget https://releases.hashicorp.com/vault/$${VAULT_VERSION}/vault_$${VAULT_VERSION}_linux_amd64.zip
unzip vault_$${VAULT_VERSION}_linux_amd64.zip
mv vault /usr/local/bin/
chmod +x /usr/local/bin/vault

# Verify installation
vault version

# Create vault user
useradd --system --home /etc/vault.d --shell /bin/false vault

# Create directories
mkdir -p /etc/vault.d
mkdir -p /opt/vault/data
mkdir -p /opt/vault/logs
chown -R vault:vault /etc/vault.d /opt/vault

# Get instance metadata
INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

echo "Instance IP: $INSTANCE_IP"
echo "Instance ID: $INSTANCE_ID"

# Create Vault configuration
cat > /etc/vault.d/vault.hcl <<EOF
# Vault Configuration for $ENVIRONMENT

storage "${storage_type}" {
%{ if storage_type == "raft" ~}
  path    = "/opt/vault/data"
  node_id = "$INSTANCE_ID"

  retry_join {
    auto_join = "provider=aws tag_key=Environment tag_value=$ENVIRONMENT"
    auto_join_scheme = "https"
  }
%{ endif ~}
%{ if storage_type == "s3" ~}
  bucket = "$S3_BUCKET"
  region = "$REGION"
%{ endif ~}
}

listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_disable   = false
  tls_cert_file = "/etc/vault.d/vault.crt"
  tls_key_file  = "/etc/vault.d/vault.key"
}

api_addr = "https://$INSTANCE_IP:8200"
cluster_addr = "https://$INSTANCE_IP:8201"

seal "awskms" {
  region     = "$REGION"
  kms_key_id = "$KMS_KEY_ID"
}

ui = true

log_level = "info"
EOF

# Generate self-signed certificate (replace with real cert in production)
echo "Generating self-signed certificate..."
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout /etc/vault.d/vault.key \
  -out /etc/vault.d/vault.crt \
  -subj "/CN=vault.$ENVIRONMENT" \
  -addext "subjectAltName=DNS:vault.$ENVIRONMENT,IP:$INSTANCE_IP"

chown vault:vault /etc/vault.d/vault.key /etc/vault.d/vault.crt
chmod 640 /etc/vault.d/vault.key

# Create systemd service
cat > /etc/systemd/system/vault.service <<EOF
[Unit]
Description=HashiCorp Vault
Documentation=https://www.vaultproject.io/docs/
Requires=network-online.target
After=network-online.target
ConditionFileNotEmpty=/etc/vault.d/vault.hcl

[Service]
Type=notify
User=vault
Group=vault
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=yes
PrivateDevices=yes
SecureBits=keep-caps
AmbientCapabilities=CAP_IPC_LOCK
CapabilityBoundingSet=CAP_SYSLOG CAP_IPC_LOCK
NoNewPrivileges=yes
ExecStart=/usr/local/bin/vault server -config=/etc/vault.d/vault.hcl
ExecReload=/bin/kill --signal HUP \$MAINPID
KillMode=process
KillSignal=SIGINT
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
StartLimitInterval=60
StartLimitBurst=3
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Vault
systemctl daemon-reload
systemctl enable vault
systemctl start vault

# Wait for Vault to start
sleep 10

# Check Vault status
export VAULT_ADDR="https://127.0.0.1:8200"
export VAULT_SKIP_VERIFY=1

vault status || true

echo "=== Vault Bootstrap Complete ==="
echo "Vault is running on https://$INSTANCE_IP:8200"
echo "Next steps:"
echo "1. Initialize Vault: vault operator init"
echo "2. Unseal will happen automatically via KMS"
echo "3. Configure secret engines and policies"
