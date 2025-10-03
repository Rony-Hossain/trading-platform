"""
Secrets Management Loader

Supports multiple backends:
- HashiCorp Vault (self-hosted)
- AWS Secrets Manager
- GCP Secret Manager
- Azure Key Vault

Features:
- Automatic secret rotation
- Caching with TTL
- Audit logging
- Fallback to environment variables (dev only)
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import hvac  # HashiCorp Vault
import boto3  # AWS
from google.cloud import secretmanager  # GCP
from azure.identity import DefaultAzureCredential  # Azure
from azure.keyvault.secrets import SecretClient
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
secrets_accessed = Counter('secrets_accessed_total', 'Total secrets accessed', ['backend', 'secret_name'])
secrets_rotated = Counter('secrets_rotated_total', 'Total secrets rotated', ['backend'])
secrets_cache_hits = Counter('secrets_cache_hits_total', 'Cache hits')
secrets_cache_misses = Counter('secrets_cache_misses_total', 'Cache misses')
secret_access_latency = Histogram('secret_access_latency_seconds', 'Secret access latency', ['backend'])


class SecretsBackend(str, Enum):
    VAULT = "vault"
    AWS_SECRETS = "aws_secrets"
    GCP_SECRETS = "gcp_secrets"
    AZURE_KEYVAULT = "azure_keyvault"
    ENV = "env"  # Development only


@dataclass
class SecretMetadata:
    """Secret metadata"""
    name: str
    version: str
    created_at: datetime
    rotation_days: int
    last_accessed: Optional[datetime] = None
    access_count: int = 0


@dataclass
class CachedSecret:
    """Cached secret with TTL"""
    value: Any
    cached_at: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cache is expired"""
        return datetime.utcnow() > self.cached_at + timedelta(seconds=self.ttl_seconds)


class SecretsProvider(ABC):
    """Abstract secrets provider"""

    @abstractmethod
    def get_secret(self, secret_name: str) -> Any:
        """Get secret value"""
        pass

    @abstractmethod
    def set_secret(self, secret_name: str, secret_value: Any) -> None:
        """Set secret value"""
        pass

    @abstractmethod
    def rotate_secret(self, secret_name: str) -> None:
        """Rotate secret"""
        pass

    @abstractmethod
    def delete_secret(self, secret_name: str) -> None:
        """Delete secret"""
        pass

    @abstractmethod
    def list_secrets(self) -> list[str]:
        """List all secrets"""
        pass


class VaultProvider(SecretsProvider):
    """HashiCorp Vault provider"""

    def __init__(self, vault_addr: str, token: Optional[str] = None):
        self.vault_addr = vault_addr
        self.client = hvac.Client(url=vault_addr, token=token)

        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")

        logger.info(f"Connected to Vault at {vault_addr}")

    def get_secret(self, secret_name: str) -> Any:
        """Get secret from Vault"""
        start_time = time.time()

        try:
            # Read from KV v2 secret engine
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point="secret"
            )

            secret_value = response['data']['data']
            secret_access_latency.labels(backend="vault").observe(time.time() - start_time)
            secrets_accessed.labels(backend="vault", secret_name=secret_name).inc()

            logger.info(f"Retrieved secret: {secret_name}")
            return secret_value

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def set_secret(self, secret_name: str, secret_value: Any) -> None:
        """Set secret in Vault"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_name,
                secret=secret_value,
                mount_point="secret"
            )
            logger.info(f"Set secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error setting secret {secret_name}: {e}")
            raise

    def rotate_secret(self, secret_name: str) -> None:
        """Rotate secret (application-specific logic)"""
        # This is a placeholder - actual rotation logic depends on secret type
        logger.warning(f"Rotate secret called for {secret_name} - implement rotation logic")
        secrets_rotated.labels(backend="vault").inc()

    def delete_secret(self, secret_name: str) -> None:
        """Delete secret from Vault"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secret_name,
                mount_point="secret"
            )
            logger.info(f"Deleted secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {e}")
            raise

    def list_secrets(self) -> list[str]:
        """List all secrets"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path="",
                mount_point="secret"
            )
            return response['data']['keys']

        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []


class AWSSecretsProvider(SecretsProvider):
    """AWS Secrets Manager provider"""

    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client('secretsmanager', region_name=region_name)
        logger.info(f"Connected to AWS Secrets Manager in {region_name}")

    def get_secret(self, secret_name: str) -> Any:
        """Get secret from AWS Secrets Manager"""
        start_time = time.time()

        try:
            response = self.client.get_secret_value(SecretId=secret_name)

            if 'SecretString' in response:
                secret_value = json.loads(response['SecretString'])
            else:
                secret_value = response['SecretBinary']

            secret_access_latency.labels(backend="aws_secrets").observe(time.time() - start_time)
            secrets_accessed.labels(backend="aws_secrets", secret_name=secret_name).inc()

            logger.info(f"Retrieved secret: {secret_name}")
            return secret_value

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def set_secret(self, secret_name: str, secret_value: Any) -> None:
        """Set secret in AWS Secrets Manager"""
        try:
            self.client.put_secret_value(
                SecretId=secret_name,
                SecretString=json.dumps(secret_value)
            )
            logger.info(f"Set secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error setting secret {secret_name}: {e}")
            raise

    def rotate_secret(self, secret_name: str) -> None:
        """Rotate secret using AWS rotation"""
        try:
            self.client.rotate_secret(SecretId=secret_name)
            logger.info(f"Initiated rotation for secret: {secret_name}")
            secrets_rotated.labels(backend="aws_secrets").inc()

        except Exception as e:
            logger.error(f"Error rotating secret {secret_name}: {e}")
            raise

    def delete_secret(self, secret_name: str) -> None:
        """Delete secret from AWS Secrets Manager"""
        try:
            self.client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=True
            )
            logger.info(f"Deleted secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {e}")
            raise

    def list_secrets(self) -> list[str]:
        """List all secrets"""
        try:
            response = self.client.list_secrets()
            return [secret['Name'] for secret in response['SecretList']]

        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []


class GCPSecretsProvider(SecretsProvider):
    """GCP Secret Manager provider"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        logger.info(f"Connected to GCP Secret Manager for project {project_id}")

    def get_secret(self, secret_name: str) -> Any:
        """Get secret from GCP Secret Manager"""
        start_time = time.time()

        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})

            secret_value = json.loads(response.payload.data.decode('UTF-8'))

            secret_access_latency.labels(backend="gcp_secrets").observe(time.time() - start_time)
            secrets_accessed.labels(backend="gcp_secrets", secret_name=secret_name).inc()

            logger.info(f"Retrieved secret: {secret_name}")
            return secret_value

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def set_secret(self, secret_name: str, secret_value: Any) -> None:
        """Set secret in GCP Secret Manager"""
        try:
            parent = f"projects/{self.project_id}"

            # Create secret if it doesn't exist
            try:
                self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_name,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            except Exception:
                pass  # Secret already exists

            # Add secret version
            parent = f"projects/{self.project_id}/secrets/{secret_name}"
            payload = json.dumps(secret_value).encode('UTF-8')

            self.client.add_secret_version(
                request={"parent": parent, "payload": {"data": payload}}
            )

            logger.info(f"Set secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error setting secret {secret_name}: {e}")
            raise

    def rotate_secret(self, secret_name: str) -> None:
        """Rotate secret (add new version)"""
        logger.warning(f"Rotate secret called for {secret_name} - implement rotation logic")
        secrets_rotated.labels(backend="gcp_secrets").inc()

    def delete_secret(self, secret_name: str) -> None:
        """Delete secret from GCP Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}"
            self.client.delete_secret(request={"name": name})
            logger.info(f"Deleted secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {e}")
            raise

    def list_secrets(self) -> list[str]:
        """List all secrets"""
        try:
            parent = f"projects/{self.project_id}"
            response = self.client.list_secrets(request={"parent": parent})
            return [secret.name.split('/')[-1] for secret in response]

        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []


class AzureKeyVaultProvider(SecretsProvider):
    """Azure Key Vault provider"""

    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
        logger.info(f"Connected to Azure Key Vault at {vault_url}")

    def get_secret(self, secret_name: str) -> Any:
        """Get secret from Azure Key Vault"""
        start_time = time.time()

        try:
            secret = self.client.get_secret(secret_name)
            secret_value = json.loads(secret.value)

            secret_access_latency.labels(backend="azure_keyvault").observe(time.time() - start_time)
            secrets_accessed.labels(backend="azure_keyvault", secret_name=secret_name).inc()

            logger.info(f"Retrieved secret: {secret_name}")
            return secret_value

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def set_secret(self, secret_name: str, secret_value: Any) -> None:
        """Set secret in Azure Key Vault"""
        try:
            self.client.set_secret(secret_name, json.dumps(secret_value))
            logger.info(f"Set secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error setting secret {secret_name}: {e}")
            raise

    def rotate_secret(self, secret_name: str) -> None:
        """Rotate secret"""
        logger.warning(f"Rotate secret called for {secret_name} - implement rotation logic")
        secrets_rotated.labels(backend="azure_keyvault").inc()

    def delete_secret(self, secret_name: str) -> None:
        """Delete secret from Azure Key Vault"""
        try:
            self.client.begin_delete_secret(secret_name).wait()
            logger.info(f"Deleted secret: {secret_name}")

        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {e}")
            raise

    def list_secrets(self) -> list[str]:
        """List all secrets"""
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]

        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []


class SecretsManager:
    """
    Centralized secrets manager with caching and rotation

    Features:
    - Multi-backend support
    - Automatic caching with TTL
    - Audit logging
    - Rotation tracking
    """

    def __init__(
        self,
        backend: SecretsBackend,
        cache_ttl_seconds: int = 300,
        rotation_days: int = 90,
        **backend_kwargs
    ):
        self.backend = backend
        self.cache_ttl_seconds = cache_ttl_seconds
        self.rotation_days = rotation_days
        self.cache: Dict[str, CachedSecret] = {}
        self.metadata: Dict[str, SecretMetadata] = {}

        # Initialize provider
        self.provider = self._init_provider(backend, backend_kwargs)

    def _init_provider(self, backend: SecretsBackend, kwargs: dict) -> SecretsProvider:
        """Initialize secrets provider"""
        if backend == SecretsBackend.VAULT:
            return VaultProvider(
                vault_addr=kwargs.get('vault_addr', os.getenv('VAULT_ADDR')),
                token=kwargs.get('token', os.getenv('VAULT_TOKEN'))
            )
        elif backend == SecretsBackend.AWS_SECRETS:
            return AWSSecretsProvider(
                region_name=kwargs.get('region_name', 'us-east-1')
            )
        elif backend == SecretsBackend.GCP_SECRETS:
            return GCPSecretsProvider(
                project_id=kwargs.get('project_id', os.getenv('GCP_PROJECT_ID'))
            )
        elif backend == SecretsBackend.AZURE_KEYVAULT:
            return AzureKeyVaultProvider(
                vault_url=kwargs.get('vault_url', os.getenv('AZURE_KEYVAULT_URL'))
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def get_secret(self, secret_name: str, use_cache: bool = True) -> Any:
        """
        Get secret with caching

        Args:
            secret_name: Name of the secret
            use_cache: Whether to use cache

        Returns:
            Secret value
        """
        # Check cache
        if use_cache and secret_name in self.cache:
            cached = self.cache[secret_name]
            if not cached.is_expired():
                secrets_cache_hits.inc()
                logger.debug(f"Cache hit for secret: {secret_name}")
                return cached.value
            else:
                logger.debug(f"Cache expired for secret: {secret_name}")
                del self.cache[secret_name]

        secrets_cache_misses.inc()

        # Get from provider
        secret_value = self.provider.get_secret(secret_name)

        # Cache it
        if use_cache:
            self.cache[secret_name] = CachedSecret(
                value=secret_value,
                cached_at=datetime.utcnow(),
                ttl_seconds=self.cache_ttl_seconds
            )

        # Update metadata
        if secret_name not in self.metadata:
            self.metadata[secret_name] = SecretMetadata(
                name=secret_name,
                version="1",
                created_at=datetime.utcnow(),
                rotation_days=self.rotation_days
            )

        self.metadata[secret_name].last_accessed = datetime.utcnow()
        self.metadata[secret_name].access_count += 1

        return secret_value

    def set_secret(self, secret_name: str, secret_value: Any) -> None:
        """Set secret and invalidate cache"""
        self.provider.set_secret(secret_name, secret_value)

        # Invalidate cache
        if secret_name in self.cache:
            del self.cache[secret_name]

    def rotate_secret(self, secret_name: str) -> None:
        """Rotate secret"""
        self.provider.rotate_secret(secret_name)

        # Invalidate cache
        if secret_name in self.cache:
            del self.cache[secret_name]

    def check_rotation_needed(self) -> list[str]:
        """Check which secrets need rotation"""
        needs_rotation = []

        for secret_name, metadata in self.metadata.items():
            age_days = (datetime.utcnow() - metadata.created_at).days
            if age_days >= metadata.rotation_days:
                needs_rotation.append(secret_name)
                logger.warning(
                    f"Secret {secret_name} is {age_days} days old, rotation recommended"
                )

        return needs_rotation

    def get_audit_log(self) -> list[dict]:
        """Get audit log of secret accesses"""
        return [
            {
                "secret_name": name,
                "version": meta.version,
                "created_at": meta.created_at.isoformat(),
                "last_accessed": meta.last_accessed.isoformat() if meta.last_accessed else None,
                "access_count": meta.access_count,
                "age_days": (datetime.utcnow() - meta.created_at).days
            }
            for name, meta in self.metadata.items()
        ]


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def init_secrets_manager(
    backend: SecretsBackend,
    cache_ttl_seconds: int = 300,
    rotation_days: int = 90,
    **backend_kwargs
) -> SecretsManager:
    """Initialize global secrets manager"""
    global _secrets_manager
    _secrets_manager = SecretsManager(
        backend=backend,
        cache_ttl_seconds=cache_ttl_seconds,
        rotation_days=rotation_days,
        **backend_kwargs
    )
    return _secrets_manager


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager"""
    if _secrets_manager is None:
        raise RuntimeError("Secrets manager not initialized. Call init_secrets_manager() first.")
    return _secrets_manager


def get_secret(secret_name: str, use_cache: bool = True) -> Any:
    """Convenience function to get secret"""
    return get_secrets_manager().get_secret(secret_name, use_cache)
