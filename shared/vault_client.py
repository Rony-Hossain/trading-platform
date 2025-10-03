"""
HashiCorp Vault Client
Reusable client library for accessing secrets from Vault
"""
import logging
import os
from typing import Dict, Any, Optional
import hvac
from hvac.exceptions import VaultError, InvalidPath
import time

logger = logging.getLogger(__name__)


class VaultClient:
    """
    HashiCorp Vault client for secrets management

    Supports:
    - AppRole authentication
    - Token authentication
    - KV v2 secrets engine
    - Database dynamic credentials
    - Token renewal
    """

    def __init__(
        self,
        vault_addr: Optional[str] = None,
        role_id: Optional[str] = None,
        secret_id: Optional[str] = None,
        token: Optional[str] = None,
        namespace: str = "trading-platform",
        mount_point: str = "kv"
    ):
        """
        Initialize Vault client

        Args:
            vault_addr: Vault server address (default: from VAULT_ADDR env)
            role_id: AppRole role ID (default: from VAULT_ROLE_ID env)
            secret_id: AppRole secret ID (default: from VAULT_SECRET_ID env)
            token: Vault token (default: from VAULT_TOKEN env)
            namespace: Base path for secrets
            mount_point: KV mount point
        """
        self.vault_addr = vault_addr or os.getenv("VAULT_ADDR", "http://localhost:8200")
        self.role_id = role_id or os.getenv("VAULT_ROLE_ID")
        self.secret_id = secret_id or os.getenv("VAULT_SECRET_ID")
        self.token = token or os.getenv("VAULT_TOKEN")
        self.namespace = namespace
        self.mount_point = mount_point

        # Initialize client
        self.client = hvac.Client(url=self.vault_addr)

        # Authenticate
        self._authenticate()

        logger.info(f"Vault client initialized for {self.vault_addr}")

    def _authenticate(self):
        """Authenticate with Vault"""
        if self.token:
            # Token authentication
            self.client.token = self.token
            logger.info("Authenticated with Vault using token")

        elif self.role_id and self.secret_id:
            # AppRole authentication
            try:
                response = self.client.auth.approle.login(
                    role_id=self.role_id,
                    secret_id=self.secret_id
                )
                self.client.token = response['auth']['client_token']
                logger.info("Authenticated with Vault using AppRole")
            except VaultError as e:
                logger.error(f"AppRole authentication failed: {e}")
                raise

        else:
            raise ValueError(
                "No authentication method provided. "
                "Set either VAULT_TOKEN or VAULT_ROLE_ID + VAULT_SECRET_ID"
            )

        # Verify authentication
        if not self.client.is_authenticated():
            raise Exception("Vault authentication failed")

    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Get a secret from Vault

        Args:
            path: Secret path (relative to namespace)
            key: Specific key to retrieve (optional, returns all if None)

        Returns:
            Secret value or dictionary of values

        Example:
            # Get entire secret
            db_config = vault.get_secret("database/postgres")
            # Returns: {"host": "...", "port": 5432, "password": "..."}

            # Get specific key
            db_password = vault.get_secret("database/postgres", "password")
            # Returns: "..."
        """
        try:
            full_path = f"{self.namespace}/{path}"

            # Read secret from KV v2
            secret = self.client.secrets.kv.v2.read_secret_version(
                path=full_path,
                mount_point=self.mount_point
            )

            data = secret['data']['data']

            if key:
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in secret at {full_path}")
                return data[key]

            return data

        except InvalidPath:
            logger.error(f"Secret not found at path: {path}")
            raise

        except VaultError as e:
            logger.error(f"Error retrieving secret {path}: {e}")
            raise

    def set_secret(self, path: str, data: Dict[str, Any]):
        """
        Set a secret in Vault

        Args:
            path: Secret path (relative to namespace)
            data: Secret data

        Example:
            vault.set_secret("database/postgres", {
                "host": "localhost",
                "port": 5432,
                "username": "trading_user",
                "password": "secret123"
            })
        """
        try:
            full_path = f"{self.namespace}/{path}"

            # Write secret to KV v2
            self.client.secrets.kv.v2.create_or_update_secret(
                path=full_path,
                secret=data,
                mount_point=self.mount_point
            )

            logger.info(f"Secret written to {full_path}")

        except VaultError as e:
            logger.error(f"Error writing secret to {path}: {e}")
            raise

    def delete_secret(self, path: str):
        """
        Delete a secret from Vault

        Args:
            path: Secret path (relative to namespace)
        """
        try:
            full_path = f"{self.namespace}/{path}"

            # Delete secret
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=full_path,
                mount_point=self.mount_point
            )

            logger.info(f"Secret deleted from {full_path}")

        except VaultError as e:
            logger.error(f"Error deleting secret {path}: {e}")
            raise

    def get_database_credentials(
        self,
        role: str,
        mount_point: str = "database"
    ) -> Dict[str, str]:
        """
        Get dynamic database credentials from Vault

        Args:
            role: Database role name
            mount_point: Database secrets engine mount point

        Returns:
            Dictionary with 'username' and 'password'

        Example:
            creds = vault.get_database_credentials("readonly")
            # Returns: {"username": "v-approle-readonly-...", "password": "..."}
        """
        try:
            response = self.client.secrets.database.generate_credentials(
                name=role,
                mount_point=mount_point
            )

            return {
                'username': response['data']['username'],
                'password': response['data']['password'],
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration']
            }

        except VaultError as e:
            logger.error(f"Error generating database credentials for role {role}: {e}")
            raise

    def renew_token(self, increment: Optional[int] = None):
        """
        Renew the current Vault token

        Args:
            increment: Requested lease increment in seconds
        """
        try:
            if increment:
                self.client.auth.token.renew_self(increment=increment)
            else:
                self.client.auth.token.renew_self()

            logger.info("Vault token renewed successfully")

        except VaultError as e:
            logger.error(f"Error renewing token: {e}")
            raise

    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self.client.is_authenticated()

    def close(self):
        """Close Vault client connection"""
        # hvac client doesn't need explicit closing
        pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience functions
def get_secret(path: str, key: Optional[str] = None) -> Any:
    """
    Quick helper to get a secret

    Example:
        from shared.vault_client import get_secret

        db_password = get_secret("database/postgres", "password")
    """
    with VaultClient() as vault:
        return vault.get_secret(path, key)


def get_all_secrets(paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Get multiple secrets in one call

    Args:
        paths: Dictionary mapping names to secret paths

    Returns:
        Dictionary mapping names to secret values

    Example:
        secrets = get_all_secrets({
            "db_password": "database/postgres:password",
            "api_key": "apis/finnhub:key",
            "jwt_secret": "services/jwt:secret"
        })
    """
    with VaultClient() as vault:
        results = {}

        for name, path_spec in paths.items():
            if ':' in path_spec:
                path, key = path_spec.split(':', 1)
                results[name] = vault.get_secret(path, key)
            else:
                results[name] = vault.get_secret(path_spec)

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Basic usage
    try:
        vault = VaultClient()

        # Set a secret
        vault.set_secret("test/example", {
            "api_key": "test-key-123",
            "api_secret": "test-secret-456"
        })

        # Get entire secret
        secret = vault.get_secret("test/example")
        print(f"Full secret: {secret}")

        # Get specific key
        api_key = vault.get_secret("test/example", "api_key")
        print(f"API Key: {api_key}")

        # Delete secret
        vault.delete_secret("test/example")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Context manager
    try:
        with VaultClient() as vault:
            # Use vault client
            secret = vault.get_secret("database/postgres")
            print(f"Database config: {secret}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Quick helper
    try:
        password = get_secret("database/postgres", "password")
        print(f"Database password: {password}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Get multiple secrets
    try:
        secrets = get_all_secrets({
            "db_url": "database/postgres:url",
            "redis_url": "database/redis:url",
            "finnhub_key": "apis/finnhub:key"
        })
        print(f"All secrets: {secrets}")

    except Exception as e:
        print(f"Error: {e}")
