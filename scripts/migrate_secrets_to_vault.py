#!/usr/bin/env python3
"""
Migrate Secrets from .env to HashiCorp Vault

This script:
1. Reads secrets from .env file
2. Connects to Vault
3. Writes secrets to appropriate paths
4. Verifies all secrets migrated
5. Creates new .env with Vault configuration only
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)


class SecretMigrator:
    """Migrates secrets from .env to Vault"""

    # Define secret categorization rules
    SECRET_CATEGORIES = {
        'database': [
            'DATABASE_URL', 'POSTGRES_URL', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
            'POSTGRES_DB', 'POSTGRES_HOST', 'POSTGRES_PORT',
            'REDIS_URL', 'REDIS_HOST', 'REDIS_PORT',
            'TIMESCALEDB_URL'
        ],
        'apis': [
            'FINNHUB_API_KEY', 'FINNHUB_TOKEN',
            'TWITTER_BEARER_TOKEN', 'TWITTER_API_KEY', 'TWITTER_API_SECRET',
            'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_SECRET',
            'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET',
            'REDDIT_USER_AGENT', 'REDDIT_USERNAME', 'REDDIT_PASSWORD',
            'FRED_API_KEY'
        ],
        'services': [
            'JWT_SECRET', 'JWT_ALGORITHM', 'JWT_EXPIRATION',
            'ENCRYPTION_KEY', 'SECRET_KEY'
        ]
    }

    def __init__(
        self,
        env_file: Path,
        vault_client: VaultClient,
        namespace: str = "trading-platform",
        dry_run: bool = False
    ):
        self.env_file = env_file
        self.vault = vault_client
        self.namespace = namespace
        self.dry_run = dry_run

        self.secrets: Dict[str, str] = {}
        self.categorized_secrets: Dict[str, Dict[str, str]] = {}
        self.migration_errors: List[Tuple[str, str]] = []

    def parse_env_file(self) -> Dict[str, str]:
        """
        Parse .env file and extract secrets

        Returns:
            Dictionary of environment variables
        """
        logger.info(f"Parsing {self.env_file}")

        secrets = {}

        if not self.env_file.exists():
            raise FileNotFoundError(f".env file not found: {self.env_file}")

        with open(self.env_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE
                match = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)$', line)
                if match:
                    key, value = match.groups()

                    # Remove quotes if present
                    value = value.strip('"').strip("'")

                    secrets[key] = value
                else:
                    logger.warning(f"Line {line_num}: Could not parse: {line}")

        logger.info(f"Found {len(secrets)} secrets in .env file")
        self.secrets = secrets

        return secrets

    def categorize_secrets(self):
        """Categorize secrets into database, apis, services, etc."""
        categorized = {
            'database': {},
            'apis': {},
            'services': {},
            'uncategorized': {}
        }

        for key, value in self.secrets.items():
            assigned = False

            for category, patterns in self.SECRET_CATEGORIES.items():
                if key in patterns:
                    categorized[category][key] = value
                    assigned = True
                    break

            if not assigned:
                categorized['uncategorized'][key] = value

        self.categorized_secrets = categorized

        logger.info("Secret categorization:")
        for category, secrets in categorized.items():
            if secrets:
                logger.info(f"  {category}: {len(secrets)} secrets")

    def migrate_category(
        self,
        category: str,
        secrets: Dict[str, str]
    ):
        """
        Migrate secrets for a category to Vault

        Args:
            category: Category name (database, apis, services)
            secrets: Dictionary of secrets for this category
        """
        if not secrets:
            logger.info(f"No secrets to migrate for category: {category}")
            return

        path = f"{category}"

        logger.info(f"Migrating {len(secrets)} secrets to {path}")

        if self.dry_run:
            logger.info(f"DRY RUN: Would write to {path}:")
            for key, value in secrets.items():
                masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '***'
                logger.info(f"  {key} = {masked_value}")
            return

        try:
            # Convert env var names to lowercase keys
            vault_secrets = {
                key.lower(): value
                for key, value in secrets.items()
            }

            self.vault.set_secret(path, vault_secrets)
            logger.info(f"✅ Successfully migrated {len(secrets)} secrets to {path}")

        except Exception as e:
            logger.error(f"❌ Error migrating {category}: {e}")
            self.migration_errors.append((category, str(e)))

    def migrate_all(self):
        """Migrate all categorized secrets to Vault"""
        logger.info("Starting secret migration to Vault")

        # Migrate each category
        for category, secrets in self.categorized_secrets.items():
            if category != 'uncategorized' and secrets:
                self.migrate_category(category, secrets)

        # Handle uncategorized secrets
        if self.categorized_secrets['uncategorized']:
            logger.warning(
                f"Found {len(self.categorized_secrets['uncategorized'])} uncategorized secrets:"
            )
            for key in self.categorized_secrets['uncategorized'].keys():
                logger.warning(f"  - {key}")

            # Optionally migrate to misc category
            self.migrate_category("misc", self.categorized_secrets['uncategorized'])

    def verify_migration(self) -> bool:
        """
        Verify all secrets were migrated successfully

        Returns:
            True if all secrets verified, False otherwise
        """
        if self.dry_run:
            logger.info("DRY RUN: Skipping verification")
            return True

        logger.info("Verifying migrated secrets...")

        all_verified = True

        for category, secrets in self.categorized_secrets.items():
            if category == 'uncategorized' or not secrets:
                continue

            path = f"{category}"

            try:
                # Read back from Vault
                vault_secrets = self.vault.get_secret(path)

                # Check each secret
                for key, original_value in secrets.items():
                    vault_key = key.lower()

                    if vault_key not in vault_secrets:
                        logger.error(f"❌ Missing in Vault: {category}/{vault_key}")
                        all_verified = False
                    elif vault_secrets[vault_key] != original_value:
                        logger.error(f"❌ Value mismatch for {category}/{vault_key}")
                        all_verified = False
                    else:
                        logger.debug(f"✅ Verified: {category}/{vault_key}")

            except Exception as e:
                logger.error(f"❌ Error verifying {category}: {e}")
                all_verified = False

        if all_verified:
            logger.info("✅ All secrets verified successfully")
        else:
            logger.error("❌ Some secrets failed verification")

        return all_verified

    def create_new_env_file(self, output_file: Path):
        """
        Create new .env file with only Vault configuration

        Args:
            output_file: Path to new .env file
        """
        if self.dry_run:
            logger.info(f"DRY RUN: Would create {output_file}")
            return

        logger.info(f"Creating new .env file: {output_file}")

        content = """# Trading Platform Environment Configuration
# Secrets are now stored in HashiCorp Vault

# Vault Configuration
VAULT_ADDR=http://localhost:8200
VAULT_ROLE_ID=<your-role-id>
VAULT_SECRET_ID=<your-secret-id>

# Optional: Use token authentication instead
# VAULT_TOKEN=<your-vault-token>

# Service URLs (non-sensitive)
MARKET_DATA_SERVICE_URL=http://localhost:8001
FUNDAMENTALS_SERVICE_URL=http://localhost:8002
SENTIMENT_SERVICE_URL=http://localhost:8003
ANALYSIS_SERVICE_URL=http://localhost:8004
STRATEGY_SERVICE_URL=http://localhost:8005
SIGNAL_SERVICE_URL=http://localhost:8000
EVENT_SERVICE_URL=http://localhost:8010
TRADE_JOURNAL_URL=http://localhost:8008

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
"""

        with open(output_file, 'w') as f:
            f.write(content)

        logger.info(f"✅ Created {output_file}")
        logger.info("⚠️  Remember to update VAULT_ROLE_ID and VAULT_SECRET_ID")


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate secrets from .env to HashiCorp Vault"
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )
    parser.add_argument(
        '--vault-addr',
        default=os.getenv('VAULT_ADDR', 'http://localhost:8200'),
        help='Vault server address'
    )
    parser.add_argument(
        '--vault-token',
        default=os.getenv('VAULT_TOKEN'),
        help='Vault root token'
    )
    parser.add_argument(
        '--namespace',
        default='trading-platform',
        help='Vault namespace for secrets'
    )
    parser.add_argument(
        '--output-env',
        default='.env.new',
        help='Output file for new .env (default: .env.new)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without writing to Vault'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing secrets in Vault'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if not args.vault_token:
        logger.error("VAULT_TOKEN not provided. Use --vault-token or set VAULT_TOKEN env var")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Secret Migration Tool")
    logger.info("=" * 80)
    logger.info(f"Vault Address: {args.vault_addr}")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Env File: {args.env_file}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 80)

    try:
        # Initialize Vault client
        vault = VaultClient(
            vault_addr=args.vault_addr,
            token=args.vault_token,
            namespace=args.namespace
        )

        if not vault.is_authenticated():
            logger.error("Failed to authenticate with Vault")
            sys.exit(1)

        logger.info("✅ Connected to Vault successfully")

        # Initialize migrator
        migrator = SecretMigrator(
            env_file=Path(args.env_file),
            vault_client=vault,
            namespace=args.namespace,
            dry_run=args.dry_run
        )

        # Parse .env file
        migrator.parse_env_file()

        # Categorize secrets
        migrator.categorize_secrets()

        if args.verify_only:
            # Only verify
            success = migrator.verify_migration()
            sys.exit(0 if success else 1)

        # Migrate secrets
        migrator.migrate_all()

        # Verify migration
        if not args.dry_run:
            success = migrator.verify_migration()

            if not success:
                logger.error("Migration verification failed. Check logs above.")
                sys.exit(1)

            # Create new .env file
            migrator.create_new_env_file(Path(args.output_env))

            logger.info("\n" + "=" * 80)
            logger.info("✅ Migration completed successfully!")
            logger.info("=" * 80)
            logger.info("Next steps:")
            logger.info("1. Review the new .env file and update Vault credentials")
            logger.info("2. Test application with Vault integration")
            logger.info("3. Backup and remove old .env file")
            logger.info("=" * 80)

        else:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN completed. No changes made to Vault.")
            logger.info("Remove --dry-run to perform actual migration.")
            logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
