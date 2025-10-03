# OPA Policy: Encryption Required
# Ensures all data stores use encryption

package terraform.encryption_required

import future.keywords.contains
import future.keywords.if

# Deny S3 buckets without encryption
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_s3_bucket"

    # Check for encryption configuration
    not has_s3_encryption(resource)

    msg := sprintf(
        "S3 bucket %s must have encryption enabled",
        [resource.address]
    )
}

has_s3_encryption(resource) if {
    encryption := resource.change.after.server_side_encryption_configuration
    count(encryption) > 0
}

# Deny RDS instances without encryption
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_db_instance"

    storage_encrypted := object.get(resource.change.after, "storage_encrypted", false)
    not storage_encrypted

    msg := sprintf(
        "RDS instance %s must have storage encryption enabled",
        [resource.address]
    )
}

# Deny EBS volumes without encryption
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_ebs_volume"

    encrypted := object.get(resource.change.after, "encrypted", false)
    not encrypted

    msg := sprintf(
        "EBS volume %s must be encrypted",
        [resource.address]
    )
}

# Deny ElastiCache clusters without encryption in transit
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_elasticache_replication_group"

    transit_encryption := object.get(resource.change.after, "transit_encryption_enabled", false)
    not transit_encryption

    msg := sprintf(
        "ElastiCache cluster %s must have transit encryption enabled",
        [resource.address]
    )
}

# Deny ElastiCache clusters without encryption at rest
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_elasticache_replication_group"

    at_rest_encryption := object.get(resource.change.after, "at_rest_encryption_enabled", false)
    not at_rest_encryption

    msg := sprintf(
        "ElastiCache cluster %s must have at-rest encryption enabled",
        [resource.address]
    )
}

# Warn about default KMS keys
warn contains msg if {
    some resource in input.resource_changes
    resource.type in {"aws_db_instance", "aws_s3_bucket", "aws_ebs_volume"}

    kms_key := object.get(resource.change.after, "kms_key_id", "")
    kms_key == ""

    msg := sprintf(
        "Resource %s using default KMS key - consider using customer-managed CMK",
        [resource.address]
    )
}

# Deny RDS instances publicly accessible
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_db_instance"

    publicly_accessible := object.get(resource.change.after, "publicly_accessible", false)
    publicly_accessible == true

    msg := sprintf(
        "RDS instance %s must not be publicly accessible",
        [resource.address]
    )
}
