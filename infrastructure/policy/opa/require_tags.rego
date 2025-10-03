# OPA Policy: Require Tags
# Ensures all resources have required tags

package terraform.required_tags

import future.keywords.contains
import future.keywords.if

# Required tags for all resources
required_tags := ["environment", "owner", "cost_center"]

# Resource types that must be tagged
taggable_resources := {
    "aws_instance",
    "aws_db_instance",
    "aws_elasticache_cluster",
    "aws_s3_bucket",
    "aws_ecs_service",
    "aws_eks_cluster",
    "aws_rds_cluster",
    "aws_lambda_function",
}

# Check if resource is taggable
is_taggable(resource) if {
    resource.type in taggable_resources
}

# Deny if required tags are missing
deny contains msg if {
    some resource in input.resource_changes
    is_taggable(resource)

    tags := object.get(resource.change.after, "tags", {})
    some required_tag in required_tags
    not tags[required_tag]

    msg := sprintf(
        "Resource %s (type: %s) is missing required tag: %s",
        [resource.address, resource.type, required_tag]
    )
}

# Warn if tag values are empty
warn contains msg if {
    some resource in input.resource_changes
    is_taggable(resource)

    tags := object.get(resource.change.after, "tags", {})
    some tag_name, tag_value in tags
    tag_value == ""

    msg := sprintf(
        "Resource %s has empty value for tag: %s",
        [resource.address, tag_name]
    )
}

# Valid environment values
valid_environments := {"dev", "staging", "prod"}

# Deny if environment tag has invalid value
deny contains msg if {
    some resource in input.resource_changes
    is_taggable(resource)

    tags := object.get(resource.change.after, "tags", {})
    env := tags.environment
    not env in valid_environments

    msg := sprintf(
        "Resource %s has invalid environment tag: %s (must be one of: %v)",
        [resource.address, env, valid_environments]
    )
}
