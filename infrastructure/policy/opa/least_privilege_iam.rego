# OPA Policy: Least Privilege IAM
# Enforces least-privilege IAM policies

package terraform.iam_least_privilege

import future.keywords.contains
import future.keywords.if

# Dangerous IAM actions that should be restricted
dangerous_actions := {
    "*",
    "iam:*",
    "s3:*",
    "ec2:*",
    "rds:DeleteDBInstance",
    "dynamodb:DeleteTable",
}

# Check if policy has wildcard permissions
has_wildcard_permission(policy) if {
    some statement in policy.Statement
    some action in statement.Action
    action == "*"
}

# Check if policy has dangerous actions
has_dangerous_action(policy) if {
    some statement in policy.Statement
    some action in statement.Action
    action in dangerous_actions
}

# Deny IAM policies with wildcard (*) permissions
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_iam_policy"

    policy_doc := json.unmarshal(resource.change.after.policy)
    has_wildcard_permission(policy_doc)

    msg := sprintf(
        "IAM policy %s uses wildcard (*) permissions - violates least privilege",
        [resource.address]
    )
}

# Deny IAM roles with overly permissive assume role policies
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_iam_role"

    assume_role_policy := json.unmarshal(resource.change.after.assume_role_policy)
    some statement in assume_role_policy.Statement

    # Check for overly broad principals
    principal := statement.Principal
    principal.AWS == "*"

    msg := sprintf(
        "IAM role %s allows any AWS principal to assume - too permissive",
        [resource.address]
    )
}

# Warn about policies with broad resource access
warn contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_iam_policy"

    policy_doc := json.unmarshal(resource.change.after.policy)
    some statement in policy_doc.Statement

    statement.Resource == "*"
    statement.Effect == "Allow"

    msg := sprintf(
        "IAM policy %s grants access to all resources (*) - consider restricting",
        [resource.address]
    )
}

# Deny policies without MFA for sensitive actions
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_iam_policy"

    policy_doc := json.unmarshal(resource.change.after.policy)
    some statement in policy_doc.Statement

    # Check for sensitive actions
    some action in statement.Action
    startswith(action, "iam:")
    statement.Effect == "Allow"

    # No MFA condition
    not statement.Condition.Bool["aws:MultiFactorAuthPresent"]

    msg := sprintf(
        "IAM policy %s allows sensitive actions without MFA requirement",
        [resource.address]
    )
}

# Deny inline policies (policies should be managed separately)
deny contains msg if {
    some resource in input.resource_changes
    resource.type in {"aws_iam_role_policy", "aws_iam_user_policy", "aws_iam_group_policy"}

    msg := sprintf(
        "Inline IAM policy %s detected - use managed policies instead",
        [resource.address]
    )
}

# Require policy name to indicate purpose
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_iam_policy"

    name := resource.change.after.name
    not contains(name, "trading")
    not contains(name, "readonly")
    not contains(name, "admin")

    msg := sprintf(
        "IAM policy %s name should clearly indicate purpose",
        [resource.address]
    )
}
