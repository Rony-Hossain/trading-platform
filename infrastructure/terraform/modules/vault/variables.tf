##
# Vault Terraform Module Variables
##

variable "environment" {
  description = "Environment name (e.g., prod, staging, dev)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS account ID"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where Vault will be deployed"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for Vault instances"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access Vault"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "vault_version" {
  description = "Vault version to install"
  type        = string
  default     = "1.15.4"
}

variable "instance_type" {
  description = "EC2 instance type for Vault nodes"
  type        = string
  default     = "t3.medium"
}

variable "vault_cluster_size" {
  description = "Number of Vault nodes in cluster"
  type        = number
  default     = 3
  validation {
    condition     = var.vault_cluster_size >= 3 && var.vault_cluster_size % 2 == 1
    error_message = "Vault cluster size must be >= 3 and odd number for Raft quorum"
  }
}

variable "storage_type" {
  description = "Vault storage backend type (raft or s3)"
  type        = string
  default     = "raft"
  validation {
    condition     = contains(["raft", "s3"], var.storage_type)
    error_message = "Storage type must be 'raft' or 's3'"
  }
}

variable "internal_lb" {
  description = "Whether load balancer should be internal"
  type        = bool
  default     = true
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection on load balancer"
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
}

variable "create_dns_record" {
  description = "Whether to create Route53 DNS record"
  type        = bool
  default     = true
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for Vault"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
