##
# HashiCorp Vault Terraform Module
# Deploys Vault for secrets management in production
##

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    vault = {
      source  = "hashicorp/vault"
      version = "~> 3.0"
    }
  }
}

# ============================================================================
# KMS Key for Auto-Unseal
# ============================================================================

resource "aws_kms_key" "vault_unseal" {
  description             = "Vault auto-unseal key for ${var.environment}"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = merge(
    var.tags,
    {
      Name = "vault-unseal-${var.environment}"
      Purpose = "vault-auto-unseal"
    }
  )
}

resource "aws_kms_alias" "vault_unseal" {
  name          = "alias/vault-unseal-${var.environment}"
  target_key_id = aws_kms_key.vault_unseal.key_id
}

# ============================================================================
# IAM Role for Vault Instances
# ============================================================================

resource "aws_iam_role" "vault" {
  name = "vault-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "vault_kms" {
  name = "vault-kms-unseal"
  role = aws_iam_role.vault.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.vault_unseal.arn
      }
    ]
  })
}

resource "aws_iam_instance_profile" "vault" {
  name = "vault-${var.environment}"
  role = aws_iam_role.vault.name
}

# ============================================================================
# Security Group
# ============================================================================

resource "aws_security_group" "vault" {
  name        = "vault-${var.environment}"
  description = "Security group for Vault cluster"
  vpc_id      = var.vpc_id

  # API port
  ingress {
    from_port   = 8200
    to_port     = 8200
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
    description = "Vault API"
  }

  # Cluster port
  ingress {
    from_port = 8201
    to_port   = 8201
    protocol  = "tcp"
    self      = true
    description = "Vault cluster communication"
  }

  # Outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = merge(
    var.tags,
    {
      Name = "vault-${var.environment}"
    }
  )
}

# ============================================================================
# S3 Bucket for Vault Storage (Raft backend alternative)
# ============================================================================

resource "aws_s3_bucket" "vault_storage" {
  count  = var.storage_type == "s3" ? 1 : 0
  bucket = "vault-storage-${var.environment}-${var.aws_account_id}"

  tags = merge(
    var.tags,
    {
      Name = "vault-storage-${var.environment}"
    }
  )
}

resource "aws_s3_bucket_versioning" "vault_storage" {
  count  = var.storage_type == "s3" ? 1 : 0
  bucket = aws_s3_bucket.vault_storage[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "vault_storage" {
  count  = var.storage_type == "s3" ? 1 : 0
  bucket = aws_s3_bucket.vault_storage[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ============================================================================
# EC2 Launch Template (for ASG)
# ============================================================================

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_launch_template" "vault" {
  name_prefix   = "vault-${var.environment}-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.vault.name
  }

  vpc_security_group_ids = [aws_security_group.vault.id]

  user_data = base64encode(templatefile("${path.module}/templates/user_data.sh", {
    vault_version    = var.vault_version
    kms_key_id       = aws_kms_key.vault_unseal.id
    region           = var.aws_region
    environment      = var.environment
    storage_type     = var.storage_type
    s3_bucket        = var.storage_type == "s3" ? aws_s3_bucket.vault_storage[0].id : ""
  }))

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = 50
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      var.tags,
      {
        Name = "vault-${var.environment}"
      }
    )
  }

  tags = var.tags
}

# ============================================================================
# Auto Scaling Group
# ============================================================================

resource "aws_autoscaling_group" "vault" {
  name                = "vault-${var.environment}"
  vpc_zone_identifier = var.private_subnet_ids
  min_size            = var.vault_cluster_size
  max_size            = var.vault_cluster_size
  desired_capacity    = var.vault_cluster_size

  launch_template {
    id      = aws_launch_template.vault.id
    version = "$Latest"
  }

  health_check_type         = "ELB"
  health_check_grace_period = 300

  target_group_arns = [aws_lb_target_group.vault.arn]

  tag {
    key                 = "Name"
    value               = "vault-${var.environment}"
    propagate_at_launch = true
  }

  dynamic "tag" {
    for_each = var.tags
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# ============================================================================
# Load Balancer
# ============================================================================

resource "aws_lb" "vault" {
  name               = "vault-${var.environment}"
  internal           = var.internal_lb
  load_balancer_type = "application"
  security_groups    = [aws_security_group.vault.id]
  subnets            = var.private_subnet_ids

  enable_deletion_protection = var.enable_deletion_protection

  tags = merge(
    var.tags,
    {
      Name = "vault-${var.environment}"
    }
  )
}

resource "aws_lb_target_group" "vault" {
  name     = "vault-${var.environment}"
  port     = 8200
  protocol = "HTTPS"
  vpc_id   = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/v1/sys/health"
    port                = "8200"
    protocol            = "HTTPS"
    matcher             = "200,429,473"  # 429 = sealed, 473 = standby
  }

  tags = var.tags
}

resource "aws_lb_listener" "vault" {
  load_balancer_arn = aws_lb.vault.arn
  port              = "8200"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.vault.arn
  }
}

# ============================================================================
# Route53 DNS Record
# ============================================================================

resource "aws_route53_record" "vault" {
  count   = var.create_dns_record ? 1 : 0
  zone_id = var.route53_zone_id
  name    = "vault.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_lb.vault.dns_name
    zone_id                = aws_lb.vault.zone_id
    evaluate_target_health = true
  }
}

# ============================================================================
# Outputs
# ============================================================================

output "vault_url" {
  description = "Vault URL"
  value       = var.create_dns_record ? "https://vault.${var.domain_name}:8200" : "https://${aws_lb.vault.dns_name}:8200"
}

output "vault_lb_dns" {
  description = "Vault load balancer DNS name"
  value       = aws_lb.vault.dns_name
}

output "kms_key_id" {
  description = "KMS key ID for Vault auto-unseal"
  value       = aws_kms_key.vault_unseal.id
}

output "vault_security_group_id" {
  description = "Vault security group ID"
  value       = aws_security_group.vault.id
}
