# Terraform configuration for PDE-Fluid-Φ infrastructure
# Supports AWS, GCP, and Azure deployments

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = "pde-fluid-phi-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}

# Local variables
locals {
  name_prefix = "pde-fluid-phi"
  environment = var.environment
  
  common_tags = {
    Project     = "PDE-Fluid-Phi"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "Terragon Labs"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${local.name_prefix}-cluster"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # CPU nodes for general workloads
    cpu_nodes = {
      name = "cpu-nodes"
      
      instance_types = ["m5.2xlarge"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 1
      max_size     = 10
      desired_size = 2
      
      disk_size = 100
      
      labels = {
        node-type = "cpu"
      }
      
      tags = merge(local.common_tags, {
        NodeType = "CPU"
      })
    }
    
    # GPU nodes for training
    gpu_nodes = {
      name = "gpu-nodes"
      
      instance_types = ["p3.2xlarge"]  # Tesla V100
      capacity_type  = "ON_DEMAND"
      
      min_size     = 0
      max_size     = 5
      desired_size = 1
      
      disk_size = 200
      
      labels = {
        node-type = "gpu"
        accelerator = "nvidia-tesla-v100"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = merge(local.common_tags, {
        NodeType = "GPU"
      })
    }
    
    # Spot instances for cost optimization
    spot_nodes = {
      name = "spot-nodes"
      
      instance_types = ["m5.large", "m5.xlarge", "m5.2xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 0
      max_size     = 20
      desired_size = 3
      
      disk_size = 50
      
      labels = {
        node-type = "spot"
      }
      
      taints = [
        {
          key    = "spot-instance"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = merge(local.common_tags, {
        NodeType = "Spot"
      })
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true
  aws_auth_users = var.aws_auth_users
  aws_auth_roles = var.aws_auth_roles

  tags = local.common_tags
}

# EKS Add-ons
resource "aws_eks_addon" "ebs_csi" {
  cluster_name = module.eks.cluster_name
  addon_name   = "aws-ebs-csi-driver"
  
  tags = local.common_tags
}

resource "aws_eks_addon" "vpc_cni" {
  cluster_name = module.eks.cluster_name
  addon_name   = "vpc-cni"
  
  tags = local.common_tags
}

resource "aws_eks_addon" "coredns" {
  cluster_name = module.eks.cluster_name
  addon_name   = "coredns"
  
  tags = local.common_tags
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = module.eks.cluster_name
  addon_name   = "kube-proxy"
  
  tags = local.common_tags
}

# S3 Bucket for data storage
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${local.name_prefix}-data-${random_id.bucket_suffix.hex}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_bucket_pab" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Random ID for unique bucket names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# RDS for metadata storage (optional)
resource "aws_db_subnet_group" "postgres" {
  count = var.enable_rds ? 1 : 0
  
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "aws_db_instance" "postgres" {
  count = var.enable_rds ? 1 : 0
  
  identifier = "${local.name_prefix}-postgres"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.rds_instance_class
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "pde_fluid_phi"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds[0].id]
  db_subnet_group_name   = aws_db_subnet_group.postgres[0].name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"
  
  tags = local.common_tags
}

# Security Group for RDS
resource "aws_security_group" "rds" {
  count = var.enable_rds ? 1 : 0
  
  name_prefix = "${local.name_prefix}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# ElastiCache for Redis (optional)
resource "aws_elasticache_subnet_group" "redis" {
  count = var.enable_redis ? 1 : 0
  
  name       = "${local.name_prefix}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "redis" {
  count = var.enable_redis ? 1 : 0
  
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for PDE-Fluid-Phi"
  
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.redis[0].name
  security_group_ids = [aws_security_group.redis[0].id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = local.common_tags
}

# Security Group for Redis
resource "aws_security_group" "redis" {
  count = var.enable_redis ? 1 : 0
  
  name_prefix = "${local.name_prefix}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# IAM Role for Kubernetes Service Account (IRSA)
module "irsa" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.name_prefix}-irsa-role"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["pde-fluid-phi:pde-fluid-phi"]
    }
  }

  role_policy_arns = {
    s3_access = aws_iam_policy.s3_access.arn
  }

  tags = local.common_tags
}

# IAM Policy for S3 access
resource "aws_iam_policy" "s3_access" {
  name_prefix = "${local.name_prefix}-s3-access-"
  description = "IAM policy for S3 access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_bucket.arn,
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      }
    ]
  })

  tags = local.common_tags
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${module.eks.cluster_name}/cluster"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# Application Load Balancer (optional)
resource "aws_lb" "main" {
  count = var.enable_alb ? 1 : 0
  
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[0].id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"

  tags = local.common_tags
}

# Security Group for ALB
resource "aws_security_group" "alb" {
  count = var.enable_alb ? 1 : 0
  
  name_prefix = "${local.name_prefix}-alb-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}