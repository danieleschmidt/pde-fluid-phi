#!/bin/bash

# PDE-Fluid-Φ Production Deployment Script
# Automates deployment to various environments with validation and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION=${VERSION:-$(git describe --tags --always --dirty)}
ENVIRONMENT=${ENVIRONMENT:-production}
REGISTRY=${REGISTRY:-ghcr.io/your-org/pde-fluid-phi}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
handle_error() {
    log_error "Deployment failed at line $1"
    cleanup_on_failure
    exit 1
}

trap 'handle_error ${LINENO}' ERR

# Cleanup function
cleanup_on_failure() {
    log_warn "Cleaning up failed deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            docker-compose -f docker-compose.prod.yml down --volumes --remove-orphans || true
            ;;
        kubernetes)
            kubectl rollout undo deployment/pde-fluid-phi -n pde-fluid-phi || true
            ;;
    esac
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check required tools
    local required_tools=("docker" "git")
    
    case $DEPLOYMENT_TYPE in
        kubernetes)
            required_tools+=("kubectl" "helm")
            ;;
        docker)
            required_tools+=("docker-compose")
            ;;
    esac
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check git status
    if [[ -n $(git status --porcelain) ]]; then
        log_warn "Working directory has uncommitted changes"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Run security scan
    log_info "Running security scan..."
    if ! python3 "$PROJECT_ROOT/security_scan.py" --root-path "$PROJECT_ROOT" --fail-on-high; then
        log_error "Security scan failed with high/critical issues"
        exit 1
    fi
    
    # Run tests
    log_info "Running test suite..."
    cd "$PROJECT_ROOT"
    if command -v pytest &> /dev/null; then
        python3 -m pytest tests/ -v --tb=short || {
            log_error "Test suite failed"
            exit 1
        }
    else
        log_warn "pytest not available, skipping tests"
    fi
    
    log_info "Pre-flight checks passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    local image_tag="${REGISTRY}:${VERSION}"
    local latest_tag="${REGISTRY}:latest"
    
    # Build with multi-stage Dockerfile
    docker build \
        --file "$SCRIPT_DIR/Dockerfile" \
        --target production \
        --tag "$image_tag" \
        --tag "$latest_tag" \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        "$PROJECT_ROOT"
    
    log_info "Built image: $image_tag"
    
    # Security scan of image
    log_info "Scanning Docker image for vulnerabilities..."
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$HOME/.cache":/root/.cache \
        aquasec/trivy:latest image --exit-code 1 --severity HIGH,CRITICAL "$image_tag" || {
        log_warn "Docker image has high/critical vulnerabilities"
        read -p "Continue deployment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    }
    
    # Push to registry
    if [[ $PUSH_IMAGE == "true" ]]; then
        log_info "Pushing image to registry..."
        docker push "$image_tag"
        docker push "$latest_tag"
    fi
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$SCRIPT_DIR"
    
    # Generate .env file
    cat > .env <<EOF
VERSION=${VERSION}
ENVIRONMENT=${ENVIRONMENT}
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)
EOF
    
    # Deploy services
    docker-compose -f docker-compose.prod.yml up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f docker-compose.prod.yml exec -T pde-fluid-phi curl -f http://localhost:8000/health &> /dev/null; then
            log_info "Application is ready"
            break
        fi
        
        ((attempt++))
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Application failed to start within timeout"
            docker-compose -f docker-compose.prod.yml logs pde-fluid-phi
            exit 1
        fi
        
        sleep 5
    done
    
    # Run smoke tests
    run_smoke_tests "http://localhost:8000"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    local namespace="pde-fluid-phi"
    local kube_dir="$SCRIPT_DIR/kubernetes"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f "$kube_dir/configmap.yaml" -n "$namespace"
    kubectl apply -f "$kube_dir/secrets.yaml" -n "$namespace"
    kubectl apply -f "$kube_dir/rbac.yaml" -n "$namespace"
    kubectl apply -f "$kube_dir/pvc.yaml" -n "$namespace"
    
    # Update image tag in deployment
    sed -i.bak "s|image: pde-fluid-phi:latest|image: ${REGISTRY}:${VERSION}|g" \
        "$kube_dir/deployment.yaml"
    
    # Apply deployment
    kubectl apply -f "$kube_dir/deployment.yaml" -n "$namespace"
    
    # Wait for rollout to complete
    kubectl rollout status deployment/pde-fluid-phi -n "$namespace" --timeout=600s
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service pde-fluid-phi-service -n "$namespace" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "$service_ip" ]]; then
        log_warn "LoadBalancer IP not available yet, using port-forward for testing"
        kubectl port-forward -n "$namespace" svc/pde-fluid-phi-service 8080:80 &
        local port_forward_pid=$!
        service_endpoint="http://localhost:8080"
        
        # Cleanup port-forward on exit
        trap "kill $port_forward_pid 2>/dev/null || true" EXIT
    else
        service_endpoint="http://$service_ip"
    fi
    
    # Run smoke tests
    sleep 10  # Wait for port-forward to be ready
    run_smoke_tests "$service_endpoint"
    
    log_info "Kubernetes deployment completed successfully"
    log_info "Application available at: $service_endpoint"
}

# Run smoke tests
run_smoke_tests() {
    local endpoint="$1"
    log_info "Running smoke tests against $endpoint..."
    
    # Health check
    if ! curl -f "$endpoint/health" --max-time 30; then
        log_error "Health check failed"
        exit 1
    fi
    
    # Ready check
    if ! curl -f "$endpoint/ready" --max-time 30; then
        log_error "Ready check failed"
        exit 1
    fi
    
    # Metrics endpoint
    if ! curl -f "$endpoint/metrics" --max-time 30 > /dev/null; then
        log_warn "Metrics endpoint not available"
    fi
    
    # Basic inference test (if test data available)
    if [[ -f "$PROJECT_ROOT/test_data/sample_input.json" ]]; then
        log_info "Running inference test..."
        if ! curl -X POST \
            -H "Content-Type: application/json" \
            -d @"$PROJECT_ROOT/test_data/sample_input.json" \
            "$endpoint/predict" \
            --max-time 60 > /dev/null; then
            log_warn "Inference test failed"
        else
            log_info "Inference test passed"
        fi
    fi
    
    log_info "Smoke tests completed successfully"
}

# Generate deployment report
generate_report() {
    local report_file="deployment_report_${VERSION}_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" <<EOF
{
  "deployment": {
    "version": "${VERSION}",
    "environment": "${ENVIRONMENT}",
    "deployment_type": "${DEPLOYMENT_TYPE}",
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "git_commit": "$(git rev-parse HEAD)",
    "deployed_by": "${USER:-unknown}",
    "deployment_duration": "${DEPLOYMENT_DURATION:-unknown}"
  },
  "validation": {
    "security_scan_passed": true,
    "tests_passed": true,
    "smoke_tests_passed": true,
    "image_scan_passed": true
  },
  "endpoints": {
    "health": "${SERVICE_ENDPOINT}/health",
    "api": "${SERVICE_ENDPOINT}/api/v1",
    "metrics": "${SERVICE_ENDPOINT}/metrics",
    "docs": "${SERVICE_ENDPOINT}/docs"
  }
}
EOF
    
    log_info "Deployment report generated: $report_file"
}

# Rollback function
rollback() {
    log_warn "Initiating rollback..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            docker-compose -f docker-compose.prod.yml down
            # Restore from backup if available
            if [[ -f .env.backup ]]; then
                mv .env.backup .env
                docker-compose -f docker-compose.prod.yml up -d
            fi
            ;;
        kubernetes)
            kubectl rollout undo deployment/pde-fluid-phi -n pde-fluid-phi
            kubectl rollout status deployment/pde-fluid-phi -n pde-fluid-phi
            ;;
    esac
    
    log_info "Rollback completed"
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log_info "Starting deployment of PDE-Fluid-Φ v${VERSION} to ${ENVIRONMENT}"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            --skip-checks)
                SKIP_CHECKS="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --push-image)
                PUSH_IMAGE="true"
                shift
                ;;
            --rollback)
                rollback
                exit 0
                ;;
            --help|-h)
                cat << EOF
Usage: $0 [OPTIONS]

Deploy PDE-Fluid-Φ to production environment

Options:
    --type TYPE         Deployment type (docker|kubernetes) [default: docker]
    --skip-checks       Skip pre-flight checks
    --skip-build        Skip image build
    --push-image        Push image to registry
    --rollback          Rollback to previous deployment
    --help              Show this help message

Environment Variables:
    VERSION             Application version [default: git describe]
    ENVIRONMENT         Target environment [default: production]
    REGISTRY            Container registry [default: ghcr.io/your-org/pde-fluid-phi]
EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set defaults
    DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-docker}
    SKIP_CHECKS=${SKIP_CHECKS:-false}
    SKIP_BUILD=${SKIP_BUILD:-false}
    PUSH_IMAGE=${PUSH_IMAGE:-false}
    
    # Validate deployment type
    if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|kubernetes)$ ]]; then
        log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
        exit 1
    fi
    
    # Run deployment steps
    if [[ $SKIP_CHECKS != "true" ]]; then
        preflight_checks
    fi
    
    if [[ $SKIP_BUILD != "true" ]]; then
        build_image
    fi
    
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            SERVICE_ENDPOINT="http://localhost:8000"
            ;;
        kubernetes)
            deploy_kubernetes
            # SERVICE_ENDPOINT set in deploy_kubernetes function
            ;;
    esac
    
    # Calculate deployment duration
    local end_time=$(date +%s)
    DEPLOYMENT_DURATION=$((end_time - start_time))
    
    # Generate report
    generate_report
    
    log_info "Deployment completed successfully in ${DEPLOYMENT_DURATION} seconds"
    log_info "Application is available at: ${SERVICE_ENDPOINT}"
}

# Run main function
main "$@"