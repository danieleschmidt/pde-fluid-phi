#!/bin/bash
# Comprehensive deployment script for PDE-Fluid-Φ
# Supports multiple environments and deployment strategies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default values
ENVIRONMENT=${ENVIRONMENT:-"development"}
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-"kubernetes"}
NAMESPACE=${NAMESPACE:-"pde-fluid-phi"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"terragonlabs"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
DRY_RUN=${DRY_RUN:-"false"}
SKIP_BUILD=${SKIP_BUILD:-"false"}
SKIP_TESTS=${SKIP_TESTS:-"false"}

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

usage() {
    cat << EOF
PDE-Fluid-Φ Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build Docker images
    test            Run tests
    deploy          Deploy to Kubernetes
    terraform       Deploy infrastructure with Terraform
    helm            Deploy using Helm charts
    cleanup         Clean up deployments
    status          Check deployment status
    logs            View logs
    help            Show this help message

Options:
    -e, --environment    Environment (development|staging|production) [default: development]
    -t, --type          Deployment type (kubernetes|docker-compose|local) [default: kubernetes]
    -n, --namespace     Kubernetes namespace [default: pde-fluid-phi]
    -r, --registry      Docker registry [default: terragonlabs]
    -i, --image-tag     Image tag [default: latest]
    --dry-run          Perform a dry run without making changes
    --skip-build       Skip Docker build step
    --skip-tests       Skip test execution
    --help             Show this help message

Examples:
    $0 build                                    # Build Docker images
    $0 deploy -e production -t kubernetes      # Deploy to production
    $0 terraform -e staging                    # Deploy infrastructure
    $0 helm -n my-namespace                    # Deploy with Helm
    $0 cleanup -e development                  # Clean up development environment

Environment Variables:
    KUBECONFIG         Path to kubeconfig file
    AWS_PROFILE        AWS profile for Terraform
    DOCKER_REGISTRY    Docker registry URL
    SKIP_BUILD         Skip building images (true/false)
    SKIP_TESTS         Skip running tests (true/false)
    DRY_RUN           Perform dry run (true/false)

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check required tools
    for cmd in docker kubectl helm terraform; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install the missing tools and retry."
        exit 1
    fi
    
    # Check Kubernetes connection
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster. Check your KUBECONFIG."
            exit 1
        fi
        log_success "Kubernetes cluster connection verified"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running or accessible"
        exit 1
    fi
    
    log_success "All dependencies verified"
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build multi-stage images
    local image_base="${DOCKER_REGISTRY}/pde-fluid-phi"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would build image ${image_base}:${IMAGE_TAG}"
        return 0
    fi
    
    # Build CPU image
    log_info "Building CPU image..."
    docker build \
        --target cpu \
        --tag "${image_base}:${IMAGE_TAG}-cpu" \
        --tag "${image_base}:cpu" \
        .
    
    # Build GPU image if CUDA is available
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        log_info "Building GPU image..."
        docker build \
            --target gpu \
            --tag "${image_base}:${IMAGE_TAG}-gpu" \
            --tag "${image_base}:gpu" \
            --tag "${image_base}:${IMAGE_TAG}" \
            --tag "${image_base}:latest" \
            .
    else
        log_warning "NVIDIA Docker not available, using CPU image as default"
        docker tag "${image_base}:${IMAGE_TAG}-cpu" "${image_base}:${IMAGE_TAG}"
        docker tag "${image_base}:${IMAGE_TAG}-cpu" "${image_base}:latest"
    fi
    
    # Build development image
    if [[ "$ENVIRONMENT" == "development" ]]; then
        log_info "Building development image..."
        docker build \
            --target development \
            --tag "${image_base}:${IMAGE_TAG}-dev" \
            --tag "${image_base}:dev" \
            .
    fi
    
    log_success "Docker images built successfully"
}

run_tests() {
    log_info "Running tests..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would run tests"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Run quality checks
    if [[ -f "scripts/run_quality_checks.sh" ]]; then
        log_info "Running quality checks..."
        bash scripts/run_quality_checks.sh
    fi
    
    # Run containerized tests
    log_info "Running containerized tests..."
    docker run --rm \
        -v "$PROJECT_ROOT:/app" \
        -w /app \
        "${DOCKER_REGISTRY}/pde-fluid-phi:${IMAGE_TAG}" \
        pytest tests/ -v --tb=short
    
    log_success "All tests passed"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    
    local manifests_dir="$DEPLOYMENT_DIR/kubernetes"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f "$manifests_dir/" -n "$NAMESPACE"
        return 0
    fi
    
    # Apply in order
    kubectl apply -f "$manifests_dir/namespace.yaml"
    kubectl apply -f "$manifests_dir/storage.yaml" -n "$NAMESPACE"
    
    # Wait for PVCs to be bound
    log_info "Waiting for persistent volumes to be ready..."
    kubectl wait --for=condition=Bound pvc --all -n "$NAMESPACE" --timeout=300s
    
    # Apply remaining manifests
    kubectl apply -f "$manifests_dir/" -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
}

deploy_helm() {
    log_info "Deploying with Helm..."
    
    local chart_dir="$DEPLOYMENT_DIR/helm/pde-fluid-phi"
    local values_file="$chart_dir/values-${ENVIRONMENT}.yaml"
    
    # Use default values if environment-specific file doesn't exist
    if [[ ! -f "$values_file" ]]; then
        values_file="$chart_dir/values.yaml"
    fi
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Install or upgrade Helm release
    local helm_args=(
        upgrade --install
        pde-fluid-phi
        "$chart_dir"
        --namespace "$NAMESPACE"
        --values "$values_file"
        --set "image.tag=$IMAGE_TAG"
        --set "global.environment=$ENVIRONMENT"
        --timeout 600s
        --wait
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_args+=(--dry-run)
    fi
    
    helm "${helm_args[@]}"
    
    log_success "Helm deployment completed"
}

deploy_terraform() {
    log_info "Deploying infrastructure with Terraform..."
    
    local terraform_dir="$DEPLOYMENT_DIR/terraform"
    cd "$terraform_dir"
    
    # Initialize Terraform
    terraform init
    
    # Select or create workspace
    terraform workspace select "$ENVIRONMENT" || terraform workspace new "$ENVIRONMENT"
    
    # Plan
    local tf_var_file="environments/${ENVIRONMENT}.tfvars"
    if [[ ! -f "$tf_var_file" ]]; then
        log_warning "Terraform variables file not found: $tf_var_file"
        tf_var_file=""
    fi
    
    local terraform_args=(plan)
    if [[ -n "$tf_var_file" ]]; then
        terraform_args+=(-var-file="$tf_var_file")
    fi
    
    terraform "${terraform_args[@]}"
    
    # Apply
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Applying Terraform configuration..."
        terraform_args=(apply -auto-approve)
        if [[ -n "$tf_var_file" ]]; then
            terraform_args+=(-var-file="$tf_var_file")
        fi
        
        terraform "${terraform_args[@]}"
        log_success "Terraform deployment completed"
    else
        log_info "DRY RUN: Terraform plan completed"
    fi
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    local compose_file="docker-compose.yml"
    local override_file="docker-compose.${ENVIRONMENT}.yml"
    
    local compose_args=(-f "$compose_file")
    if [[ -f "$override_file" ]]; then
        compose_args+=(-f "$override_file")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        docker-compose "${compose_args[@]}" config
        return 0
    fi
    
    # Deploy
    docker-compose "${compose_args[@]}" up -d
    
    log_success "Docker Compose deployment completed"
}

cleanup_deployment() {
    log_info "Cleaning up deployment..."
    
    case "$DEPLOYMENT_TYPE" in
        kubernetes)
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "DRY RUN: Would delete namespace $NAMESPACE"
                return 0
            fi
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
        helm)
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "DRY RUN: Would uninstall Helm release"
                return 0
            fi
            helm uninstall pde-fluid-phi -n "$NAMESPACE" || true
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
        docker-compose)
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "DRY RUN: Would stop Docker Compose services"
                return 0
            fi
            cd "$PROJECT_ROOT"
            docker-compose down --remove-orphans --volumes
            ;;
        terraform)
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "DRY RUN: Would destroy Terraform resources"
                return 0
            fi
            cd "$DEPLOYMENT_DIR/terraform"
            terraform workspace select "$ENVIRONMENT"
            terraform destroy -auto-approve
            ;;
    esac
    
    log_success "Cleanup completed"
}

check_status() {
    log_info "Checking deployment status..."
    
    case "$DEPLOYMENT_TYPE" in
        kubernetes|helm)
            kubectl get all -n "$NAMESPACE"
            kubectl get pvc -n "$NAMESPACE"
            ;;
        docker-compose)
            cd "$PROJECT_ROOT"
            docker-compose ps
            ;;
        terraform)
            cd "$DEPLOYMENT_DIR/terraform"
            terraform show
            ;;
    esac
}

view_logs() {
    log_info "Viewing logs..."
    
    case "$DEPLOYMENT_TYPE" in
        kubernetes|helm)
            kubectl logs -l app=pde-fluid-phi -n "$NAMESPACE" --tail=100 -f
            ;;
        docker-compose)
            cd "$PROJECT_ROOT"
            docker-compose logs -f --tail=100
            ;;
    esac
}

push_images() {
    log_info "Pushing images to registry..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would push images to registry"
        return 0
    fi
    
    local image_base="${DOCKER_REGISTRY}/pde-fluid-phi"
    
    # Push all built images
    docker push "${image_base}:${IMAGE_TAG}"
    docker push "${image_base}:latest"
    
    if docker images "${image_base}:${IMAGE_TAG}-cpu" --format "table" | grep -q cpu; then
        docker push "${image_base}:${IMAGE_TAG}-cpu"
        docker push "${image_base}:cpu"
    fi
    
    if docker images "${image_base}:${IMAGE_TAG}-gpu" --format "table" | grep -q gpu; then
        docker push "${image_base}:${IMAGE_TAG}-gpu"
        docker push "${image_base}:gpu"
    fi
    
    log_success "Images pushed to registry"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Main execution
case "${COMMAND:-deploy}" in
    build)
        check_dependencies
        build_images
        ;;
    test)
        check_dependencies
        if [[ "$SKIP_BUILD" != "true" ]]; then
            build_images
        fi
        run_tests
        ;;
    deploy)
        check_dependencies
        if [[ "$SKIP_BUILD" != "true" ]]; then
            build_images
        fi
        if [[ "$SKIP_TESTS" != "true" ]]; then
            run_tests
        fi
        
        case "$DEPLOYMENT_TYPE" in
            kubernetes) deploy_kubernetes ;;
            helm) deploy_helm ;;
            docker-compose) deploy_docker_compose ;;
            terraform) deploy_terraform ;;
            *) log_error "Unknown deployment type: $DEPLOYMENT_TYPE"; exit 1 ;;
        esac
        ;;
    terraform)
        check_dependencies
        deploy_terraform
        ;;
    helm)
        check_dependencies
        if [[ "$SKIP_BUILD" != "true" ]]; then
            build_images
        fi
        deploy_helm
        ;;
    push)
        check_dependencies
        push_images
        ;;
    cleanup)
        check_dependencies
        cleanup_deployment
        ;;
    status)
        check_dependencies
        check_status
        ;;
    logs)
        check_dependencies
        view_logs
        ;;
    help)
        usage
        ;;
    *)
        log_error "Unknown command: ${COMMAND:-deploy}"
        usage
        exit 1
        ;;
esac

log_success "Deployment script completed successfully"