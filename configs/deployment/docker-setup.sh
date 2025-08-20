#!/bin/bash

# HFT Trading System Docker Setup Script
# Automated setup for development and production environments

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Docker is installed and running
check_docker() {
    print_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        print_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    print_info "Checking Docker Compose..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker Compose is available: $COMPOSE_CMD"
}

# Function to create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    directories=("logs" "reports" "data" "outputs" "configs/database" "configs/monitoring" "configs/monitoring/grafana/dashboards" "configs/monitoring/grafana/datasources")
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    done
}

# Function to create basic configuration files
create_config_files() {
    print_info "Creating basic configuration files..."
    
    # Create basic nginx config if it doesn't exist
    if [ ! -f "configs/deployment/nginx.conf" ]; then
        cat > configs/deployment/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream hft_app {
        server hft-trading:8080;
    }
    
    server {
        listen 80;
        location / {
            proxy_pass http://hft_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
        print_success "Created nginx.conf"
    fi
    
    # Create basic database init script
    if [ ! -f "configs/database/init.sql" ]; then
        cat > configs/database/init.sql << 'EOF'
-- HFT Trading System Database Initialization

-- Create tables for backtesting and analysis
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    volume INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp ON trading_signals(symbol, timestamp);
EOF
        print_success "Created init.sql"
    fi
    
    # Create basic Prometheus config
    if [ ! -f "configs/monitoring/prometheus.yml" ]; then
        cat > configs/monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hft-trading'
    static_configs:
      - targets: ['hft-trading:8080']
EOF
        print_success "Created prometheus.yml"
    fi
}

# Function to build Docker images
build_images() {
    print_info "Building Docker images..."
    
    $COMPOSE_CMD build --no-cache
    print_success "Docker images built successfully"
}

# Function to start services based on profile
start_services() {
    local profile=$1
    print_info "Starting services with profile: $profile"
    
    case $profile in
        "development")
            $COMPOSE_CMD --profile development up -d
            ;;
        "production")
            $COMPOSE_CMD --profile production up -d
            ;;
        "monitoring")
            $COMPOSE_CMD --profile development --profile monitoring up -d
            ;;
        "basic")
            $COMPOSE_CMD up -d hft-trading redis
            ;;
        *)
            print_error "Unknown profile: $profile"
            print_info "Available profiles: development, production, monitoring, basic"
            exit 1
            ;;
    esac
    
    print_success "Services started with profile: $profile"
}

# Function to show service status
show_status() {
    print_info "Service status:"
    $COMPOSE_CMD ps
    
    print_info "Service logs (last 20 lines):"
    $COMPOSE_CMD logs --tail=20 hft-trading
}

# Function to run health checks
health_check() {
    print_info "Running health checks..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check if main service is healthy
    if docker ps --filter "name=hft-trading" --filter "health=healthy" | grep -q hft-trading; then
        print_success "HFT Trading service is healthy"
    else
        print_warning "HFT Trading service health check failed"
    fi
    
    # Check Redis
    if docker ps --filter "name=hft-redis" --filter "health=healthy" | grep -q hft-redis; then
        print_success "Redis service is healthy"
    else
        print_warning "Redis service health check failed"
    fi
}

# Function to clean up Docker resources
cleanup() {
    print_info "Cleaning up Docker resources..."
    
    $COMPOSE_CMD down --volumes --remove-orphans
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Function to show usage
show_help() {
    cat << 'EOF'
HFT Trading System Docker Setup Script

Usage: ./docker-setup.sh [COMMAND] [OPTIONS]

Commands:
    setup [PROFILE]     Setup and start the system with specified profile
                       Profiles: development, production, monitoring, basic
    build              Build Docker images
    start [PROFILE]    Start services with specified profile
    stop               Stop all services
    status             Show service status
    logs [SERVICE]     Show logs for a service
    cleanup            Remove all containers, volumes, and unused images
    health             Run health checks
    help               Show this help message

Examples:
    ./docker-setup.sh setup development    # Setup for development
    ./docker-setup.sh setup production     # Setup for production
    ./docker-setup.sh start basic         # Start only core services
    ./docker-setup.sh logs hft-trading    # Show trading service logs
    ./docker-setup.sh cleanup             # Clean everything

Default profile is 'basic' if not specified.
EOF
}

# Main script logic
main() {
    local command=${1:-"help"}
    local profile=${2:-"basic"}
    
    case $command in
        "setup")
            check_docker
            check_docker_compose
            create_directories
            create_config_files
            build_images
            start_services "$profile"
            health_check
            show_status
            print_success "Setup completed successfully!"
            print_info "Access the system at http://localhost:8080"
            ;;
        "build")
            check_docker
            check_docker_compose
            build_images
            ;;
        "start")
            check_docker
            check_docker_compose
            start_services "$profile"
            ;;
        "stop")
            check_docker_compose
            $COMPOSE_CMD down
            print_success "Services stopped"
            ;;
        "status")
            check_docker_compose
            show_status
            ;;
        "logs")
            check_docker_compose
            local service=${2:-"hft-trading"}
            $COMPOSE_CMD logs -f "$service"
            ;;
        "cleanup")
            check_docker_compose
            cleanup
            ;;
        "health")
            check_docker_compose
            health_check
            ;;
        "help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"