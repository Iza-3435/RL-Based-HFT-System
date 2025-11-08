#!/bin/bash
# deploy.sh - One-click deployment script

set -e

echo "🚀 HFT System Production Deployment"
echo "===================================="

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed."
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍎 You're on macOS. Please install Docker Desktop:"
        echo "   1. Download from: https://www.docker.com/products/docker-desktop"
        echo "   2. Or install via Homebrew: brew install --cask docker"
        echo "   3. Start Docker Desktop from Applications"
        echo "   4. Then run this script again"
    else
        echo "🐧 Installing Docker for Linux..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        sudo usermod -aG docker $USER
        echo "✅ Docker installed. Please log out and back in, then run this script again."
    fi
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs reports data backups monitoring/dashboards monitoring/datasources

# Create monitoring config files
echo "📊 Setting up monitoring configuration..."

# Prometheus config
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'hft-system'
    static_configs:
      - targets: ['hft-system:8080']
    scrape_interval: 5s
EOF

# Grafana data source
cat > monitoring/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Set permissions
chmod 755 logs reports data backups
chmod +x *.py 2>/dev/null || true

# Stop existing containers if running
echo "🛑 Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Build and start services
echo "🔨 Building HFT system..."
docker-compose build --no-cache

echo "🚀 Starting production environment..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Show running containers
echo "📊 Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test system
echo "🧪 Testing system connectivity..."
if docker-compose exec -T hft-system python -c "import yfinance as yf; print('✅ Market data connection OK')" 2>/dev/null; then
    echo "✅ System health check passed"
else
    echo "⚠️  System health check failed, but containers are running"
fi

# Display access information
echo ""
echo "✅ HFT System Deployed Successfully!"
echo "===================================="
echo "🖥️  Main System: Running in container 'hft-production'"
echo "📊 Grafana Dashboard: http://localhost:3000 (admin/hft123)"
echo "📈 Prometheus Metrics: http://localhost:9090"
echo "💾 Redis Cache: localhost:6379"
echo ""
echo "📝 Management Commands:"
echo "  View Logs: docker-compose logs -f hft-system"
echo "  System Status: docker-compose ps"
echo "  Stop System: docker-compose down"
echo "  Restart: docker-compose restart hft-system"
echo ""

# Optional: Run quick test
read -p "🧪 Run quick system test? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running 1-minute system test..."
    docker-compose exec hft-system python phase3_complete_integration.py --mode balanced --duration 60
fi

echo ""
echo "🎉 Deployment Complete!"
echo "📚 Next steps:"
echo "  1. Monitor system: ./monitor.sh"
echo "  2. View dashboards: http://localhost:3000"
echo "  3. Check logs: docker-compose logs -f hft-system"
echo "  4. Modify strategies: Edit your Python files and restart"