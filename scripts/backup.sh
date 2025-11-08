#!/bin/bash
# backup.sh - System backup script

echo "💾 HFT System Backup"
echo "===================="

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR..."

# Backup logs
if [ -d "logs" ]; then
    cp -r logs "$BACKUP_DIR/"
    echo "✅ Logs backed up"
fi

# Backup reports  
if [ -d "reports" ]; then
    cp -r reports "$BACKUP_DIR/"
    echo "✅ Reports backed up"
fi

# Backup configuration
cp docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  docker-compose.yml not found"
cp Dockerfile "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  Dockerfile not found"
cp requirements.txt "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  requirements.txt not found"
cp *.py "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  Python files not found"
echo "✅ Configuration backed up"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "✅ Backup complete: $BACKUP_DIR.tar.gz"
echo "📊 Backup size: $(du -h $BACKUP_DIR.tar.gz | cut -f1)"

---

#!/bin/bash
# update.sh - System update script (for when you modify strategies)

echo "🔄 HFT System Update"
echo "==================="

echo "💾 Creating backup before update..."
./backup.sh

echo "🛑 Stopping services..."
docker-compose down

echo "🔨 Rebuilding with latest changes..."
docker-compose build --no-cache

echo "🚀 Restarting services..."
docker-compose up -d

echo "⏳ Waiting for services to stabilize..."
sleep 20

echo "🔍 Checking system health..."
docker-compose ps

# Test system
if docker-compose exec -T hft-system python -c "import yfinance as yf; print('System OK')" 2>/dev/null; then
    echo "✅ Update successful - System is healthy"
else
    echo "⚠️  Update completed but system health check failed"
fi

echo "🎉 Update complete!"

---

#!/bin/bash
# run-strategy.sh - Run specific trading strategies

echo "🎯 HFT Strategy Runner"
echo "====================="

# Default parameters
MODE="balanced"
DURATION=300  # 5 minutes
SYMBOLS="expanded"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2" 
            shift 2
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --fast)
            MODE="fast"
            DURATION=120
            shift
            ;;
        --production)
            MODE="production"
            DURATION=3600
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--mode fast|balanced|production] [--duration seconds] [--symbols expanded|tech|finance] [--fast] [--production]"
            exit 1
            ;;
    esac
done

echo "🚀 Running HFT strategy with:"
echo "   Mode: $MODE"
echo "   Duration: $DURATION seconds"
echo "   Symbols: $SYMBOLS"
echo ""

# Check if system is running
if ! docker-compose ps | grep -q "hft-production.*Up"; then
    echo "❌ HFT system is not running. Starting it first..."
    docker-compose up -d hft-system
    echo "⏳ Waiting for system to start..."
    sleep 10
fi

# Run the strategy
echo "🎯 Executing strategy..."
docker-compose exec hft-system python phase3_complete_integration.py \
    --mode "$MODE" \
    --duration "$DURATION" \
    --symbols "$SYMBOLS"

echo ""
echo "✅ Strategy execution completed!"
echo "📊 Check results in:"
echo "   - Logs: docker-compose logs hft-system"
echo "   - Reports: ./reports/ directory"
echo "   - Monitoring: http://localhost:3000"

---

#!/bin/bash
# quick-test.sh - Quick system test

echo "🧪 Quick HFT System Test"
echo "========================"

if ! docker-compose ps | grep -q "hft-production.*Up"; then
    echo "🚀 Starting system..."
    docker-compose up -d
    sleep 15
fi

echo "🧪 Running 1-minute test..."
docker-compose exec hft-system python phase3_complete_integration.py --mode fast --duration 60

echo "✅ Test complete!"