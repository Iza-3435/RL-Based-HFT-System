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