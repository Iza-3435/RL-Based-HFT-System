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