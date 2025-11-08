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