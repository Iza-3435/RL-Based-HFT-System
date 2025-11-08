#!/bin/bash
# monitor.sh - Real-time system monitoring

echo "📊 HFT System Monitor"
echo "====================="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "📊 HFT System Status - $(date)"
    echo "=================================="
    
    # Container status
    echo "🐳 Container Status:"
    docker-compose ps
    echo ""
    
    # Resource usage
    echo "💻 Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" | head -n 5
    echo ""
    
    # Recent logs (last 3 lines from main system)
    echo "📝 Recent System Logs:"
    docker-compose logs --tail=3 hft-system | tail -n 3
    echo ""
    
    # System health checks
    echo "❤️  Health Status:"
    
    # Check main system
    if docker-compose exec -T hft-system python -c "import yfinance as yf; yf.Ticker('AAPL').info; print('✅ HFT System: OK')" 2>/dev/null; then
        echo "✅ HFT System: Healthy"
    else
        echo "❌ HFT System: Error"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo "✅ Redis Cache: Connected"
    else
        echo "❌ Redis Cache: Error"
    fi
    
    # Check Grafana
    if curl -s http://localhost:3000/api/health 2>/dev/null | grep -q "ok"; then
        echo "✅ Grafana: Available at http://localhost:3000"
    else
        echo "❌ Grafana: Not responding"
    fi
    
    # Check Prometheus
    if curl -s http://localhost:9090/-/healthy 2>/dev/null | grep -q "Prometheus"; then
        echo "✅ Prometheus: Available at http://localhost:9090"
    else
        echo "❌ Prometheus: Not responding"
    fi
    
    echo ""
    echo "🔧 Quick Commands:"
    echo "  View full logs: docker-compose logs -f hft-system"
    echo "  Restart system: docker-compose restart hft-system"
    echo "  Stop all: docker-compose down"
    echo ""
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done