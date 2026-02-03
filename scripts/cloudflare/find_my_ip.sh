#!/bin/bash

# Quick script to find your IPv4 addresses

echo "🔍 Finding Your IPv4 Addresses..."
echo "===================================="
echo ""

# Find local IP
echo "📡 Local/Private IPv4 Address:"
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [ -n "$LOCAL_IP" ]; then
    echo "   $LOCAL_IP"
else
    echo "   Could not determine local IP"
fi
echo ""

# Find public IP
echo "🌐 Public IPv4 Address:"
PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || curl -s --max-time 5 icanhazip.com 2>/dev/null || curl -s --max-time 5 api.ipify.org 2>/dev/null)
if [ -n "$PUBLIC_IP" ]; then
    echo "   $PUBLIC_IP"
else
    echo "   Could not determine public IP (check internet connection)"
fi
echo ""

# Show all network interfaces
echo "📋 All Network Interfaces:"
ip addr show | grep -E "inet [0-9]" | grep -v "127.0.0.1" | while read line; do
    IP=$(echo "$line" | awk '{print $2}' | cut -d'/' -f1)
    INTERFACE=$(echo "$line" | awk '{print $NF}')
    echo "   $IP ($INTERFACE)"
done
echo ""

# Show default gateway
echo "🚪 Default Gateway (Router):"
GATEWAY=$(ip route | grep default | awk '{print $3}' | head -n 1)
if [ -n "$GATEWAY" ]; then
    echo "   $GATEWAY"
else
    echo "   Could not determine gateway"
fi
echo ""

echo "===================================="
echo "✅ Done!"
