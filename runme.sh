#!/bin/bash

# Quick start script for Commute Time Analysis Tool
# This script runs the analysis with the default configuration

echo "🚀 Starting Commute Time Analysis..."
echo

# Check if config file exists
if [ ! -f "config.json" ]; then
    echo "❌ Configuration file not found!"
    echo "Please copy config-example.json to config.json and customize it:"
    echo "  cp config-example.json config.json"
    echo "  # Edit config.json with your API key and addresses"
    exit 1
fi

# Check if Python dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import googlemaps, folium, pandas, numpy, sklearn, haversine, holidays" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies! Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please install manually:"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

echo "✅ Dependencies OK"
echo "🔄 Running analysis..."
echo

# Run the analysis
python3 commute_analyzer.py

echo
echo "✅ Analysis complete! Check the generated files."
