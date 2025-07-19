#!/bin/bash
#
# This script clears all locally generated intelligence (calibration, profiles)
# but KEEPS the expensive Google API samples, allowing for a fast, free rebuild.
#

echo "🔥 Clearing local caches..."

# Remove the generated traffic multipliers
if [ -f ".traffic_intelligence_cache/calibration/traffic_multipliers.json" ]; then
    rm .traffic_intelligence_cache/calibration/traffic_multipliers.json
    echo "  - Removed calibration data."
fi

# Remove the generated LUA profiles for driving (leaves bike.lua alone)
if [ -f "profiles/car_traffic_morning_rush.lua" ]; then
    rm profiles/car_traffic_*.lua
    echo "  - Removed generated OSRM profiles."
fi

echo "✅ Local caches cleared. You can now re-run the setup."