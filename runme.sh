#!/bin/bash
# Smart Run Script (v6 - Final)
set -e

# --- Configuration ---
REGION_NAME="provence-alpes-cote-d-azur-latest"
PBF_FILE="${REGION_NAME}.osm.pbf"
CAR_PROFILE_NAME="morning_rush"

# --- NEW: Point to the permanent 'profiles' directory ---
CAR_PROFILE_LUA="profiles/car_traffic_${CAR_PROFILE_NAME}.lua"
BIKE_PROFILE_LUA="profiles/bike.lua"

CAR_CONTAINER="osrm_server_car"
BIKE_CONTAINER="osrm_server_bike"

# --- Helper Functions ---
function print_step {
    echo -e "\n========================================================================\nSTEP $1: $2\n========================================================================"
}

# --- Main Script ---
print_step 1 "Checks & Installation"
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ Dependencies and configuration are OK."

print_step 2 "Building Traffic Intelligence Cache"
python3 hybrid_commute_analyzer.py --setup-system

print_step 3 "Building OSRM Data for DRIVING"
./setup_osrm.sh "$PBF_FILE" "$CAR_PROFILE_LUA" "$CAR_PROFILE_NAME"

print_step 4 "Building OSRM Data for BICYCLING"
./setup_osrm.sh "$PBF_FILE" "$BIKE_PROFILE_LUA" "bike"

print_step 5 "Launching OSRM Servers"
# Clean up any old containers
docker stop $CAR_CONTAINER $BIKE_CONTAINER > /dev/null 2>&1 || true
docker rm $CAR_CONTAINER $BIKE_CONTAINER > /dev/null 2>&1 || true
echo "🚀 Starting DRIVING server (port 5000) and BICYCLING server (port 5001)..."
docker run -d --name "$CAR_CONTAINER" -p 5000:5000 -v "$(pwd):/data" osrm/osrm-backend osrm-routed --algorithm mld "/data/${REGION_NAME}-${CAR_PROFILE_NAME}.osrm" > /dev/null
docker run -d --name "$BIKE_CONTAINER" -p 5001:5000 -v "$(pwd):/data" osrm/osrm-backend osrm-routed --algorithm mld "/data/${REGION_NAME}-bike.osrm" > /dev/null
echo "⏳ Waiting for servers to initialize..."
sleep 5
echo "✅ OSRM servers launched."

print_step 6 "Running Final Analysis"
python3 hybrid_commute_analyzer.py

echo -e "\n🎉 All steps completed successfully!"
echo "To stop servers, run: docker stop $CAR_CONTAINER $BIKE_CONTAINER"