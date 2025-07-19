#!/bin/bash
# OSRM Setup Script (v3 - Final)
# Builds data for a specific profile and saves it to a unique file.
set -e

PBF_FILE="$1"
PROFILE_LUA_FILE="$2"
PROFILE_NAME="$3" # e.g., "morning_rush" or "bike"

if [[ -z "$PBF_FILE" || -z "$PROFILE_LUA_FILE" || -z "$PROFILE_NAME" ]]; then
    echo "Usage: $0 <map.osm.pbf> <profile.lua> <profile_name>"
    exit 1
fi

OSRM_DATA_FILE="${PBF_FILE%.osm.pbf}-${PROFILE_NAME}.osrm"
echo "🛠️  Building OSRM data for '$PROFILE_NAME' profile..."

# Run OSRM processing steps
docker run --rm -t -v "$(pwd):/data" osrm/osrm-backend osrm-extract -p "/data/$PROFILE_LUA_FILE" "/data/$PBF_FILE"
docker run --rm -t -v "$(pwd):/data" osrm/osrm-backend osrm-partition "/data/${PBF_FILE%.osm.pbf}.osrm"
docker run --rm -t -v "$(pwd):/data" osrm/osrm-backend osrm-customize "/data/${PBF_FILE%.osm.pbf}.osrm"

# Rename the final files to be profile-specific
for f in $(ls ${PBF_FILE%.osm.pbf}.osrm*); do
    # Use mv -f to force overwrite without prompting
    mv -f -- "$f" "${f/${PBF_FILE%.osm.pbf}.osrm/${OSRM_DATA_FILE}}"
done

echo "✅ OSRM data build complete: ${OSRM_DATA_FILE}"