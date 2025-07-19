Advanced Commute Time Analysis Tool (Hybrid Intelligence Version)

A sophisticated tool for analyzing optimal residential locations based on commute times. This enhanced version uses a Hybrid Traffic Intelligence System, combining the power of Google Maps traffic data with the speed and cost-effectiveness of a self-hosted OSRM (Open Source Routing Machine) server.

This approach reduces API costs by over 90% while maintaining high accuracy for peak-hour commute analysis.

✨ Features

    Hybrid Traffic Intelligence: Samples real-world traffic patterns and applies them to a free, local routing engine. 

💰 90%+ Cost Reduction: Drastically cuts Google Maps API costs by replacing per-query calls with periodic, strategic sampling. 

🚀 High-Performance Routing: Leverages the speed of OSRM for analyzing hundreds of points in seconds. 

Smart Calibration: Automatically calibrates OSRM's baseline against Google's off-peak data to ensure accuracy. 

Fully Automated Setup: Helper scripts guide you through setting up the intelligence system and the OSRM server. 

🔧 Full Setup and Usage: A Step-by-Step Guide

Follow these steps to get the complete system running from scratch.

Step 1: Prerequisites

    Python 3.8+

    Docker: Required for running the OSRM server. Install it from the official Docker website.

    Google Maps API Key: You need a key with the Distance Matrix API and Geocoding API enabled.

Step 2: Install Dependencies

Install the required Python packages.
Bash

pip install -r requirements.txt

Step 3: Create and Edit Configuration

Create your config.json file. You can copy config-example.json to get started. The only essential change is adding your Google API key.
Bash

cp config-example.json config.json
nano config.json

Step 4: Download OpenStreetMap Data

The local routing engine needs map data.

    Download the file: Go to Geofabrik for your region (e.g., Europe -> France -> Provence-Alpes-Côte d'Azur).

    Save the file: Place the downloaded .osm.pbf file into your project's root directory (e.g., provence-alpes-cote-d-azur-latest.osm.pbf).

Step 5: Build the Traffic Intelligence Cache (One-Time Cost)

This step makes a small number of calls to the Google API to learn about traffic patterns in your area. This will have a small one-time cost.
Bash

# Check the estimated cost first (optional, free)
python3 hybrid_commute_analyzer.py --estimate-cost

# Run the setup to build the cache
python3 hybrid_commute_analyzer.py --setup-system

Step 6: Build the OSRM Routing Data (Free)

This step uses your generated traffic profiles to process the map data.
Bash

# Make the script executable
chmod +x setup_osrm.sh

# Run the build process for the "morning_rush" profile
./setup_osrm.sh provence-alpes-cote-d-azur-latest.osm.pbf morning_rush

Step 7: Run the OSRM Server (Free)

The previous step will output a docker run... command. Run this command in a new terminal.
Bash

# This starts the routing server and will occupy the terminal
docker run -t -i -p 5000:5000 -v "$(pwd):/data" osrm/osrm-backend osrm-routed --algorithm mld /data/provence-alpes-cote-d-azur-latest.osrm

Leave this terminal running. Your local server is now active.

Step 8: Run the Final Analysis (Free)

Go back to your original terminal. You can now run the analysis as many times as you like with zero API cost.
Bash

python3 hybrid_commute_analyzer.py

The script will use your local OSRM server, apply the cached traffic intelligence, and save the output map and CSV file to the results/ directory.