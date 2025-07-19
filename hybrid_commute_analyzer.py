#!/usr/bin/env python3
"""
Hybrid Commute Time Analysis Tool
Advanced commute analyzer with traffic intelligence using OSRM + Google samples
"""

import json
import os
import time
import logging
import argparse
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
import folium
import requests
from haversine import haversine
from shapely.geometry import Point, Polygon
import numpy as np

# --- Local Hybrid System Imports ---
from traffic_cache import TrafficCache
from cache_interface import CacheInterface
from traffic_sampler import TrafficSampler
from traffic_calibrator import TrafficCalibrator
from osrm_profile_generator import OSRMProfileGenerator

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation for the hybrid system."""
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding config.json: {e}.")

    def get(self, *keys, required=True, default=None):
        value = self.config
        path = ""
        for key in keys:
            path += f"['{key}']"
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                if required:
                    raise KeyError(f"Required configuration key not found: config{path}")
                return default
        return value

# In hybrid_commute_analyzer.py

class OSRMClient:
    """OSRM client that queries different ports for different modes."""
    def __init__(self, car_url: str, bike_url: str):
        self.car_url = car_url
        self.bike_url = bike_url

    def get_route_time(self, origin: Dict, destination: Dict, mode: str) -> Optional[float]:
        """Get route time from the correct OSRM server based on mode."""
        
        # Choose the correct server URL
        server_url = self.bike_url if mode == 'bicycling' else self.car_url
        osrm_profile = 'bike' if mode == 'bicycling' else 'car' # OSRM still needs a profile in the URL path
        
        coords = f"{origin['lng']},{origin['lat']};{destination['lng']},{destination['lat']}"
        url = f"{server_url}/route/v1/{osrm_profile}/{coords}?overview=false"
        
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200 and response.json()['code'] == 'Ok':
                return response.json()['routes'][0]['duration'] / 60
        except requests.RequestException:
            pass
        return None

class HybridCommuteAnalyzer:
    """Main class for running the commute analysis using the hybrid system."""
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        cache_dir = self.config.get('hybrid_system', 'cache_dir', required=False, default=".traffic_intelligence_cache")
        
        # --- THIS LINE WAS MISSING ---
        self.cache_interface = CacheInterface(cache_dir)
        # ---------------------------

        # Initialize the OSRMClient with separate URLs for car and bike
        self.osrm_client = OSRMClient(
            car_url=self.config.get('hybrid_system', 'osrm_url', required=False, default="http://localhost:5000"),
            bike_url=self.config.get('hybrid_system', 'osrm_bike_url', required=False, default="http://localhost:5001")
        )
        
        self.land_polygon = self.config.get('geographic_area', 'land_polygon', required=False, default=[])
        self.results = []

    def run_analysis(self):
        logger.info("🚀 Launching Hybrid Traffic Intelligence Analysis...")
        if not self.osrm_client.get_route_time(
            self.config.get('workplaces', 'person_a', 'coords'),
            self.config.get('workplaces', 'person_b', 'coords'),
            'driving' # Add the missing 'mode' argument
        ):
            logger.error(f"❌ OSRM server not reachable at {self.osrm_client.osrm_url}. Please ensure it is running.")
            sys.exit(1)
            
        workplace_a = self.config.get('workplaces', 'person_a', 'coords')
        workplace_b = self.config.get('workplaces', 'person_b', 'coords')
        bounds = self._calculate_bounds_from_points([workplace_a, workplace_b])
        resolution_km = self.config.get('analysis', 'grid_resolution_km', required=False, default=2.0)
        grid_points = self._generate_analysis_grid(resolution_km, bounds)

        logger.info(f"Analyzing {len(grid_points)} points...")
        departure_time = datetime.now().replace(hour=8, minute=0) 

        for i, point in enumerate(grid_points):
            if i > 0 and i % 50 == 0: logger.info(f"  ...processed {i}/{len(grid_points)} points.")
            analysis_result = self._analyze_single_location_hybrid(point, departure_time)
            if analysis_result: self.results.append(analysis_result)
        
        self._present_final_results()

    def _analyze_single_location_hybrid(self, location: Tuple[float, float], departure_time: datetime) -> Optional[Dict]:
        time_window = self.cache_interface.get_time_window(departure_time)
        person_a_wp = self.config.get('workplaces', 'person_a')
        person_b_wp = self.config.get('workplaces', 'person_b')
        dest = {'lat': location[0], 'lng': location[1]}
        person_a_time = self._get_hybrid_commute_time(person_a_wp['coords'], dest, time_window, person_a_wp['transport_mode'])
        person_b_time = self._get_hybrid_commute_time(person_b_wp['coords'], dest, time_window, person_b_wp['transport_mode'])
        if person_a_time is None or person_b_time is None: return None
        is_viable = (person_a_time <= person_a_wp['max_commute_minutes'] and person_b_time <= person_b_wp['max_commute_minutes'])
        return {'lat': location[0], 'lng': location[1], 'person_a_commute_min': person_a_time, 'person_b_commute_min': person_b_time, 'viable': is_viable}

    def _get_hybrid_commute_time(self, origin: Dict, dest: Dict, time_window: str, mode: str) -> Optional[float]:
        """Calculates commute time using OSRM, applying traffic ONLY for driving."""
        # Get the baseline time from OSRM for the correct travel mode
        osrm_baseline_time = self.osrm_client.get_route_time(origin, dest, mode)
        if osrm_baseline_time is None: return None

        # ONLY apply traffic intelligence if the mode is 'driving'
        if mode == "driving":
            road_type = 'primary'
            traffic_multiplier = self.cache_interface.get_traffic_multiplier(road_type, time_window)
            baseline_correction = self.cache_interface.get_baseline_correction(road_type)
            return osrm_baseline_time * baseline_correction * traffic_multiplier
        else:
            # For bicycling, walking, etc., return the simple baseline time
            return osrm_baseline_time

    def _calculate_bounds_from_points(self, points: List[Dict]) -> Dict:
        if not points:
            return {'south': self.config.get('geographic_area','center_lat')-0.1, 'north': self.config.get('geographic_area','center_lat')+0.1, 'west': self.config.get('geographic_area','center_lng')-0.1, 'east': self.config.get('geographic_area','center_lng')+0.1}
        avg_lat = pd.Series([p['lat'] for p in points]).mean()
        avg_lng = pd.Series([p['lng'] for p in points]).mean()
        search_radius_km = self.config.get('analysis', 'search_radius_km', required=False, default=12.0)
        import math
        lat_margin = search_radius_km / 111.0
        lng_margin = search_radius_km / (111.0 * math.cos(math.radians(avg_lat)))
        return {'south': avg_lat-lat_margin, 'north': avg_lat+lat_margin, 'west': avg_lng-lng_margin, 'east': avg_lng+lng_margin}

    def _generate_analysis_grid(self, resolution_km: float, bounds: Dict) -> List[Tuple[float, float]]:
        lat_min, lat_max, lng_min, lng_max = bounds['south'], bounds['north'], bounds['west'], bounds['east']
        height_km = haversine((lat_min, lng_min), (lat_max, lng_min))
        width_km = haversine((lat_min, lng_min), (lat_min, lng_max))
        height_pts, width_pts = max(3, int(height_km/resolution_km)), max(3, int(width_km/resolution_km))
        lats, lngs = np.linspace(lat_min, lat_max, height_pts), np.linspace(lng_min, lng_max, width_pts)
        grid_points = [(lat, lng) for lat in lats for lng in lngs]
        if not self.land_polygon: return grid_points
        poly = Polygon(self.land_polygon)
        land_points = [p for p in grid_points if poly.contains(Point(p[1], p[0]))]
        logger.info(f"{len(grid_points)} grid points generated, {len(land_points)} kept after land filter.")
        return land_points

    def _present_final_results(self):
        if not self.results:
            logger.warning("No results to present.")
            return
        df = pd.DataFrame(self.results)
        csv_file = self.config.get('output', 'csv_results_file', required=False, default='results/hybrid_viable_locations.csv')
        viable_df = df[df['viable'] == True]
        if not viable_df.empty:
            logger.info(f"📈 Saved {len(viable_df)} viable locations to {csv_file}")
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            viable_df.to_csv(csv_file, index=False)
        map_file = self.config.get('output', 'html_map_prefix', required=False, default='hybrid_commute_analysis') + ".html"
        center = [self.config.get('geographic_area', 'center_lat'), self.config.get('geographic_area', 'center_lng')]
        m = folium.Map(location=center, zoom_start=self.config.get('geographic_area', 'zoom_level'))
        person_a, person_b = self.config.get('workplaces', 'person_a'), self.config.get('workplaces', 'person_b')
        folium.Marker(location=[person_a['coords']['lat'], person_a['coords']['lng']], popup=f"{person_a.get('emoji','')} {person_a.get('name')}", icon=folium.Icon(color='green', icon='bicycle', prefix='fa')).add_to(m)
        folium.Marker(location=[person_b['coords']['lat'], person_b['coords']['lng']], popup=f"{person_b.get('emoji','')} {person_b.get('name')}", icon=folium.Icon(color='red', icon='car', prefix='fa')).add_to(m)
        legend_html = f'''<div style="position: fixed; bottom: 50px; left: 50px; width: 280px; height: 180px; border:2px solid grey; z-index:9999; font-size:14px; background-color: white; opacity: 0.9;">&nbsp; <b>Hybrid Commute Analysis</b><br>&nbsp; <b>Constraints:</b><br>&nbsp; {person_a.get('emoji','')} {person_a.get('name')}: ≤ {person_a.get('max_commute_minutes')} min<br>&nbsp; {person_b.get('emoji','')} {person_b.get('name')}: ≤ {person_b.get('max_commute_minutes')} min<br><hr>&nbsp; <i class="fa fa-circle" style="color:green"></i>&nbsp; Viable Location<br>&nbsp; <i class="fa fa-circle" style="color:red"></i>&nbsp; Non-Viable Location<br></div>'''
        m.get_root().html.add_child(folium.Element(legend_html))
        for _, row in df.iterrows():
            folium.CircleMarker(location=[row['lat'], row['lng']], radius=5, color='green' if row['viable'] else 'red', fill=True, fill_color='green' if row['viable'] else 'red', popup=f"Fabien: {row['person_a_commute_min']:.0f}min, Vesna: {row['person_b_commute_min']:.0f}min").add_to(m)
        m.save(map_file)
        logger.info(f"🗺️  Interactive map saved to ./{map_file}")

# --- NEW: Function to display the calibration report ---
def show_calibration_report():
    """Loads and displays the calibration results from the cache."""
    try:
        cache = TrafficCache(".traffic_intelligence_cache")
        multipliers = cache.load_json('calibration/traffic_multipliers.json')
        
        print("\n" + "="*60)
        print("🔎 Hybrid Intelligence Calibration Report")
        print("="*60)
        
        print("\nOSRM Baseline Correction Factors (vs. Google off-peak):")
        for road, data in multipliers.get('baseline_corrections', {}).items():
            print(f"  - {road.capitalize():<12}: {data.get('correction_factor', 1.0):.2f}x")

        print("\nTraffic Multipliers (Slowdown vs. Off-Peak):")
        for road, windows in multipliers.get('road_type_multipliers', {}).items():
            print(f"  - {road.capitalize()} Roads:")
            for window, data in windows.items():
                print(f"    - {window:<15}: {data.get('multiplier', 1.0):.2f}x slower")
        print("="*60)

    except FileNotFoundError:
        logger.error("❌ Calibration report not found. Please run the setup first with '--setup-system'.")
    except Exception as e:
        logger.error(f"❌ Failed to load calibration report: {e}")

def estimate_api_cost(config_manager: ConfigManager):
    """Calculates and prints the estimated cost for the one-time traffic sampling."""
    # ... (this function remains the same)
    pass

def main():
    """Main function to run setup or analysis."""
    parser = argparse.ArgumentParser(description="Hybrid commute time analysis tool.")
    parser.add_argument('--setup-system', action='store_true', help='Run the one-time setup to build the traffic intelligence cache.')
    parser.add_argument('--estimate-cost', action='store_true', help='Perform a dry run to estimate the one-time Google API cost for setup.')
    # --- NEW: Argument to show the report ---
    parser.add_argument('--report', action='store_true', help='Display the current calibration report from the cache.')
    args = parser.parse_args()

    try:
        config_manager = ConfigManager()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)

    if args.estimate_cost:
        # For brevity, the full estimate_api_cost function is omitted here, but should be included
        print("Cost estimation logic would run here.")
        sys.exit(0)
    
    # --- NEW: Handle the report argument ---
    if args.report:
        show_calibration_report()
        sys.exit(0)

    if args.setup_system:
        logger.info("🛠️  Running full system setup...")
        sampler = TrafficSampler(config_manager)
        sampler.run_strategic_sampling()
        calibrator = TrafficCalibrator()
        calibration_results = calibrator.run_calibration_analysis(force_update=True)
        profile_generator = OSRMProfileGenerator()
        profile_generator.generate_traffic_profiles()
        show_calibration_report() # Show the report automatically after setup
        logger.info("✅ System setup complete. You can now build and run the OSRM server.")
    else:
        analyzer = HybridCommuteAnalyzer(config_manager)
        analyzer.run_analysis()

if __name__ == "__main__":
    main()