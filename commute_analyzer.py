#!/usr/bin/env python3
"""
Advanced Commute Time Analysis Tool

This tool analyzes optimal residential locations based on commute times to two workplaces.
Features:
- Geographic Land Filter: Ensures analyzed points are on land using point-in-polygon algorithm
- Rich Visualization: Color-coded maps showing compromise scores with interactive legends
- Multi-Corridor Seeding: Scans multiple parallel corridors for diverse solutions
- Multi-Day Comparison: Runs analysis across different peak-hour days
- Smart Caching: Reduces API costs through intelligent result caching

Configurable for any geographic region and commute scenario.
"""

import googlemaps
import folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import hashlib
import logging
import argparse
import sys
import warnings
import math
from typing import List, Tuple, Dict, Optional

try:
    from sklearn.cluster import DBSCAN
    from haversine import haversine, Unit
    import holidays
except ImportError:
    print("❌ Error: Missing dependencies. Run 'pip install -r requirements.txt'")
    sys.exit(1)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _validate_config(self):
        """Validate essential configuration parameters."""
        api_key = self.config.get('api', {}).get('google_maps_api_key', '')
        if not api_key or api_key == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
            print("❌ ERROR: Google Maps API key not configured!")
            print("Please edit config.json and set your API key in api.google_maps_api_key")
            print("Get your API key at: https://developers.google.com/maps/documentation/distance-matrix/get-api-key")
            sys.exit(1)
        
        # Validate required sections
        required_sections = ['workplaces', 'geographic_area', 'analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class CommuteAnalyzer:
    """Advanced commute analyzer with configurable parameters."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        api_key = self.config.get('api', 'google_maps_api_key')
        self.gmaps = googlemaps.Client(key=api_key)
        
        self.max_total_budget = self.config.get('api', 'max_budget_eur', 20.0)
        self.cost_per_call = self.config.get('api', 'cost_per_call_eur', 0.0046)
        self.cache = SmartCache(self.config)
        self.all_runs_results = {}
        self.api_call_count, self.total_cost = 0, 0.0
        
        # Load geographic boundaries
        self.land_polygon = self.config.get('geographic_area', 'land_polygon', [])
        
        logger.info("🤖 Advanced Commute Analyzer initialized")

    def run_multi_day_comparison(self, **override_params):
        """Run complete analysis across multiple simulation dates."""
        main_start_time = time.time()
        schedule_helper = ScheduleManager(self.config)
        simulation_dates = schedule_helper.get_simulation_dates()
        
        print("\n" + "="*70)
        print("🚀 LAUNCHING MULTI-DAY COMPARATIVE ANALYSIS (MULTI-CORRIDOR STRATEGY)")
        for date in simulation_dates:
            print(f"   - Simulation for {date.strftime('%A %B %d, %Y')}")
        print("="*70)

        for i, sim_date in enumerate(simulation_dates):
            run_key = sim_date.strftime('%Y-%m-%d')
            print(f"\n--- STARTING ANALYSIS #{i+1}/{len(simulation_dates)} FOR {run_key} ---")
            
            run_config = self._prepare_single_run_config(sim_date, **override_params)
            all_analyzed_points, viable_points = self._run_single_analysis(run_config)
            
            self.all_runs_results[run_key] = {
                'config': run_config, 
                'all_points': all_analyzed_points, 
                'viable_points': viable_points
            }
        
        self._compile_and_present_final_results(main_start_time)

    def _run_single_analysis(self, config: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Execute analysis logic for a single date."""
        all_analyzed_points_in_run, viable_points_in_run = [], []
        
        print("\n" + "="*70)
        print(" PHASE 0: MULTI-CORRIDOR SEEDING")
        print("="*70)
        
        corridor_compromises = []
        offsets = self.config.get('analysis', 'corridor_offsets_km', [0, -2.5, 2.5])
        names = self.config.get('analysis', 'corridor_names', ['Central', 'Southern', 'Northern'])
        
        for offset, name in zip(offsets, names):
            corridor_points = self._run_corridor_scan(config, offset_km=offset, tag=name)
            corridor_compromises.extend(corridor_points)
            all_analyzed_points_in_run.extend(corridor_points)

        corridor_compromises.sort(key=lambda x: x.get('compromise_score', float('inf')))
        best_seeds = corridor_compromises[:10]
        if not best_seeds:
            return [], []

        viable_points_in_run.extend([p for p in best_seeds if p['compromise_score'] == 0])
        
        # Identify zones using DBSCAN clustering
        compromise_zones = self._identify_zones_with_dbscan(best_seeds)
        if not compromise_zones:
            compromise_zones = [{'description': 'Best global compromises', 'locations': best_seeds}]
        
        # Analyze top zones in detail
        max_zones = self.config.get('clustering', 'max_zones_to_analyze', 2)
        for i, zone in enumerate(compromise_zones[:max_zones]):
            zone_bounds = self._calculate_bounds_from_points(zone['locations'])
            analyzed_in_level, viable_in_level = self._run_analysis_level(
                level=f"1.{i+1}", 
                description=f"Exploration - {zone['description']}", 
                bounds=zone_bounds, 
                config=config
            )
            all_analyzed_points_in_run.extend(analyzed_in_level)
            viable_points_in_run.extend(viable_in_level)
            
        return all_analyzed_points_in_run, viable_points_in_run

    def _prepare_single_run_config(self, simulation_date: datetime, **kwargs) -> Dict:
        """Prepare configuration for a single analysis run."""
        config = kwargs.copy()
        schedule_helper = ScheduleManager(self.config)
        scenarios = schedule_helper.get_departure_scenarios(simulation_date)
        
        # Use default scenario if not specified
        scenario_name = config.get('school_scenario', 'person_a_school_drop')
        scenario = scenarios[scenario_name]

        print("\n" + "="*70)
        print("🗓️  FAMILY SCHEDULE SIMULATION")
        print(f"   Date: {simulation_date.strftime('%A %B %d, %Y')}")
        print(f"   Scenario: {scenario['description']}")
        print(f"   Person A departs: {scenario['person_a_departs_at']:%H:%M}")
        print(f"   Person B departs: {scenario['person_b_departs_at']:%H:%M}")
        print("="*70)

        config.update({
            'person_a_departure_time': scenario['person_a_departs_at'],
            'person_b_departure_time': scenario['person_b_departs_at']
        })
        
        # Get workplace coordinates
        person_a_addr = self.config.get('workplaces', 'person_a', 'address')
        person_b_addr = self.config.get('workplaces', 'person_b', 'address')
        config['workplace_coords'] = self._get_workplace_coordinates(person_a_addr, person_b_addr)
        
        # Set constraints
        config['constraints'] = {
            'max_person_a_minutes': self.config.get('workplaces', 'person_a', 'max_commute_minutes'),
            'max_person_b_minutes': self.config.get('workplaces', 'person_b', 'max_commute_minutes')
        }
        
        return config

    def _run_corridor_scan(self, config: Dict, offset_km: float, tag: str) -> List[Dict]:
        """Analyze points along a single corridor (central or offset)."""
        print(f"\n--- Scanning corridor: {tag} ---")
        
        coords1 = config['workplace_coords']['person_a']
        coords2 = config['workplace_coords']['person_b']
        
        lat_deg_per_km = 1.0 / 111.0
        offset_lat = offset_km * lat_deg_per_km
        
        num_points = self.config.get('analysis', 'corridor_points', 30)
        lats = np.linspace(coords1['lat'] - offset_lat, coords2['lat'] - offset_lat, num_points)
        lngs = np.linspace(coords1['lng'], coords2['lng'], num_points)

        corridor_points = []
        for i, point in enumerate(zip(lats, lngs)):
            if not self._is_point_in_polygon(point):
                continue  # Geographic filter
            if self.total_cost >= self.max_total_budget:
                break
                
            result = self._analyze_single_location(point, config)
            if result.get('person_a_commute_min') is not None or result.get('person_b_commute_min') is not None:
                corridor_points.append(result)
                
        return corridor_points
    
    def _run_analysis_level(self, level: str, description: str, bounds: Dict, config: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Execute an analysis level, returning all analyzed points AND viable points."""
        print("\n" + "="*70)
        resolution_km = self.config.get('analysis', 'grid_resolution_km', 2.0)
        print(f" PHASE {level}: {description.upper()} | 📏 Resolution: {resolution_km}km")
        print("="*70)
        
        grid_points = self._generate_analysis_grid(resolution_km, bounds)
        all_analyzed_in_level, newly_found_viable = [], []
        
        print(f" -> Analyzing {len(grid_points)} grid points (after geographic filter)...")
        
        for i, location in enumerate(grid_points):
            if i > 0 and i % 50 == 0:
                print(f"   ... {i}/{len(grid_points)} analyzed.")
            if self.total_cost >= self.max_total_budget:
                break
                
            result = self._analyze_single_location(location, config)
            if result.get('person_a_commute_min') is not None or result.get('person_b_commute_min') is not None:
                all_analyzed_in_level.append(result)
                if result['viable']:
                    newly_found_viable.append(result)
                    
        print(f"✅ Phase {level} completed. {len(newly_found_viable)} new viable points found out of {len(all_analyzed_in_level)} analyzed.")
        return all_analyzed_in_level, newly_found_viable

    def _analyze_single_location(self, location: Tuple[float, float], config: Dict) -> Dict:
        """Analyze a single geographic point and return results dictionary."""
        lat, lng = location
        
        # Get commute times for both people
        person_a_mode = self.config.get('workplaces', 'person_a', 'transport_mode')
        person_b_mode = self.config.get('workplaces', 'person_b', 'transport_mode')
        
        person_a_addr = self.config.get('workplaces', 'person_a', 'address')
        person_b_addr = self.config.get('workplaces', 'person_b', 'address')
        
        person_a_time = self._get_cached_commute_time(
            config['workplace_coords']['person_a'], location,
            config['person_a_departure_time'], person_a_mode, person_a_addr
        )
        
        person_b_time = self._get_cached_commute_time(
            config['workplace_coords']['person_b'], location,
            config['person_b_departure_time'], person_b_mode, person_b_addr
        )
        
        # Check viability
        constraints = config['constraints']
        is_viable = (person_a_time is not None and person_b_time is not None and 
                    person_a_time <= constraints['max_person_a_minutes'] and 
                    person_b_time <= constraints['max_person_b_minutes'])
        
        return {
            'lat': lat, 'lng': lng,
            'person_a_commute_min': person_a_time,
            'person_b_commute_min': person_b_time,
            'viable': is_viable,
            'compromise_score': self._calculate_compromise_score(person_a_time, person_b_time, constraints)
        }

    def _get_cached_commute_time(self, origin_coords: Dict, dest: Tuple, departure_time: datetime, 
                                 mode: str, origin_address: str) -> Optional[int]:
        """Get commute time from cache or via API with better error handling."""
        cached = self.cache.get_distance_matrix(origin_address, dest, mode, departure_time)
        if cached:
            return cached.get('duration_minutes')
            
        try:
            params = {
                "origins": [origin_coords],
                "destinations": [dest],
                "mode": mode
            }
            if mode == "driving":
                params.update({
                    "departure_time": departure_time,
                    "traffic_model": "best_guess"
                })
            
            result = self.gmaps.distance_matrix(**params)
            self.api_call_count += 1
            self.total_cost += self.cost_per_call
            
            elem = result['rows'][0]['elements'][0]
            if result['status'] == 'OK' and elem['status'] == 'OK':
                key = 'duration_in_traffic' if mode == "driving" and 'duration_in_traffic' in elem else 'duration'
                duration_min = elem[key]['value'] // 60
                self.cache.store_distance_matrix(origin_address, dest, mode, departure_time, 
                                               {'duration_minutes': duration_min})
                return duration_min
                
        except Exception as e:
            logger.error(f"API error for {origin_address} -> {dest} in mode {mode}: {e}")
            
        self.cache.store_distance_matrix(origin_address, dest, mode, departure_time, 
                                       {'duration_minutes': None})
        return None

    def _get_workplace_coordinates(self, person_a_address, person_b_address):
        """Get coordinates for both workplaces."""
        return {
            'person_a': self._get_cached_geocoding(person_a_address),
            'person_b': self._get_cached_geocoding(person_b_address)
        }

    def _get_cached_geocoding(self, address):
        """Get geocoding from cache or API."""
        cached = self.cache.get_geocoding(address)
        if cached:
            return cached
            
        logger.info(f"API Call: Geocoding for {address}")
        result = self.gmaps.geocode(address)
        self.api_call_count += 1
        self.total_cost += self.cost_per_call
        
        if not result:
            raise Exception(f"Geocoding failed for: {address}")
            
        geocode_data = result[0]['geometry']['location']
        self.cache.store_geocoding(address, geocode_data)
        return geocode_data
    
    def _calculate_compromise_score(self, person_a_time: int, person_b_time: int, constraints: Dict) -> float:
        """Calculate compromise score based on constraint violations."""
        if person_a_time is None or person_b_time is None:
            return float('inf')
            
        person_a_penalty = max(0, person_a_time - constraints['max_person_a_minutes'])
        person_b_penalty = max(0, person_b_time - constraints['max_person_b_minutes'])
        
        weight_a = self.config.get('compromise_scoring', 'person_a_penalty_weight', 1.5)
        weight_b = self.config.get('compromise_scoring', 'person_b_penalty_weight', 2.0)
        
        return (person_a_penalty * weight_a) + (person_b_penalty * weight_b)
    
    def _is_point_in_polygon(self, point: Tuple[float, float]) -> bool:
        """Determine if a point is inside the defined land polygon."""
        if not self.land_polygon:
            return True  # No filtering if no polygon defined
            
        lat, lng = point
        n = len(self.land_polygon)
        inside = False
        p1_lng, p1_lat = self.land_polygon[0]
        
        for i in range(n + 1):
            p2_lng, p2_lat = self.land_polygon[i % n]
            if lat > min(p1_lat, p2_lat):
                if lat <= max(p1_lat, p2_lat):
                    if lng <= max(p1_lng, p2_lng):
                        if p1_lat != p2_lat:
                            xinters = (lat - p1_lat) * (p2_lng - p1_lng) / (p2_lat - p1_lat) + p1_lng
                        if p1_lng == p2_lng or lng <= xinters:
                            inside = not inside
            p1_lng, p1_lat = p2_lng, p2_lat
            
        return inside

    def _calculate_bounds_from_points(self, points: List[Dict]) -> Dict:
        """Calculate search bounds from a list of points."""
        if not points:
            # Default bounds from config
            geo_config = self.config.get('geographic_area', {})
            center_lat = geo_config.get('center_lat', 43.65)
            center_lng = geo_config.get('center_lng', 7.15)
            margin = 0.1
            return {
                'south': center_lat - margin, 'north': center_lat + margin,
                'west': center_lng - margin, 'east': center_lng + margin
            }
            
        avg_lat = np.mean([p['lat'] for p in points])
        avg_lng = np.mean([p['lng'] for p in points])
        search_radius_km = self.config.get('analysis', 'search_radius_km', 12.0)
        
        lat_margin = search_radius_km / 111.0
        lng_margin = search_radius_km / (111.0 * math.cos(math.radians(avg_lat)))
        
        bounds = {
            'south': avg_lat - lat_margin, 'north': avg_lat + lat_margin,
            'west': avg_lng - lng_margin, 'east': avg_lng + lng_margin
        }
        
        logger.info(f"Creating search area of ~{search_radius_km*2:.0f}km x {search_radius_km*2:.0f}km")
        return bounds

    def _generate_analysis_grid(self, resolution_km: float, bounds: Dict) -> List[Tuple[float, float]]:
        """Generate analysis grid points with geographic filtering."""
        lat_min, lat_max = bounds['south'], bounds['north']
        lng_min, lng_max = bounds['west'], bounds['east']
        
        height_km = haversine((lat_min, lng_min), (lat_max, lng_min))
        width_km = haversine((lat_min, lng_min), (lat_min, lng_max))
        
        height_pts = max(3, int(height_km / resolution_km))
        width_pts = max(3, int(width_km / resolution_km))
        
        lats = np.linspace(lat_min, lat_max, height_pts)
        lngs = np.linspace(lng_min, lng_max, width_pts)
        
        # Apply geographic filtering
        grid_points = [(lat, lng) for lat in lats for lng in lngs]
        land_points = [p for p in grid_points if self._is_point_in_polygon(p)]
        
        logger.info(f"{len(grid_points)} grid points generated, {len(land_points)} kept after geographic filter")
        return land_points

    def _identify_zones_with_dbscan(self, locations: List[Dict]) -> List[Dict]:
        """Identify zones using DBSCAN clustering."""
        eps_km = self.config.get('clustering', 'dbscan_eps_km', 3.0)
        min_samples = self.config.get('clustering', 'dbscan_min_samples', 2)
        
        if len(locations) < min_samples:
            return []
            
        coords = np.array([[loc['lat'], loc['lng']] for loc in locations])
        coords_rad = np.radians(coords)
        eps_rad = eps_km / 6371.0
        
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, 
                   algorithm='ball_tree', metric='haversine').fit(coords_rad)
        
        zones = []
        area_names = self.config.get('geographic_area', 'area_names', {})
        
        for k in set(db.labels_):
            if k == -1:  # Noise points
                continue
                
            cluster_locs = [loc for i, loc in enumerate(locations) if (db.labels_ == k)[i]]
            lats = [loc['lat'] for loc in cluster_locs]
            lngs = [loc['lng'] for loc in cluster_locs]
            
            center_lat, center_lng = np.mean(lats), np.mean(lngs)
            
            zones.append({
                'zone_id': k,
                'count': len(cluster_locs),
                'center_lat': center_lat,
                'center_lng': center_lng,
                'bounds': {
                    'north': max(lats), 'south': min(lats),
                    'east': max(lngs), 'west': min(lngs)
                },
                'avg_person_a_time': np.mean([l['person_a_commute_min'] for l in cluster_locs 
                                            if l.get('person_a_commute_min')]),
                'avg_person_b_time': np.mean([l['person_b_commute_min'] for l in cluster_locs 
                                            if l.get('person_b_commute_min')]),
                'locations': cluster_locs,
                'description': self._get_zone_description(center_lat, center_lng, area_names)
            })
            
        return sorted(zones, key=lambda z: z['count'], reverse=True)
    
    def _get_zone_description(self, lat: float, lng: float, area_names: Dict) -> str:
        """Get descriptive name for a zone based on nearest known area."""
        if not area_names:
            return f"Zone at {lat:.3f}, {lng:.3f}"
            
        nearest_area = min(area_names.keys(), 
                          key=lambda name: haversine((lat, lng), tuple(area_names[name])))
        return f"Near {nearest_area}"

    def _compile_and_present_final_results(self, start_time: float):
        """Compile and present the final analysis results."""
        total_duration_m = (time.time() - start_time) / 60
        
        # Generate output files
        map_prefix = self.config.get('output', 'html_map_prefix', 'commute_analysis')
        map_file = f"{map_prefix}_{int(time.time())}.html"
        csv_file = self.config.get('output', 'csv_results_file', 'viable_locations.csv')
        
        all_points_df = self._create_combined_dataframe()
        self._create_final_map(map_file, all_points_df)
        
        if not all_points_df.empty:
            viable_df = all_points_df[all_points_df['viable'] == True]
            if not viable_df.empty:
                viable_df.to_csv(csv_file, index=False)
        
        # Present results
        print("\n" + "="*70)
        print("🎉 COMPARATIVE ANALYSIS COMPLETED 🎉")
        print("="*70)
        print(f"\n📊 GLOBAL SUMMARY:")
        print(f"   • ⏱️ Duration: {total_duration_m:.1f} min | 💰 Cost: ~€{self.total_cost:.2f} | 📞 API calls: {self.api_call_count}")
        
        print("\n🔍 RESULTS BY SIMULATION DATE:")
        total_viable = 0
        for run_date, data in self.all_runs_results.items():
            num_viable = len(data['viable_points'])
            num_analyzed = len(data['all_points'])
            total_viable += num_viable
            print(f"   • {run_date}: {num_viable} viable points found out of {num_analyzed} analyzed")
            
        if not all_points_df.empty:
            unique_locations = len(all_points_df['lat_lng'].unique())
            print(f"\n- Total of {unique_locations} unique locations analyzed")
            
        print(f"\n📁 RESULT FILES:")
        print(f"   • 🗺️ Interactive map: ./{map_file}")
        if total_viable > 0:
            print(f"   • 📈 CSV data (viable points): ./{csv_file}")
            
        # Recommendations
        if total_viable < 10 and self.all_runs_results:
            print("\n" + "="*70)
            print("💡 INTELLIGENT RECOMMENDATION")
            first_config = list(self.all_runs_results.values())[0]['config']['constraints']
            max_a = first_config['max_person_a_minutes']
            max_b = first_config['max_person_b_minutes']
            print("   Very few points meet your constraints. To get more results, consider relaxing criteria:")
            print(f"   Edit config.json: person_a max_commute_minutes to {max_a + 5}, person_b to {max_b + 5}")
            print("="*70)
            
        self.cache.finalize()

    def _score_to_color(self, score: float) -> str:
        """Convert compromise score to color."""
        color_scheme = self.config.get('compromise_scoring', 'color_scheme', {})
        thresholds = self.config.get('compromise_scoring', 'score_thresholds', {})
        
        if score == 0:
            return color_scheme.get('perfect', '#28a745')
        elif score <= thresholds.get('good_max', 10):
            return color_scheme.get('good', '#fdbe41')
        elif score <= thresholds.get('medium_max', 25):
            return color_scheme.get('medium', '#ff7700')
        else:
            return color_scheme.get('poor', '#dc3545')

    def _create_final_map(self, output_file: str, all_points_df: pd.DataFrame):
        """Create the final interactive map visualization."""
        if not self.all_runs_results:
            return
            
        first_run_config = list(self.all_runs_results.values())[0]['config']
        
        # Map center
        geo_config = self.config.get('geographic_area', {})
        center_lat = geo_config.get('center_lat', 43.65)
        center_lng = geo_config.get('center_lng', 7.15)
        zoom_level = geo_config.get('zoom_level', 11)
        
        if not all_points_df.empty:
            center_lat = all_points_df['lat'].mean()
            center_lng = all_points_df['lng'].mean()
            
        m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_level, tiles='OpenStreetMap')
        
        # Create legend
        constraints = first_run_config['constraints']
        person_a_name = self.config.get('workplaces', 'person_a', 'name', 'Person A')
        person_b_name = self.config.get('workplaces', 'person_b', 'name', 'Person B')
        person_a_emoji = self.config.get('workplaces', 'person_a', 'emoji', '🚴')
        person_b_emoji = self.config.get('workplaces', 'person_b', 'emoji', '🚗')
        
        legend_width = self.config.get('output', 'map_legend_width_px', 340)
        
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: {legend_width}px; background-color: rgba(255, 255, 255, 0.9); border:2px solid grey; z-index:9999; font-size:13px; padding: 10px; border-radius: 8px; font-family: Arial, sans-serif;">
        <h4><center>Commute Time Analysis</center></h4>
        <p>This map shows the "quality" of each location based on commute times.</p><hr>
        <strong><u>Time Constraints:</u></strong>
        <ul>
        <li><b>{person_a_emoji} {person_a_name}:</b> ≤ <strong>{constraints['max_person_a_minutes']} min</strong></li>
        <li><b>{person_b_emoji} {person_b_name}:</b> ≤ <strong>{constraints['max_person_b_minutes']} min</strong></li>
        </ul>
        <strong><u>Quality Scale (Compromise Score):</u></strong>
        <p>Point color indicates constraint compliance. Score of 0 is perfect.</p>
        <ul>
        <li><span style="font-size: 18px; color: #28a745; vertical-align: middle;"><b>&#9733;</b></span> <b>Green Star:</b> Perfect (meets both constraints)</li>
        <li><span style="background-color: #fdbe41; border-radius: 50%; display: inline-block; width: 12px; height: 12px;"></span> <b>Yellow:</b> Good compromise (slight excess)</li>
        <li><span style="background-color: #ff7700; border-radius: 50%; display: inline-block; width: 12px; height: 12px;"></span> <b>Orange:</b> Medium compromise</li>
        <li><span style="background-color: #dc3545; border-radius: 50%; display: inline-block; width: 12px; height: 12px;"></span> <b>Red:</b> Poor compromise</li>
        </ul>
        <p><i>Map shows results from first simulation date ({list(self.all_runs_results.keys())[0]}).</i></p>
        </div>'''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add workplace markers
        person_a_coords = first_run_config['workplace_coords']['person_a']
        person_b_coords = first_run_config['workplace_coords']['person_b']
        
        person_a_color = self.config.get('workplaces', 'person_a', 'marker_color', 'green')
        person_a_icon = self.config.get('workplaces', 'person_a', 'marker_icon', 'info-sign')
        person_b_color = self.config.get('workplaces', 'person_b', 'marker_color', 'red')
        person_b_icon = self.config.get('workplaces', 'person_b', 'marker_icon', 'info-sign')
        
        folium.Marker([person_a_coords['lat'], person_a_coords['lng']], 
                     popup=f"{person_a_emoji} {person_a_name}",
                     icon=folium.Icon(color=person_a_color, icon=person_a_icon, prefix='fa')).add_to(m)
        
        folium.Marker([person_b_coords['lat'], person_b_coords['lng']], 
                     popup=f"{person_b_emoji} {person_b_name}",
                     icon=folium.Icon(color=person_b_color, icon=person_b_icon, prefix='fa')).add_to(m)

        # Add analysis points
        if not all_points_df.empty:
            df_to_plot = all_points_df[all_points_df['simulation_date'] == list(self.all_runs_results.keys())[0]]
            
            for _, row in df_to_plot.iterrows():
                score = row['compromise_score']
                is_viable = row['viable']
                
                popup_html = f"""<b>📍 Location ({row['lat']:.4f}, {row['lng']:.4f})</b><br><hr>
                {person_a_emoji}: {row['person_a_commute_min']:.0f}min, {person_b_emoji}: {row['person_b_commute_min']:.0f}min<br>
                Score: {score:.1f}"""
                
                if is_viable:
                    folium.Marker([row['lat'], row['lng']], 
                                popup=folium.Popup(popup_html, max_width=300),
                                icon=folium.Icon(color='green', icon='star')).add_to(m)
                else:
                    color = self._score_to_color(score)
                    folium.CircleMarker([row['lat'], row['lng']], radius=5,
                                      popup=folium.Popup(popup_html, max_width=300),
                                      color=color, fill=True, fill_color=color, 
                                      fill_opacity=0.7).add_to(m)
        
        m.save(output_file)

    def _create_combined_dataframe(self) -> pd.DataFrame:
        """Create combined dataframe from all analysis runs."""
        all_dfs = []
        for run_date, data in self.all_runs_results.items():
            if data['all_points']:
                df = pd.DataFrame(data['all_points'])
                df['simulation_date'] = run_date
                all_dfs.append(df)
                
        if not all_dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        if combined_df.empty:
            return combined_df
            
        combined_df['lat_lng'] = combined_df.apply(lambda row: f"{row['lat']:.4f},{row['lng']:.4f}", axis=1)
        return combined_df


class SmartCache:
    """Smart caching system for expensive API calls."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.cache_dir = config.get('caching', 'cache_directory', '.commute_cache')
        self.hits = {'geocoding': 0, 'distance': 0}
        self.misses = {'geocoding': 0, 'distance': 0}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        geocoding_file = config.get('caching', 'geocoding_cache_file', 'geocoding.json')
        distance_file = config.get('caching', 'distance_cache_file', 'distance.json')
        
        self.geocoding_cache = self._load_cache(geocoding_file)
        self.distance_cache = self._load_cache(distance_file)
    
    def _load_cache(self, filename: str) -> Dict:
        """Load cache from file."""
        path = os.path.join(self.cache_dir, filename)
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}
    
    def _save_cache(self, data: Dict, filename: str):
        """Save cache to file."""
        try:
            path = os.path.join(self.cache_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving cache {filename}: {e}")
    
    def get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        return hashlib.md5(json.dumps(sorted(kwargs.items())).encode()).hexdigest()
    
    def get_geocoding(self, address: str) -> Optional[Dict]:
        """Get geocoding result from cache."""
        key = self.get_cache_key(address=address.strip().lower())
        if key in self.geocoding_cache:
            self.hits['geocoding'] += 1
            return self.geocoding_cache[key]
        self.misses['geocoding'] += 1
        return None
    
    def store_geocoding(self, address: str, result: Dict):
        """Store geocoding result in cache."""
        key = self.get_cache_key(address=address.strip().lower())
        self.geocoding_cache[key] = result
    
    def get_distance_matrix(self, origin: str, dest: Tuple, mode: str, time: datetime) -> Optional[Dict]:
        """Get distance matrix result from cache."""
        params = {
            'origin': origin,
            'dest': f"{dest[0]:.5f},{dest[1]:.5f}",
            'mode': mode
        }
        if mode == "driving":
            params['hour'] = time.hour
            
        key = self.get_cache_key(**params)
        if key in self.distance_cache:
            self.hits['distance'] += 1
            return self.distance_cache[key]
        self.misses['distance'] += 1
        return None
    
    def store_distance_matrix(self, origin: str, dest: Tuple, mode: str, time: datetime, result: Dict):
        """Store distance matrix result in cache."""
        params = {
            'origin': origin,
            'dest': f"{dest[0]:.5f},{dest[1]:.5f}",
            'mode': mode
        }
        if mode == "driving":
            params['hour'] = time.hour
            
        key = self.get_cache_key(**params)
        self.distance_cache[key] = result
    
    def finalize(self):
        """Save caches and display statistics."""
        geocoding_file = self.config.get('caching', 'geocoding_cache_file', 'geocoding.json')
        distance_file = self.config.get('caching', 'distance_cache_file', 'distance.json')
        
        self._save_cache(self.geocoding_cache, geocoding_file)
        self._save_cache(self.distance_cache, distance_file)
        
        total = self.hits['distance'] + self.misses['distance']
        hit_rate = (self.hits['distance'] / total * 100) if total > 0 else 0
        logger.info(f"💾 Cache finalized. Hit rate (routes): {hit_rate:.1f}%")


class ScheduleManager:
    """Manages scheduling and holiday calculations."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        year = config.get('scheduling', 'simulation_year', 2025)
        self.holidays = holidays.FR(years=year)
    
    def is_school_day(self, date: datetime) -> bool:
        """Check if date is a school day."""
        if date.weekday() >= 5:  # Weekend
            return False
        return date not in self.holidays
    
    def get_simulation_dates(self) -> List[datetime]:
        """Get simulation dates based on configuration."""
        dates = []
        target_months = self.config.get('scheduling', 'target_months', [10, 11, 9, 5])
        target_weekdays = self.config.get('scheduling', 'target_weekdays', [0, 1])
        num_dates = self.config.get('scheduling', 'num_simulation_dates', 2)
        year = self.config.get('scheduling', 'simulation_year', 2025)
        
        for month in target_months:
            if len(dates) >= num_dates:
                break
                
            date = datetime(year, month, 1)
            while date.month == month:
                if self.is_school_day(date) and date.weekday() in target_weekdays:
                    dates.append(date)
                    break
                date += timedelta(days=1)
                
        return dates[:num_dates]
    
    def get_departure_scenarios(self, base_date: datetime) -> Dict:
        """Get departure scenarios for the given date."""
        scenarios = {}
        
        for scenario_key, scenario_config in self.config.get('family_scenarios', {}).items():
            scenarios[scenario_key] = {
                'description': scenario_config['description'],
                'person_a_departs_at': base_date.replace(
                    hour=scenario_config['person_a_departure_hour'],
                    minute=scenario_config['person_a_departure_minute']
                ),
                'person_b_departs_at': base_date.replace(
                    hour=scenario_config['person_b_departure_hour'],
                    minute=scenario_config['person_b_departure_minute']
                )
            }
            
        return scenarios


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Advanced commute time analysis tool")
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--scenario', type=str, default='person_a_school_drop',
                       help='School drop-off scenario to use')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear existing cache before analysis')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config_manager = ConfigManager(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Clear cache if requested
    if args.clear_cache:
        import shutil
        cache_dir = config_manager.get('caching', 'cache_directory', '.commute_cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"🗑️ Cache '{cache_dir}' cleared")
    
    try:
        analyzer = CommuteAnalyzer(config_manager)
        analyzer.run_multi_day_comparison(school_scenario=args.scenario)
        
    except Exception as e:
        logger.error(f"Critical error occurred: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()