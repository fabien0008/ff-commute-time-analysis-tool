#!/usr/bin/env python3
"""
Traffic Calibrator - Analyzes Google samples and calculates traffic multipliers
"""

import logging
from datetime import datetime, timezone
from typing import Dict
from statistics import mean
from traffic_cache import TrafficCache
import requests

logger = logging.getLogger(__name__)


class OSRMClient:
    """Simple OSRM client for baseline route queries"""
    def __init__(self, osrm_url: str = "http://localhost:5000"):
        self.osrm_url = osrm_url

    def get_route_time(self, origin: Dict, destination: Dict) -> float:
        """Get route time from OSRM in minutes"""
        coords = f"{origin['lng']},{origin['lat']};{destination['lng']},{destination['lat']}"
        url = f"{self.osrm_url}/route/v1/driving/{coords}?overview=false"
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 'Ok':
                return data['routes'][0]['duration'] / 60
        except requests.RequestException:
            from haversine import haversine
            dist_km = haversine((origin['lat'], origin['lng']), (destination['lat'], destination['lng']))
            return (dist_km / 40) * 60
        return 0.0

class TrafficCalibrator:
    """Calibrates traffic patterns from Google API samples"""

    def __init__(self, cache_dir: str = ".traffic_intelligence_cache"):
        self.cache = TrafficCache(cache_dir)
        self.osrm_client = OSRMClient()

    def run_calibration_analysis(self, force_update: bool = False):        
        """Run complete calibration analysis"""
        logger.info("🧮 Starting traffic calibration analysis...")
        if not force_update and not self.cache.should_update('traffic_multipliers'):        
            logger.info("Traffic multipliers cache is fresh, skipping calibration.")
            return {'status': 'skipped'}
        
        try:
            samples = self.cache.load_json('strategic_routes/route_samples.json')
            baseline_analysis = self._analyze_baseline_correlation(samples)
            traffic_analysis = self._analyze_traffic_impact(samples)
            multipliers = self._generate_traffic_multipliers(baseline_analysis, traffic_analysis)
            self._save_calibration_results(multipliers)
            logger.info("✅ Traffic calibration complete!")
            return {'status': 'success', 'multipliers': multipliers}
        except Exception as e:
            logger.error(f"❌ Calibration failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    def _analyze_baseline_correlation(self, samples: Dict) -> Dict:
        """Compares OSRM and Google during off-peak times."""
        corrections = {}
        for key, sample in samples.get('baseline_samples', {}).items():
            road_type = sample['road_type']
            if road_type not in corrections:
                corrections[road_type] = []
            
            google_time = sample['duration_min']
            osrm_time = self.osrm_client.get_route_time(sample['google_request']['origin'], sample['google_request']['destination'])
            if osrm_time > 1: # Avoid division by zero or near-zero
                corrections[road_type].append(google_time / osrm_time)
        
        return {
            road: {'correction_factor': mean(ratios) if ratios else 1.0}
            for road, ratios in corrections.items()
        }

    def _analyze_traffic_impact(self, samples: Dict) -> Dict:
        """Calculates traffic impact during rush hours."""
        baseline_times = {
            sample['route_id']: sample['duration_min']
            for sample in samples.get('baseline_samples', {}).values()
        }
        
        impacts = {}
        for key, sample in samples.get('traffic_samples', {}).items():
            road_type = sample['road_type']
            time_window = sample['context']
            impact_key = f"{road_type}_{time_window}"
            if impact_key not in impacts: impacts[impact_key] = []

            baseline = baseline_times.get(sample['route_id'])
            if baseline and baseline > 1:
                impacts[impact_key].append(sample['duration_traffic_min'] / baseline)

        return {
            key: {'multiplier': mean(mults) if mults else 1.0}
            for key, mults in impacts.items()
        }

    def _generate_traffic_multipliers(self, baseline: Dict, traffic: Dict) -> Dict:
        """Generate final multipliers for OSRM"""
        multipliers = {}
        for impact_key, summary in traffic.items():
            road_type, time_window = impact_key.split('_', 1)
            if road_type not in multipliers: multipliers[road_type] = {}
            multipliers[road_type][time_window] = {'multiplier': summary['multiplier']}
        return {
            'metadata': {'generated_at': datetime.now(timezone.utc).isoformat()},
            'baseline_corrections': baseline,
            'road_type_multipliers': multipliers
        }

    def _save_calibration_results(self, multipliers: Dict):
        """Save calibration results to cache"""
        self.cache.update_with_metadata(
            'traffic_multipliers', 'calibration/traffic_multipliers.json', multipliers,
            {'script': 'traffic_calibrator.py', 'reason': 'traffic_calibration'}
        )