#!/usr/bin/env python3
"""
Traffic Sampler - Strategic Google Maps API sampling for traffic intelligence
"""

import googlemaps
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from traffic_cache import TrafficCache
import holidays

logger = logging.getLogger(__name__)


class TrafficSampler:
    """Strategic traffic sampling system"""

    def __init__(self, config_manager, cache_dir: str = ".traffic_intelligence_cache"):
        self.config = config_manager
        self.cache = TrafficCache(cache_dir)
        api_key = self.config.get('api', 'Maps_api_key')
        if not api_key or api_key == "YOUR_Maps_API_KEY_HERE":
            raise ValueError("Google Maps API key not configured in config.json!")
        self.gmaps = googlemaps.Client(key=api_key)
        self.api_call_count = 0
        self.total_cost = 0.0
        self.cost_per_call = self.config.get('api', 'cost_per_call_eur', required=False, default=0.005)
        self.holidays = holidays.France(years=datetime.now().year)

    def run_strategic_sampling(self, force_update: bool = False) -> Dict:
        """Execute strategic traffic sampling plan"""

        logger.info("🚀 Starting strategic traffic sampling...")
        if not force_update and not self.cache.should_update('strategic_routes'):
            logger.info("✅ Samples already exist in cache. Skipping Google API calls.")
            return {'status': 'skipped', 'reason': 'cache_fresh'}
        
        strategic_routes = self.get_strategic_routes()
        results = {'baseline_samples': {}, 'traffic_samples': {}, 'errors': []}
        
        try:
            logger.info("📊 Phase 1: Baseline sampling (off-peak hours)...")
            results['baseline_samples'] = self._sample_baseline_traffic(strategic_routes)
            
            logger.info("🚦 Phase 2: Rush hour sampling...")
            results['traffic_samples'] = self._sample_rush_hour_traffic(strategic_routes)
            
            results['api_calls'] = self.api_call_count
            results['total_cost'] = self.total_cost
            self._save_sampling_results(results)
            logger.info(f"✅ Strategic sampling complete! {self.api_call_count} API calls, cost ~€{self.total_cost:.2f}")
        except Exception as e:
            logger.error(f"❌ Strategic sampling failed: {e}", exc_info=True)
            results['errors'].append(str(e))
        
        return results


    def get_strategic_routes(self) -> List[Dict]:
        """Defines strategic routes for the French Riviera"""
        return [
            {'id': 'A8_nice_antibes', 'origin': {'lat': 43.7102, 'lng': 7.2620}, 'destination': {'lat': 43.5804, 'lng': 7.1247}, 'road_type': 'highway'},
            {'id': 'coastal_nice_antibes', 'origin': {'lat': 43.6951, 'lng': 7.2658}, 'destination': {'lat': 43.5828, 'lng': 7.1239}, 'road_type': 'primary'},
            {'id': 'cannes_grasse', 'origin': {'lat': 43.5528, 'lng': 6.9619}, 'destination': {'lat': 43.6584, 'lng': 6.9226}, 'road_type': 'secondary'},
            {'id': 'nice_center_airport', 'origin': {'lat': 43.7034, 'lng': 7.2663}, 'destination': {'lat': 43.6584, 'lng': 7.2159}, 'road_type': 'primary'},
            {'id': 'var_valley_road', 'origin': {'lat': 43.7093, 'lng': 7.2454}, 'destination': {'lat': 43.7667, 'lng': 7.2023}, 'road_type': 'primary'},
            {'id': 'A8_cannes_mandelieu', 'origin': {'lat': 43.5528, 'lng': 6.9619}, 'destination': {'lat': 43.5486, 'lng': 6.9178}, 'road_type': 'highway'},
            {'id': 'moyenne_corniche', 'origin': {'lat': 43.7034, 'lng': 7.2663}, 'destination': {'lat': 43.7278, 'lng': 7.3629}, 'road_type': 'secondary'},
        ]

    # def get_strategic_routes(self) -> List[Dict]:
    #     """Defines strategic routes for the French Riviera"""
    #     return [
    #         {'id': 'A8_nice_antibes', 'origin': {'lat': 43.7102, 'lng': 7.2620}, 'destination': {'lat': 43.5804, 'lng': 7.1247}, 'road_type': 'highway'},
    #         {'id': 'coastal_nice_antibes', 'origin': {'lat': 43.6951, 'lng': 7.2658}, 'destination': {'lat': 43.5828, 'lng': 7.1239}, 'road_type': 'primary'},
    #         {'id': 'cannes_grasse', 'origin': {'lat': 43.5528, 'lng': 6.9619}, 'destination': {'lat': 43.6584, 'lng': 6.9226}, 'road_type': 'secondary'},
    #         {'id': 'nice_center_airport', 'origin': {'lat': 43.7034, 'lng': 7.2663}, 'destination': {'lat': 43.6584, 'lng': 7.2159}, 'road_type': 'primary'}
    #     ]

    def _sample_baseline_traffic(self, routes: List[Dict]) -> Dict:
        """Sample traffic during non-rush hours"""
        samples = {}
        sample_date = self._get_next_weekday(1)
        for route in routes:
            try:
                departure_time = sample_date.replace(hour=11, minute=0)
                sample = self._sample_single_route(route, departure_time, 'baseline')
                samples[route['id']] = sample
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed baseline sample for {route['id']}: {e}")
        return samples

    def _sample_rush_hour_traffic(self, routes: List[Dict]) -> Dict:
        """Sample traffic during rush hours more densely."""
        samples = {}
        sample_date = self._get_next_weekday(1)
        # More detailed time windows
        rush_windows = [
            {'name': 'morning_rush', 'hour': 7, 'minute': 30},
            {'name': 'morning_rush', 'hour': 8, 'minute': 0},
            {'name': 'morning_rush', 'hour': 8, 'minute': 30},
            {'name': 'evening_rush', 'hour': 17, 'minute': 30},
            {'name': 'evening_rush', 'hour': 18, 'minute': 0},
            {'name': 'evening_rush', 'hour': 18, 'minute': 30},
        ]
        
        for route in routes:
            for window in rush_windows:
                try:
                    departure_time = sample_date.replace(hour=window['hour'], minute=window['minute'])
                    # Unique key for each sample time
                    sample_key = f"{route['id']}_{window['name']}_{window['hour']}{window['minute']}"
                    sample = self._sample_single_route(route, departure_time, window['name'])
                    samples[sample_key] = sample
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed traffic sample for {route['id']} at {departure_time.strftime('%H:%M')}: {e}")
        return samples

    # def _sample_rush_hour_traffic(self, routes: List[Dict]) -> Dict:
    #     """Sample traffic during rush hours"""
    #     samples = {}
    #     sample_date = self._get_next_weekday(1)
    #     for route in routes:
    #         for window in [{'name': 'morning_rush', 'hour': 8}, {'name': 'evening_rush', 'hour': 18}]:
    #             try:
    #                 departure_time = sample_date.replace(hour=window['hour'], minute=0)
    #                 sample = self._sample_single_route(route, departure_time, window['name'])
    #                 samples[f"{route['id']}_{window['name']}"] = sample
    #                 time.sleep(0.1)
    #             except Exception as e:
    #                 logger.warning(f"Failed traffic sample for {route['id']}: {e}")
    #     return samples

    def _sample_single_route(self, route: Dict, departure_time: datetime, context: str) -> Dict:
        """Sample a single route at a specific time"""
        result = self.gmaps.distance_matrix(
            origins=[route['origin']],
            destinations=[route['destination']],
            mode='driving', departure_time=departure_time, traffic_model='best_guess'
        )
        self.api_call_count += 1
        self.total_cost += self.cost_per_call
        if result['status'] == 'OK' and result['rows'][0]['elements'][0]['status'] == 'OK':
            elem = result['rows'][0]['elements'][0]
            duration = elem['duration']['value']
            duration_traffic = elem.get('duration_in_traffic', {}).get('value', duration)
            return {
                'route_id': route['id'], 'road_type': route['road_type'],
                'departure_time': departure_time.isoformat(), 'context': context,
                'duration_min': duration / 60, 'duration_traffic_min': duration_traffic / 60,
                'google_request': {'origin': route['origin'], 'destination': route['destination']},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        raise Exception(f"Google API Error: {result.get('status', 'Unknown')}")

    def _get_next_weekday(self, days_ahead: int) -> datetime:
        """Get next non-holiday weekday"""
        target_date = datetime.now() + timedelta(days=days_ahead)
        while target_date.weekday() >= 5 or target_date in self.holidays:
            target_date += timedelta(days=1)
        return target_date

    def _save_sampling_results(self, results: Dict):
        """Save sampling results to the cache"""
        samples_data = {
            'metadata': {
                'sampling_completed': datetime.now(timezone.utc).isoformat(),
                'total_samples': len(results.get('baseline_samples',{})) + len(results.get('traffic_samples',{})),
            },
            'baseline_samples': results.get('baseline_samples',{}),
            'traffic_samples': results.get('traffic_samples',{})
        }
        self.cache.update_with_metadata(
            'strategic_routes', 'strategic_routes/route_samples.json', samples_data,
            {'script': 'traffic_sampler.py', 'reason': 'strategic_sampling'}
        )