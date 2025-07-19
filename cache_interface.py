#!/usr/bin/env python3
"""
Cache Interface - Easy access to traffic intelligence cache
Provides high-level interface for cache access across scripts
"""

import logging
import time
from typing import Dict, Optional, Tuple
from traffic_cache import TrafficCache
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class CacheInterface:
    """High-level interface for cache access across scripts"""

    def __init__(self, cache_dir: str = ".traffic_intelligence_cache"):
        self.cache = TrafficCache(cache_dir)
        self._zones_cache = None
        self._multipliers_cache = None
        self._last_cache_time = 0

    def _refresh_cache_if_needed(self):
        """Refresh internal cache if data is stale"""
        current_time = time.time()
        if current_time - self._last_cache_time > 300:  # Refresh every 5 minutes
            self._zones_cache = None
            self._multipliers_cache = None
            self._last_cache_time = current_time

    def get_traffic_multiplier(self, road_type: str, time_window: str) -> float:
        """Get traffic multiplier with fallback logic"""
        self._refresh_cache_if_needed()
        try:
            if self._multipliers_cache is None:
                self._multipliers_cache = self.cache.load_json('calibration/traffic_multipliers.json')
            
            multipliers = self._multipliers_cache
            
            if road_type in multipliers.get('road_type_multipliers', {}):
                road_data = multipliers['road_type_multipliers'][road_type]
                if time_window in road_data:
                    return road_data[time_window].get('multiplier', 1.0)

            fallback_multipliers = {
                'morning_rush': 1.5, 'evening_rush': 1.4,
                'off_peak': 1.1, 'weekend': 1.0, 'night': 1.0
            }
            return fallback_multipliers.get(time_window, 1.0)
        except Exception as e:
            logger.debug(f"Could not get traffic multiplier for {road_type}/{time_window}: {e}")
            return 1.0

    def get_baseline_correction(self, road_type: str) -> float:
        """Get baseline correction factor for OSRM vs Google differences"""
        self._refresh_cache_if_needed()
        try:
            if self._multipliers_cache is None:
                self._multipliers_cache = self.cache.load_json('calibration/traffic_multipliers.json')
            
            corrections = self._multipliers_cache.get('baseline_corrections', {})
            
            if road_type in corrections:
                return corrections[road_type].get('correction_factor', 1.0)
            
            return 1.0
        except Exception as e:
            logger.debug(f"Could not get baseline correction for {road_type}: {e}")
            return 1.0

    def get_time_window(self, departure_time) -> str:
        """Convert departure time to traffic time window"""
        try:
            hour = departure_time.hour
            weekday = departure_time.weekday()
            
            if weekday >= 5: return 'weekend'
            if 7 <= hour < 10: return 'morning_rush'
            if 17 <= hour < 20: return 'evening_rush'
            if 22 <= hour or hour < 6: return 'night'
            return 'off_peak'
        except Exception:
            return 'off_peak'