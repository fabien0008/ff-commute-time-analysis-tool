#!/usr/bin/env python3
"""
Smart Traffic Intelligence Cache System
Handles reflective caching with metadata, versioning, and validation
"""

import json
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TrafficCache:
    """Smart cache manager with reflective metadata and versioning"""

    CACHE_VERSION = "1.2.0"

    def __init__(self, cache_dir: str = ".traffic_intelligence_cache"):
        self.cache_dir = Path(cache_dir)
        self.setup_cache_structure()
        self.metadata = self.load_metadata()

    def setup_cache_structure(self):
        """Create cache directory structure"""
        subdirs = [
            'metadata', 'strategic_routes', 'zones', 'calibration',
            'osrm_profiles', 'api_usage', 'validation'
        ]

        for subdir in subdirs:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

        if not (self.cache_dir / 'metadata' / 'cache_info.json').exists():
            self.initialize_cache_metadata()

    def initialize_cache_metadata(self):
        """Initialize cache metadata on first run"""
        cache_info = {
            "cache_info": {
                "version": self.CACHE_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "region": "French Riviera",
                "update_strategy": {
                    "strategic_routes": "weekly",
                    "zone_patterns": "monthly",
                    "baseline_validation": "quarterly"
                }
            }
        }
        self.save_json('metadata/cache_info.json', cache_info)

    def load_metadata(self) -> Dict:
        """Load cache metadata"""
        try:
            return self.load_json('metadata/cache_info.json')
        except FileNotFoundError:
            self.initialize_cache_metadata()
            return self.load_json('metadata/cache_info.json')

    def save_json(self, relative_path: str, data: Dict):
        """Save data to JSON file with error handling"""
        file_path = self.cache_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save {relative_path}: {e}")
            raise

    def load_json(self, relative_path: str) -> Dict:
        """Load data from JSON file with error handling"""
        file_path = self.cache_dir / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {relative_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {relative_path}: {e}")
            raise

    def update_with_metadata(self, cache_type: str, relative_path: str, data: Dict, update_info: Dict):
        """Update cache with comprehensive metadata"""
        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata'].update({
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'updated_by': update_info.get('script', 'unknown'),
        })
        self.save_json(relative_path, data)

    def should_update(self, cache_type: str) -> bool:
        """Check if cache needs updating based on age"""
        update_intervals = {
            'strategic_routes': 7 * 24 * 60 * 60,
            'traffic_multipliers': 7 * 24 * 60 * 60
        }
        try:
            last_update = self.get_last_update(cache_type)
            interval = update_intervals.get(cache_type, 7 * 24 * 60 * 60)
            return (time.time() - last_update) > interval
        except:
            return True

    def get_last_update(self, cache_type: str) -> float:
        """Get timestamp of last update for cache type"""
        try:
            # Simplified: check the file's modification time
            # A more robust system would use the metadata log
            if cache_type == "strategic_routes":
                path = self.cache_dir / 'strategic_routes/route_samples.json'
            elif cache_type == "traffic_multipliers":
                path = self.cache_dir / 'calibration/traffic_multipliers.json'
            else:
                return 0
            
            if path.exists():
                return path.stat().st_mtime
            return 0
        except:
            return 0