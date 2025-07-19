#!/usr/bin/env python3
"""
OSRM Profile Generator - Creates traffic-aware OSRM profiles
"""
import logging
from datetime import datetime, timezone
from typing import Dict
from traffic_cache import TrafficCache
from pathlib import Path

logger = logging.getLogger(__name__)

class OSRMProfileGenerator:
    """Generates traffic-aware OSRM routing profiles"""

    def __init__(self, cache_dir: str = ".traffic_intelligence_cache"):
        self.cache = TrafficCache(cache_dir)
        # NEW: Save generated profiles to the permanent 'profiles' directory
        self.profile_dir = Path('profiles')
        self.profile_dir.mkdir(exist_ok=True)

    def _calculate_traffic_speeds(self, multipliers: Dict, time_window: str) -> Dict[str, int]:
        """Calculates effective speeds for each road type in km/h"""
        base_speeds = {'highway': 110, 'primary': 70, 'secondary': 60, 'residential': 40}
        effective_speeds = {}
        road_multipliers = multipliers.get('road_type_multipliers', {})
        for road_type, base_speed in base_speeds.items():
            multiplier = road_multipliers.get(road_type, {}).get(time_window, {}).get('multiplier', 1.0)
            final_speed = base_speed / multiplier
            effective_speeds[road_type] = int(round(max(10, min(final_speed, 130))))
        return effective_speeds

    def _generate_profile_content(self, multipliers: Dict, time_window: str) -> str:
        """Generates the content for a single Lua profile with correct edge creation."""
        speeds = self._calculate_traffic_speeds(multipliers, time_window)
        
        return f"""
-- OSRM profile for: {time_window}
-- Generated at {datetime.now(timezone.utc).isoformat()}
api_version = 4

function setup()
    return {{
        speeds = {{
          highway = {speeds.get('highway', 80)},
          primary = {speeds.get('primary', 60)},
          secondary = {speeds.get('secondary', 50)},
          tertiary = {speeds.get('residential', 30)},
          residential = {speeds.get('residential', 25)},
          motorway = {speeds.get('highway', 90)},
          trunk = {speeds.get('highway', 85)},
          unclassified = {speeds.get('residential', 25)}
        }}
    }}
end

function process_way(profile, way, result)
  local highway = way:get_value_by_key('highway')
  if highway then
    local speed = profile.speeds[highway]
    if speed then
      local maxspeed = way:get_value_by_key('maxspeed')
      if maxspeed then
        local maxspeed_numeric = tonumber(string.match(maxspeed, '%d+'))
        if maxspeed_numeric then
          speed = math.min(speed, maxspeed_numeric)
        end
      end
      
      -- This is the corrected block
      result.forward_speed = speed
      result.backward_speed = speed
      result.forward_mode = 1 -- Set forward as drivable by default
      result.backward_mode = 1 -- Set backward as drivable by default
    end

    local oneway = way:get_value_by_key('oneway')
    if oneway == 'yes' or oneway == '1' or oneway == 'true' then
      result.backward_mode = 0 -- Make backward inaccessible
    elseif oneway == '-1' then
      result.forward_mode = 0 -- Make forward inaccessible
    end
  end
end

return {{
    setup = setup,
    process_way = process_way
}}
"""

    def generate_traffic_profiles(self) -> Dict:
        """Generate all traffic-aware OSRM profiles"""
        logger.info("⚙️ Generating traffic-aware OSRM profiles...")
        try:
            multipliers = self.cache.load_json('calibration/traffic_multipliers.json')
            time_windows = ['morning_rush', 'evening_rush', 'off_peak', 'weekend']
            for time_window in time_windows:
                profile_content = self._generate_profile_content(multipliers, time_window)
                profile_path = self.profile_dir / f"car_traffic_{time_window}.lua"
                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(profile_content)
            logger.info(f"✅ Generated {len(time_windows)} OSRM profiles in {self.profile_dir.resolve()}")
            return {'status': 'success'}
        except Exception as e:
            logger.error(f"❌ Profile generation failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}