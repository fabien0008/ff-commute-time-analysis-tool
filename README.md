# Advanced Commute Time Analysis Tool

A sophisticated tool for analyzing optimal residential locations based on commute times to multiple workplaces. Originally designed for finding the perfect home location on the French Riviera, but configurable for any geographic region.

## ✨ Features

- **🗺️ Geographic Land Filter**: Uses point-in-polygon algorithm to ensure analyzed points are on land
- **🎨 Rich Interactive Visualization**: Color-coded maps with compromise scores and detailed legends
- **🔄 Multi-Corridor Seeding**: Scans multiple parallel corridors for diverse solution discovery
- **📅 Multi-Day Comparison**: Runs analysis across different peak-hour days for comprehensive results
- **💰 Smart Caching**: Reduces Google Maps API costs through intelligent result caching
- **⚙️ Fully Configurable**: JSON-based configuration for any geographic region and commute scenario

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.7+
- Google Maps API key with Distance Matrix API enabled
- Required Python packages (see `requirements.txt`)

### 2. Installation

```bash
# Clone or download the repository
git clone https://github.com/your-username/commute-time-analysis-tool.git
cd commute-time-analysis-tool

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. **Get Google Maps API Key**:
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Distance Matrix API and Geocoding API
   - Create an API key

2. **Configure the tool**:
   ```bash
   # Edit the configuration file
   nano config.json
   ```
   
   Update the following essential fields:
   ```json
   {
     "api": {
       "google_maps_api_key": "YOUR_ACTUAL_API_KEY_HERE"
     },
     "workplaces": {
       "person_a": {
         "address": "Your first workplace address",
         "transport_mode": "bicycling",
         "max_commute_minutes": 45
       },
       "person_b": {
         "address": "Your second workplace address", 
         "transport_mode": "driving",
         "max_commute_minutes": 35
       }
     }
   }
   ```

### 4. Run Analysis

```bash
# Basic run
python3 commute_analyzer.py

# Or use the quick start script
chmod +x runme.sh
./runme.sh

# Advanced options
python3 commute_analyzer.py --scenario person_b_school_drop --clear-cache
```

## 📊 Output

The tool generates:

- **📍 Interactive HTML Map**: Visual analysis with color-coded compromise scores
- **📈 CSV Data File**: Viable locations with detailed metrics
- **💾 Cached Results**: Saved API responses for future runs

Example output:
```
🎉 COMPARATIVE ANALYSIS COMPLETED 🎉
📊 GLOBAL SUMMARY:
   • ⏱️ Duration: 12.3 min | 💰 Cost: ~€4.50 | 📞 API calls: 1,247
🔍 RESULTS BY SIMULATION DATE:
   • 2025-10-01: 23 viable points found out of 456 analyzed
📁 RESULT FILES:
   • 🗺️ Interactive map: ./commute_analysis_1642358190.html
   • 📈 CSV data (viable points): ./viable_locations.csv
```

## ⚙️ Configuration Guide

### Geographic Area Setup

For regions outside the French Riviera, update the `geographic_area` section:

```json
{
  "geographic_area": {
    "name": "Your Region Name",
    "center_lat": 40.7128,
    "center_lng": -74.0060,
    "zoom_level": 11,
    "land_polygon": [
      [lng1, lat1], [lng2, lat2], [lng3, lat3], ...
    ],
    "area_names": {
      "Downtown": [40.7128, -74.0060],
      "Brooklyn": [40.6782, -73.9442]
    }
  }
}
```

### Transport Modes

Supported Google Maps transport modes:
- `"driving"`: Car with real-time traffic
- `"bicycling"`: Bicycle routes
- `"walking"`: Pedestrian paths
- `"transit"`: Public transportation

### Advanced Parameters

```json
{
  "analysis": {
    "grid_resolution_km": 2.0,        // Analysis grid density
    "search_radius_km": 12.0,         // Search area radius
    "corridor_points": 30,            // Points per corridor scan
    "corridor_offsets_km": [0, -2.5, 2.5]  // Parallel corridor offsets
  },
  "clustering": {
    "dbscan_eps_km": 3.0,            // Clustering distance threshold
    "dbscan_min_samples": 2,         // Minimum cluster size
    "max_zones_to_analyze": 2        // Top zones for detailed analysis
  }
}
```

## 💰 Cost Management

Google Maps API pricing (as of 2024):
- Distance Matrix: ~$0.005 per element
- Geocoding: ~$0.005 per request

Cost control features:
- **Smart Caching**: Saves previous API responses
- **Budget Limits**: Set maximum spending per analysis
- **Progressive Analysis**: Focuses on promising areas first

Typical costs:
- Small analysis (~500 points): €2-4
- Large analysis (~2000 points): €8-15

## 🔧 Advanced Usage

### Command Line Options

```bash
# Use custom configuration file
python3 commute_analyzer.py --config my_config.json

# Select different family scenario
python3 commute_analyzer.py --scenario person_b_school_drop

# Clear cache before running
python3 commute_analyzer.py --clear-cache
```

### Family Scenarios

Define different departure time scenarios:

```json
{
  "family_scenarios": {
    "early_start": {
      "description": "Early departure for both",
      "person_a_departure_hour": 7,
      "person_a_departure_minute": 0,
      "person_b_departure_hour": 7,
      "person_b_departure_minute": 15
    }
  }
}
```

### Customizing Compromise Scoring

Adjust penalty weights for constraint violations:

```json
{
  "compromise_scoring": {
    "person_a_penalty_weight": 1.5,  // Weight for person A time overruns
    "person_b_penalty_weight": 2.0,  // Weight for person B time overruns
    "color_scheme": {
      "perfect": "#28a745",   // Green for perfect solutions
      "good": "#fdbe41",      // Yellow for good compromises
      "medium": "#ff7700",    // Orange for medium compromises
      "poor": "#dc3545"       // Red for poor compromises
    }
  }
}
```

## 🗂️ Project Structure

```
commute-time-analysis-tool/
├── 📄 commute_analyzer.py     # Main analysis script
├── ⚙️ config.json           # Configuration file
├── 📋 requirements.txt      # Python dependencies
├── 🚀 runme.sh             # Quick start script
├── 📊 results/             # Analysis outputs
│   ├── *.html             # Interactive maps
│   ├── *.csv              # Data exports
│   └── *.json             # Raw results
└── 📚 examples/            # Alternative implementations
    ├── pareto_optimization/ # Pareto front analysis
    └── development_versions/ # Original versions
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature X"`
5. Push and create a pull request

## 🐛 Troubleshooting

### Common Issues

**❌ "Google Maps API key not configured"**
- Solution: Edit `config.json` and set your API key

**❌ "Missing dependencies"**
- Solution: Run `pip install -r requirements.txt`

**❌ API quota exceeded**
- Solution: Check Google Cloud Console quotas and billing

**❌ No viable locations found**
- Solution: Increase `max_commute_minutes` in config or expand `search_radius_km`

### Performance Tips

- Use caching to avoid repeated API calls
- Start with larger `grid_resolution_km` (3-5km) for initial exploration
- Reduce `corridor_points` if budget is limited
- Clear cache if addresses change

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- Google Maps Platform for routing data
- OpenStreetMap contributors for base maps
- scikit-learn for clustering algorithms
- Folium for interactive visualizations

---

**Happy house hunting! 🏡**