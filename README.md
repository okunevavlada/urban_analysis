# urban_analysis

urban_analysis is a Python library for the automated generation of raster databases of urban building and underlying surface parameters for climate and meteorological modeling. It provides a comprehensive pipeline for processing geospatial data from loading raw vector and raster datasets to generating a final multi-channel raster containing 12 key urban environment parameters.

The library implements advanced machine learning (CatBoost) for building height reconstruction, urban canyon geometry calculations (width, height, length, direction, Sky View Factor), and land cover classification.

## Output Data (12 Channels)

The library generates an integer multi-channel raster (BigTIFF, LZW compression) containing the following parameters:

| № | Parameter | Description |
|---|-----------|-------------|
| 1 | Building height | Reconstructed building height (in meters) |
| 2 | Height source | Flag: predicted by ML or taken from original data |
| 3 | Urban canyon width | Width of street space between buildings |
| 4 | Urban canyon height | Averaged height of surrounding buildings |
| 5 | Urban canyon length | Continuous space extent |
| 6 | Canyon direction | Azimuth (in degrees) |
| 7 | Sky View Factor (SVF) | Sky openness coefficient (0-1) |
| 8 | Roads | Impervious surface fraction |
| 9 | Industrial zones | Binary industrial area mask |
| 10 | Land cover | ESA WorldCover classification |
| 11 | Local Climate Zones (LCZ) | LCZ classification |
| 12 | Digital Elevation Model (DEM) | Digital elevation model |


## Key Features

- Data integration: Overture Maps (buildings, roads, land use), ESA WorldCover, WUDAPT, ArcticDEM, Copernicus DEM / FABDEM.
- Building height reconstruction: Uses gradient boosting (CatBoost) based on morphometric features, building type, and DSM/DEM data.
- Canyon calculation: Algorithms for computing width, height, length, direction, and SVF.
- Flexible architecture: Modular structure with YAML configuration support.
- Fault tolerance: Built-in state serialization (pickle) allows resuming calculations from the interrupted step.
- Big data support: Tile-based raster processing.


## Installation

### Prerequisites
- Python 3.9+

### Dependencies

The library relies on the following key packages:

- numpy, pandas - core computations
- rasterio - raster processing
- geopandas, shapely - vector processing
- catboost - machine learning
- scikit-learn - model quality assessment
- pyyaml - configuration file handling
- rasterspace - urban canyon parameter calculations (width, height, length, direction, SVF) (https://github.com/tsamsonov/osmlc2grid)

## Usage

### Run the full pipeline

```bash
python create_base.py --config config.yaml
```

## Command-Line Arguments
### Argument	Short	Description

| Argument | Short | Description |
|---|-----------|-------------|
| --config | -c |Path to configuration file |
| --only-step | -o | Run only a specific step |
| --range |-rng | Run range of steps (e.g., 3-7) |
| --step | -stp | Run multiple specific steps |
| --reset | -r | Start from specific step |
| --start-from | -s | Azimuth (in degrees) |
| --status | -st | Show current pipeline status|

## Funding
The urban_analysis library was developed in 2025-2026.
