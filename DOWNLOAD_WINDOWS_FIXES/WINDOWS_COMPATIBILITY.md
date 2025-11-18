# Windows Compatibility Notes

## Changes Made for Windows Compatibility

### 1. Removed graph-tool Package
**Issue**: `graph-tool==2.55` is not compatible with Windows due to C++ dependencies and lack of Windows builds.

**Solution**: Removed from requirements.txt as it was never used in the codebase.

**Evidence**:
- The project uses `networkx` for all graph operations (see `visualization/graphs/complaint_network.py:8`)
- No imports of `graph_tool` found in any Python files
- `networkx` is fully Windows-compatible and provides all needed functionality

### 2. Updated Version Pinning Strategy

**Previous**: All packages used exact version pinning with `==`
```
pandas==2.1.4
numpy==1.24.3
```

**New**: Flexible version ranges with upper bounds
```
pandas>=2.1.4,<3.0.0
numpy>=1.24.3,<2.0.0
```

**Benefits**:
- **Security patches**: Allows automatic security updates
- **Bug fixes**: Gets minor version improvements
- **Dependency resolution**: Reduces conflicts between packages
- **Python version compatibility**: Works better with newer Python versions
- **Breaking changes protection**: Upper bounds prevent major version changes

### 3. Version Range Strategy

The format `>=X.Y.Z,<(X+1).0.0` means:
- **Minimum version**: X.Y.Z (tested/known to work)
- **Maximum version**: Before next major version
- **Allows**: Patch updates (X.Y.Z+1) and minor updates (X.(Y+1).0)
- **Blocks**: Major version updates (X+1.0.0) that might break compatibility

## Windows-Specific Installation Notes

### Packages That May Need Special Attention on Windows

1. **GeoPandas & Fiona**
   - May require GDAL installation
   - Consider using conda: `conda install -c conda-forge geopandas`

2. **PyTorch**
   - Choose CPU or CUDA version from https://pytorch.org/
   - Update requirements if you need GPU support

3. **Scrapy**
   - Requires Visual C++ compiler
   - Install Visual Studio Build Tools if needed

4. **Apache Airflow**
   - Has limited Windows support
   - Consider using WSL2 (Windows Subsystem for Linux) for Airflow workflows

## Recommended Installation on Windows

### Option 1: Using conda (Recommended)
```bash
# Create conda environment
conda create -n urban-forecasting python=3.10
conda activate urban-forecasting

# Install geospatial packages via conda
conda install -c conda-forge geopandas fiona shapely rtree

# Install remaining packages via pip
pip install -r requirements.txt
```

### Option 2: Using pip with virtual environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Option 3: WSL2 (for full compatibility)
For components like Apache Airflow that have limited Windows support:
1. Install WSL2
2. Install Ubuntu from Microsoft Store
3. Run the project in the Linux environment

## Testing Installation

```bash
# Test core imports
python -c "import pandas, numpy, networkx, sklearn; print('Core packages OK')"

# Test ML packages
python -c "import tensorflow, torch, xgboost; print('ML packages OK')"

# Test visualization
python -c "import plotly, streamlit, dash; print('Viz packages OK')"

# Test geospatial (may fail without GDAL)
python -c "import geopandas, fiona; print('Geospatial packages OK')"
```

## Notes

- The project now uses **only Windows-compatible packages**
- All graph analysis uses `networkx` (not graph-tool)
- Version ranges allow flexibility while maintaining stability
- Consider using conda for geospatial packages on Windows
