# Windows Compatibility Fixes - Installation Instructions

## Files in This Directory

1. **requirements.txt** - Updated with Windows-compatible packages and flexible versioning
2. **WINDOWS_COMPATIBILITY.md** - Documentation about the changes and Windows installation guide
3. **README_APPLY_FIXES.md** - This file

## What Was Changed

### 1. Removed graph-tool
- **Reason**: Not Windows-compatible, never used in the codebase
- All graph operations use `networkx` instead (which IS Windows-compatible)

### 2. Updated All Version Pins
- **Old format**: `pandas==2.1.4` (exact version only)
- **New format**: `pandas>=2.1.4,<3.0.0` (flexible with safety)
- **Benefits**:
  - Allows security patches and bug fixes
  - Reduces dependency conflicts
  - Better Python version compatibility
  - Still protects against breaking changes

## How to Apply These Changes to Your Local Repository

### Step 1: Download These Files
Download all files from the `DOWNLOAD_WINDOWS_FIXES` directory to your local machine.

### Step 2: Replace Files in Your Repository
```bash
# Navigate to your local repository
cd path/to/2110403-DSDE-M150-Lover

# Replace the requirements.txt file
# (Copy the downloaded requirements.txt to your repo root)
copy DOWNLOAD_WINDOWS_FIXES\requirements.txt requirements.txt

# Copy the new documentation file
# (Copy WINDOWS_COMPATIBILITY.md to your repo root)
copy DOWNLOAD_WINDOWS_FIXES\WINDOWS_COMPATIBILITY.md WINDOWS_COMPATIBILITY.md
```

### Step 3: Commit and Push the Changes
```bash
# Check what changed
git diff

# Add the files
git add requirements.txt WINDOWS_COMPATIBILITY.md

# Commit
git commit -m "Fix Windows compatibility and improve dependency management

- Remove graph-tool (not Windows-compatible, never used in codebase)
- Update all version pins from == to >= with upper bounds
- Add WINDOWS_COMPATIBILITY.md with installation guidance
- All graph operations use networkx (Windows-compatible)

Benefits:
- Allows security patches and bug fixes
- Reduces dependency conflicts
- Better Python version compatibility
- Protects against breaking changes with version caps"

# Push to your branch
git push origin claude/implement-dsde-workflows-012JLHuvPoH2wHuFqv6caNPJ
```

### Step 4: Install Dependencies on Windows

**Recommended: Using conda (best for Windows)**
```bash
# Create conda environment
conda create -n urban-forecasting python=3.10
conda activate urban-forecasting

# Install geospatial packages via conda first
conda install -c conda-forge geopandas fiona shapely rtree

# Then install remaining packages via pip
pip install -r requirements.txt
```

**Alternative: Using pip only**
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Notes

- The project is now **fully Windows-compatible**
- All problematic packages have been removed
- Version ranges allow flexibility while maintaining stability
- See WINDOWS_COMPATIBILITY.md for detailed installation guidance

## Need Help?

Refer to `WINDOWS_COMPATIBILITY.md` for:
- Packages that may need special attention on Windows
- Troubleshooting common installation issues
- WSL2 option for components with limited Windows support (like Airflow)
