# Quick Start Guide

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 10GB free
- **CPU**: Multi-core processor recommended

### Software Requirements
- Python 3.8 or higher
- Docker & Docker Compose
- Git

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/ILFforever/2110403-DSDE-M150-Lover.git
cd 2110403-DSDE-M150-Lover
```

### 2. Download Data
Download the cleaned dataset (~800MB) from:
```
https://github.com/ParinthornThammarux/2110403-DSDE-M150-Lover/releases/tag/v1.0
```

Place the CSV file in the project root directory as `bangkok_traffy.csv`.

### 3. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

### 4. Install Python Dependencies

#### Option A: Using pip
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Docker (Recommended)
```bash
# All dependencies are included in containers
docker-compose up -d
```

## Running the System

### Quick Demo (Standalone)

#### 1. Run Data Processing
```bash
python data_engineering/spark/process_traffy_data.py
```

#### 2. Scrape External Data
```bash
python web_scraping/bma_flood_scraper.py
```

#### 3. Train ML Models
```bash
# LSTM Forecasting
python ml_models/forecasting/train_lstm_model.py

# Anomaly Detection
python ml_models/anomaly_detection/detect_anomalies.py
```

#### 4. Generate Network Graphs
```bash
python visualization/graphs/complaint_network.py
```

#### 5. Launch Dashboard
```bash
streamlit run visualization/dashboard/app.py
```

Access at: http://localhost:8501

### Full Production Setup (Docker)

#### 1. Start All Services
```bash
docker-compose up -d
```

This will start:
- PostgreSQL (port 5432)
- MongoDB (port 27017)
- Redis (port 6379)
- Airflow (port 8080)
- Spark Master (port 8082)
- Spark Worker
- Streamlit Dashboard (port 8501)
- Jupyter Notebook (port 8888)

#### 2. Access Airflow
1. Open http://localhost:8080
2. Login with:
   - Username: `admin`
   - Password: `admin`
3. Enable the `traffy_urban_forecasting_pipeline` DAG
4. Trigger manually or wait for scheduled run

#### 3. Access Dashboard
Open http://localhost:8501 to view the interactive dashboard

#### 4. Access Jupyter (Optional)
Open http://localhost:8888 for development notebooks

#### 5. Monitor Spark Jobs
Open http://localhost:8082 to view Spark cluster status

## Verify Installation

### Check Services Status
```bash
docker-compose ps
```

All services should show "Up" status.

### Check Logs
```bash
# View all logs
docker-compose logs

# View specific service
docker-compose logs airflow
docker-compose logs dashboard
```

### Test Database Connection
```bash
# PostgreSQL
docker exec -it traffy_postgres psql -U postgres -d traffy_db

# MongoDB
docker exec -it traffy_mongodb mongosh -u admin -p admin
```

## Common Commands

### Docker Management
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a service
docker-compose restart dashboard

# View logs
docker-compose logs -f airflow

# Remove all containers and volumes
docker-compose down -v
```

### Python Scripts
```bash
# Activate virtual environment
source venv/bin/activate

# Run specific component
python data_engineering/spark/process_traffy_data.py
python ml_models/forecasting/train_lstm_model.py
python web_scraping/bma_flood_scraper.py

# Run tests
pytest tests/
```

### Airflow
```bash
# Trigger DAG manually
docker exec traffy_airflow airflow dags trigger traffy_urban_forecasting_pipeline

# List DAGs
docker exec traffy_airflow airflow dags list

# View task status
docker exec traffy_airflow airflow tasks list traffy_urban_forecasting_pipeline
```

## Project Structure Overview

```
├── data_engineering/      # Spark, dbt, Delta Lake
├── ml_models/            # Forecasting & anomaly detection
├── web_scraping/         # External data collection
├── airflow/              # DAGs and orchestration
├── visualization/        # Dashboard and graphs
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
├── docker-compose.yml    # Container orchestration
└── requirements.txt      # Python dependencies
```

## Next Steps

1. **Explore the Dashboard**: Navigate through different visualizations
2. **Review Airflow DAG**: Understand the workflow
3. **Check Model Outputs**: View predictions and anomaly reports
4. **Customize Configuration**: Adjust `config/config.yaml` for your needs
5. **Run Tests**: Ensure everything works correctly

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

### Docker Out of Memory
```bash
# Increase Docker memory in Docker Desktop settings
# Recommended: 6GB+
```

### Permission Denied
```bash
# Fix file permissions
chmod +x scripts/*.sh
```

### Data Not Loading
1. Verify CSV file is in correct location
2. Check file path in config
3. Ensure sufficient disk space

### Airflow DAG Not Showing
1. Check `airflow/dags/` directory
2. Verify no Python syntax errors
3. Refresh Airflow UI

## Support

For issues or questions:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for system details
2. Review [README.md](../README.md) for project overview
3. Open an issue on GitHub
4. Contact team members

## Development Mode

For active development:

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run in debug mode
export DEBUG=1
python -m pdb visualization/dashboard/app.py

# Watch for file changes
streamlit run visualization/dashboard/app.py --server.runOnSave true
```

---

**Happy Forecasting! 🏙️📊**
