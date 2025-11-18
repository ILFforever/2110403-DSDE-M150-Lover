# System Architecture

## Overview
The Urban Issue Forecasting System is built with a modern data engineering and machine learning stack, following best practices for scalability, maintainability, and performance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  Traffy Fondue (700k+ records)  │  External Data Sources        │
│  - Citizen Complaints            │  - BMA Flood Monitoring       │
│  - Bangkok Metropolitan Area     │  - Traffic Incidents          │
│  - Aug 2021 - Jan 2025          │  - Construction Permits       │
│                                  │  - Public Events              │
└────────────┬────────────────────┴───────────────┬───────────────┘
             │                                    │
             │                                    │ Web Scraping
             │                                    │ (BeautifulSoup,
             │                                    │  Selenium)
             ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  - CSV File Loader                                              │
│  - API Connectors                                               │
│  - Scheduled Scrapers (Airflow)                                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PROCESSING LAYER (Apache Spark)                  │
├─────────────────────────────────────────────────────────────────┤
│  - Distributed Data Processing                                  │
│  - Data Cleaning & Validation                                   │
│  - Feature Engineering                                          │
│  - Deduplication & Normalization                                │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STORAGE LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Delta Lake   │  │ PostgreSQL   │  │ MongoDB            │   │
│  │ (Versioned)  │  │ (Analytics)  │  │ (External Data)    │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRANSFORMATION LAYER (dbt)                          │
├─────────────────────────────────────────────────────────────────┤
│  - Staging Models (Views)                                       │
│  - Core Fact Tables                                             │
│  - Dimension Tables                                             │
│  - Analytics Marts                                              │
│  - Data Quality Tests                                           │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──────────────────┬──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
┌──────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
│   ML MODELS      │ │  ANOMALY           │ │  GRAPH ANALYTICS    │
├──────────────────┤ │  DETECTION         │ ├─────────────────────┤
│ - LSTM Network   │ ├────────────────────┤ │ - NetworkX          │
│ - Prophet        │ │ - Isolation Forest │ │ - Community         │
│ - XGBoost        │ │ - Z-Score          │ │   Detection         │
│                  │ │ - DBSCAN           │ │ - Centrality        │
│ Forecast:        │ │                    │ │   Metrics           │
│ - 7-30 days      │ │ Detects:           │ │                     │
│ - By category    │ │ - Spikes           │ │ Analysis:           │
│ - By district    │ │ - Spatial          │ │ - Co-occurrence     │
│                  │ │ - Temporal         │ │ - Collaboration     │
└────────┬─────────┘ └──────────┬─────────┘ └──────────┬──────────┘
         │                      │                       │
         │                      │                       │
         └──────────────────────┴───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 VISUALIZATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────┐  ┌────────────────────────────┐    │
│  │  Streamlit Dashboard   │  │  Network Graphs            │    │
│  ├────────────────────────┤  ├────────────────────────────┤    │
│  │ - Geospatial Maps      │  │ - Plotly Interactive       │    │
│  │ - Time-slider          │  │ - Force-directed Layout    │    │
│  │ - Heat Maps            │  │ - Community Coloring       │    │
│  │ - Forecast Viz         │  │                            │    │
│  │ - Anomaly Alerts       │  │                            │    │
│  └────────────────────────┘  └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              ORCHESTRATION (Apache Airflow)                      │
├─────────────────────────────────────────────────────────────────┤
│  Daily DAG:                                                     │
│  1. Scrape External Data                                        │
│  2. Process with Spark                                          │
│  3. Data Quality Checks                                         │
│  4. dbt Transformations                                         │
│  5. Train/Update ML Models                                      │
│  6. Run Anomaly Detection                                       │
│  7. Generate Visualizations                                     │
│  8. Send Notifications                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion
- **Primary Source**: Traffy Fondue CSV (~800MB, 700k+ records)
- **External Sources**: Web scrapers collecting 1,000+ records
- **Format**: CSV, JSON, API responses
- **Frequency**: Daily automated collection

### 2. Apache Spark Processing
- **Purpose**: Distributed data processing
- **Operations**:
  - Deduplication by ticket_id
  - Geographic filtering (Bangkok only)
  - Coordinate parsing
  - Feature engineering
  - Temporal feature extraction
- **Output**: Parquet files, Delta Lake tables

### 3. Delta Lake Storage
- **Features**:
  - ACID transactions
  - Time travel (version control)
  - Schema evolution
  - Scalable metadata
- **Tables**:
  - traffy_complaints (main)
  - district_metrics (aggregated)
  - category_metrics (aggregated)

### 4. dbt Transformations
- **Layers**:
  - **Staging**: Raw data cleaning and validation
  - **Core**: Fact and dimension tables
  - **Marts**: Business-specific aggregations
- **Models**:
  - stg_complaints
  - fct_complaints
  - mart_district_metrics

### 5. Machine Learning Models

#### LSTM Forecasting
- **Architecture**: Bidirectional LSTM
- **Layers**: 128 → 64 → 32 units
- **Input**: 30-day historical window
- **Output**: 7-day forecast
- **Metrics**: MAE, RMSE, MAPE

#### Anomaly Detection
- **Methods**:
  - Isolation Forest (ensemble)
  - Statistical Z-score
  - DBSCAN (spatial)
  - Temporal spike detection
- **Output**: Anomaly scores, binary labels

### 6. Graph Analytics
- **Library**: NetworkX
- **Analyses**:
  - Complaint type co-occurrence
  - Organization collaboration networks
  - Community detection (Louvain)
  - Centrality metrics

### 7. Visualization Dashboard
- **Framework**: Streamlit
- **Features**:
  - Interactive maps (Folium)
  - Time-series plots (Plotly)
  - Filters and controls
  - Real-time updates
- **URL**: http://localhost:8501

### 8. Workflow Orchestration
- **Tool**: Apache Airflow
- **Schedule**: Daily at midnight
- **DAG Tasks**: 9 sequential/parallel tasks
- **Monitoring**: Web UI at http://localhost:8080

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Apache Airflow | Workflow scheduling |
| Processing | Apache Spark | Distributed computing |
| Storage | Delta Lake, PostgreSQL, MongoDB | Data persistence |
| Transformation | dbt | SQL modeling |
| ML Framework | TensorFlow/Keras, Prophet | Forecasting |
| ML Libraries | Scikit-learn, XGBoost | Anomaly detection |
| Graph Analysis | NetworkX | Network analysis |
| Visualization | Streamlit, Plotly, Folium | Dashboards |
| Web Scraping | BeautifulSoup, Selenium | Data collection |
| Containerization | Docker Compose | Deployment |

## Data Flow

1. **Daily 00:00**: Airflow triggers DAG
2. **00:05**: Web scrapers collect external data
3. **00:15**: Spark processes raw data → Delta Lake
4. **00:45**: Data quality checks run
5. **01:00**: dbt transforms data into marts
6. **01:30**: ML models train/update (parallel)
7. **02:00**: Anomaly detection runs (parallel)
8. **02:30**: Visualizations generate
9. **03:00**: Dashboard refreshes
10. **03:15**: Notifications sent

## Scalability Considerations

- **Horizontal Scaling**: Spark cluster can add workers
- **Vertical Scaling**: Increase memory/CPU per node
- **Caching**: Redis for frequently accessed data
- **Partitioning**: Data partitioned by year/month
- **Incremental Processing**: Only new data processed daily

## Security

- Environment variables for credentials
- No hardcoded passwords
- Database access controls
- API rate limiting
- Input validation and sanitization

## Monitoring & Logging

- Airflow task logs
- Spark application logs
- Model training metrics
- Data quality test results
- Dashboard access logs

## Future Enhancements

1. Real-time streaming (Apache Kafka)
2. Advanced NLP for complaint text
3. Mobile application integration
4. Multi-city expansion
5. AutoML for model optimization
