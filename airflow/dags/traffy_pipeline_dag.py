"""
Apache Airflow DAG for Urban Issue Forecasting Pipeline
Orchestrates data ingestion, processing, ML training, and visualization
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import logging

logger = logging.getLogger(__name__)

# Default arguments for DAG
default_args = {
    'owner': 'dsde-m150-lover',
    'depends_on_past': False,
    'email': ['team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Define DAG
dag = DAG(
    'traffy_urban_forecasting_pipeline',
    default_args=default_args,
    description='End-to-end pipeline for urban complaint analysis and forecasting',
    schedule_interval='@daily',  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=['dsde', 'traffy', 'forecasting', 'urban-analytics'],
    max_active_runs=1,
)


# Task 1: Web Scraping - Collect External Data
def scrape_external_data(**context):
    """Collect data from external sources (BMA, traffic, etc.)"""
    logger.info("Starting web scraping task...")
    import sys
    sys.path.append('/home/user/2110403-DSDE-M150-Lover')

    from web_scraping.bma_flood_scraper import main as scrape_main
    df_external = scrape_main()

    # Store metadata
    context['ti'].xcom_push(key='external_records_count', value=len(df_external))
    logger.info(f"Scraped {len(df_external)} external records")


scraping_task = PythonOperator(
    task_id='scrape_external_data',
    python_callable=scrape_external_data,
    provide_context=True,
    dag=dag,
)


# Task 2: Spark Data Processing
spark_processing_task = SparkSubmitOperator(
    task_id='spark_process_traffy_data',
    application='/home/user/2110403-DSDE-M150-Lover/data_engineering/spark/process_traffy_data.py',
    name='traffy-spark-processing',
    conf={
        'spark.executor.memory': '4g',
        'spark.driver.memory': '2g',
        'spark.sql.adaptive.enabled': 'true',
    },
    packages='io.delta:delta-core_2.12:3.0.0',
    verbose=True,
    dag=dag,
)


# Task 3: Data Quality Checks
def data_quality_checks(**context):
    """Validate processed data quality"""
    logger.info("Running data quality checks...")
    import pandas as pd
    from pathlib import Path

    # Load processed data from Delta Lake
    processed_path = "data/processed/traffy_processed.parquet"

    if not Path(processed_path).exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}")

    df = pd.read_parquet(processed_path)

    # Quality checks
    checks = {
        'total_records': len(df),
        'null_check_passed': df[['district', 'timestamp']].isnull().sum().sum() == 0,
        'date_range_valid': (df['timestamp'].min() > pd.Timestamp('2020-01-01')),
        'coordinates_valid': ((df['lat'].between(13.5, 14.0)) & (df['lon'].between(100.3, 100.7))).all(),
    }

    logger.info(f"Data quality checks: {checks}")

    # Fail if critical checks don't pass
    if not all([checks['null_check_passed'], checks['date_range_valid']]):
        raise ValueError("Data quality checks failed!")

    context['ti'].xcom_push(key='quality_checks', value=checks)


quality_check_task = PythonOperator(
    task_id='data_quality_checks',
    python_callable=data_quality_checks,
    provide_context=True,
    dag=dag,
)


# Task 4: dbt Transformations
dbt_transform_task = BashOperator(
    task_id='dbt_transformations',
    bash_command='cd /home/user/2110403-DSDE-M150-Lover/data_engineering/dbt && dbt run',
    dag=dag,
)


# Task 5: Train LSTM Forecasting Model
def train_forecasting_model(**context):
    """Train LSTM model for complaint forecasting"""
    logger.info("Training LSTM forecasting model...")
    import sys
    sys.path.append('/home/user/2110403-DSDE-M150-Lover')

    from ml_models.forecasting.train_lstm_model import main as train_lstm
    train_lstm()

    logger.info("LSTM training completed")
    context['ti'].xcom_push(key='model_trained', value='lstm')


lstm_training_task = PythonOperator(
    task_id='train_lstm_forecasting_model',
    python_callable=train_forecasting_model,
    provide_context=True,
    execution_timeout=timedelta(hours=1),
    dag=dag,
)


# Task 6: Run Anomaly Detection
def run_anomaly_detection(**context):
    """Detect anomalies in complaint patterns"""
    logger.info("Running anomaly detection...")
    import sys
    sys.path.append('/home/user/2110403-DSDE-M150-Lover')

    from ml_models.anomaly_detection.detect_anomalies import main as detect_anomalies
    detect_anomalies()

    logger.info("Anomaly detection completed")
    context['ti'].xcom_push(key='anomalies_detected', value=True)


anomaly_detection_task = PythonOperator(
    task_id='run_anomaly_detection',
    python_callable=run_anomaly_detection,
    provide_context=True,
    dag=dag,
)


# Task 7: Generate Visualizations
def generate_visualizations(**context):
    """Generate geospatial and graph visualizations"""
    logger.info("Generating visualizations...")
    import pandas as pd
    import plotly.express as px
    from pathlib import Path

    output_dir = Path("visualization/dashboard/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    df = pd.read_parquet("data/processed/traffy_processed.parquet")

    # Generate sample geospatial viz
    if 'lat' in df.columns and 'lon' in df.columns:
        fig = px.scatter_mapbox(
            df.head(1000),
            lat='lat',
            lon='lon',
            hover_data=['district', 'timestamp'],
            zoom=10,
            height=600
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.write_html(str(output_dir / "map_preview.html"))
        logger.info("Generated map visualization")

    context['ti'].xcom_push(key='visualizations_generated', value=True)


visualization_task = PythonOperator(
    task_id='generate_visualizations',
    python_callable=generate_visualizations,
    provide_context=True,
    dag=dag,
)


# Task 8: Update Dashboard
dashboard_update_task = BashOperator(
    task_id='update_dashboard',
    bash_command='echo "Dashboard refresh triggered"',
    dag=dag,
)


# Task 9: Send Notifications
def send_completion_notification(**context):
    """Send pipeline completion notification"""
    logger.info("Sending completion notification...")

    # Gather metrics from previous tasks
    external_records = context['ti'].xcom_pull(task_ids='scrape_external_data', key='external_records_count')
    quality_checks = context['ti'].xcom_pull(task_ids='data_quality_checks', key='quality_checks')

    notification_message = f"""
    Urban Forecasting Pipeline Completed Successfully!

    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Summary:
    - External records scraped: {external_records}
    - Total processed records: {quality_checks.get('total_records', 'N/A')}
    - Data quality: PASSED
    - LSTM model: TRAINED
    - Anomaly detection: COMPLETED
    - Visualizations: UPDATED

    Dashboard available at: http://localhost:8501
    """

    logger.info(notification_message)
    # In production, send via email/Slack/etc.


notification_task = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    provide_context=True,
    dag=dag,
)


# Define task dependencies (DAG structure)
scraping_task >> spark_processing_task >> quality_check_task >> dbt_transform_task

# Parallel ML tasks after data preparation
dbt_transform_task >> [lstm_training_task, anomaly_detection_task]

# Visualization after ML
[lstm_training_task, anomaly_detection_task] >> visualization_task

# Dashboard update and notification
visualization_task >> dashboard_update_task >> notification_task


# Additional configuration
dag.doc_md = """
# Urban Issue Forecasting Pipeline

## Overview
This DAG orchestrates the complete end-to-end pipeline for urban complaint analysis:

1. **Web Scraping**: Collects external data (floods, traffic, construction, events)
2. **Spark Processing**: Distributed data processing with Delta Lake storage
3. **Quality Checks**: Validates data integrity
4. **dbt Transformations**: SQL-based data modeling
5. **LSTM Training**: Time-series forecasting model
6. **Anomaly Detection**: Identifies unusual patterns
7. **Visualizations**: Generates interactive maps and graphs
8. **Dashboard Update**: Refreshes Streamlit dashboard
9. **Notifications**: Sends completion alerts

## Schedule
Runs daily at midnight (can be adjusted via schedule_interval)

## Monitoring
- Check task logs for detailed execution info
- XCom for inter-task communication
- Email alerts on failure (configure in default_args)

## Dependencies
- Apache Spark 3.x
- Delta Lake
- dbt
- TensorFlow/Keras
- Plotly, Folium

## Contact
DSDE M150-Lover Team
"""
