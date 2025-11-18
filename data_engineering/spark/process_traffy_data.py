"""
Apache Spark Data Processing Pipeline for Traffy Fondue Dataset
Processes 100k+ complaint records with distributed computing
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, split, regexp_replace,
    datediff, year, month, dayofweek, hour,
    when, lit, concat_ws, explode, trim,
    count, avg, sum as spark_sum, max as spark_max, min as spark_min
)
from pyspark.sql.types import DoubleType, IntegerType, StringType
from delta import configure_spark_with_delta_pip
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TraffyDataProcessor:
    """Spark-based data processor for Traffy Fondue complaints"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_spark_session(self) -> SparkSession:
        """Create Spark session with Delta Lake support"""
        builder = (
            SparkSession.builder
            .appName(self.config['spark']['app_name'])
            .master(self.config['spark']['master'])
            .config("spark.executor.memory", self.config['spark']['executor_memory'])
            .config("spark.driver.memory", self.config['spark']['driver_memory'])
            .config("spark.sql.adaptive.enabled",
                   self.config['spark']['sql']['adaptive']['enabled'])
            .config("spark.sql.shuffle.partitions",
                   self.config['spark']['sql']['shuffle']['partitions'])
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                   "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )

        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logger.info(f"Spark session created: {spark.version}")
        return spark

    def load_raw_data(self, file_path: str):
        """Load raw CSV data into Spark DataFrame"""
        logger.info(f"Loading data from {file_path}")
        df = self.spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            encoding='utf-8'
        )
        logger.info(f"Loaded {df.count():,} records with {len(df.columns)} columns")
        return df

    def clean_data(self, df):
        """Apply data cleaning transformations"""
        logger.info("Starting data cleaning...")

        # 1. Remove duplicates
        initial_count = df.count()
        df = df.dropDuplicates(['ticket_id'])
        logger.info(f"Removed {initial_count - df.count():,} duplicate records")

        # 2. Filter Bangkok province only
        df = df.filter(
            col('province').rlike('(?i)(กรุงเทพ|Bangkok)')
        )
        logger.info(f"Filtered to Bangkok: {df.count():,} records")

        # 3. Drop rows with missing critical fields
        df = df.dropna(subset=['district', 'subdistrict', 'timestamp', 'coords'])

        # 4. Parse coordinates
        df = df.withColumn('lon',
                          split(col('coords'), ',').getItem(0).cast(DoubleType()))
        df = df.withColumn('lat',
                          split(col('coords'), ',').getItem(1).cast(DoubleType()))

        # 5. Parse timestamps
        df = df.withColumn('timestamp',
                          to_timestamp(col('timestamp')))
        df = df.withColumn('last_activity',
                          to_timestamp(col('last_activity')))

        # 6. Calculate resolution time
        df = df.withColumn('solve_days',
                          datediff(col('last_activity'), col('timestamp')))

        # 7. Extract temporal features
        df = (df
              .withColumn('year', year(col('timestamp')))
              .withColumn('month', month(col('timestamp')))
              .withColumn('day_of_week', dayofweek(col('timestamp')))
              .withColumn('hour', hour(col('timestamp')))
        )

        # 8. Clean text fields
        for text_col in ['comment', 'type', 'organization']:
            df = df.withColumn(
                text_col,
                when(col(text_col).isNull(), 'ไม่ระบุ')
                .otherwise(col(text_col))
            )

        # 9. Parse complaint types (multi-label)
        df = df.withColumn('type_clean',
                          regexp_replace(col('type'), r'[\{\}]', ''))
        df = df.withColumn('types_array',
                          split(col('type_clean'), ','))

        logger.info(f"Data cleaning complete: {df.count():,} clean records")
        return df

    def create_features(self, df):
        """Create engineered features for ML models"""
        logger.info("Creating engineered features...")

        # 1. Complaint category indicators
        common_types = ['น้ำท่วม', 'ถนน', 'ความสะอาด', 'ทางเท้า', 'จราจร']
        for cat in common_types:
            df = df.withColumn(
                f'is_{cat}',
                when(col('type_clean').contains(cat), 1).otherwise(0)
            )

        # 2. Status encoding
        df = df.withColumn('is_completed',
                          when(col('state') == 'เสร็จสิ้น', 1).otherwise(0))

        # 3. Has photo indicator
        df = df.withColumn('has_photo',
                          when(col('photo').isNotNull(), 1).otherwise(0))
        df = df.withColumn('has_photo_after',
                          when(col('photo_after').isNotNull(), 1).otherwise(0))

        # 4. Resolution efficiency (complaints per day)
        df = df.withColumn('resolution_speed',
                          when(col('solve_days') > 0,
                               1.0 / col('solve_days')).otherwise(0))

        # 5. Weekend flag
        df = df.withColumn('is_weekend',
                          when(col('day_of_week').isin([1, 7]), 1).otherwise(0))

        # 6. Season (Rainy: May-Oct, Dry: Nov-Apr)
        df = df.withColumn('season',
                          when(col('month').between(5, 10), 'rainy')
                          .otherwise('dry'))

        logger.info("Feature engineering complete")
        return df

    def aggregate_metrics(self, df):
        """Calculate aggregate metrics by district and category"""
        logger.info("Computing aggregate metrics...")

        # District-level aggregations
        district_metrics = (df
            .groupBy('district', 'year', 'month')
            .agg(
                count('ticket_id').alias('complaint_count'),
                avg('solve_days').alias('avg_solve_days'),
                spark_sum('is_completed').alias('completed_count'),
                avg('has_photo').alias('photo_rate')
            )
        )

        # Category-level aggregations
        category_metrics = (df
            .groupBy('type_clean', 'year', 'month')
            .agg(
                count('ticket_id').alias('complaint_count'),
                avg('solve_days').alias('avg_solve_days')
            )
        )

        return district_metrics, category_metrics

    def save_to_delta_lake(self, df, table_name: str):
        """Save DataFrame to Delta Lake with versioning"""
        delta_path = Path(self.config['data']['delta_lake_path']) / table_name
        logger.info(f"Saving to Delta Lake: {delta_path}")

        (df.write
         .format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .save(str(delta_path)))

        logger.info(f"Successfully saved {table_name} to Delta Lake")

    def run_pipeline(self, input_file: str):
        """Execute complete data processing pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Traffy Fondue Data Processing Pipeline")
        logger.info("=" * 80)

        # Load data
        df = self.load_raw_data(input_file)

        # Clean data
        df_clean = self.clean_data(df)

        # Create features
        df_features = self.create_features(df_clean)

        # Save main table
        self.save_to_delta_lake(df_features, "traffy_complaints")

        # Compute and save aggregations
        district_metrics, category_metrics = self.aggregate_metrics(df_features)
        self.save_to_delta_lake(district_metrics, "district_metrics")
        self.save_to_delta_lake(category_metrics, "category_metrics")

        # Show statistics
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total records processed: {df_features.count():,}")
        logger.info(f"Date range: {df_features.agg(spark_min('timestamp')).collect()[0][0]} to "
                   f"{df_features.agg(spark_max('timestamp')).collect()[0][0]}")
        logger.info(f"Unique districts: {df_features.select('district').distinct().count()}")
        logger.info(f"Unique complaint types: {df_features.select('type_clean').distinct().count()}")

        logger.info("\nTop 10 Districts by Complaint Volume:")
        (df_features
         .groupBy('district')
         .count()
         .orderBy(col('count').desc())
         .show(10, truncate=False))

        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)

        return df_features


def main():
    """Main execution function"""
    processor = TraffyDataProcessor()

    # Process the main dataset
    input_file = "bangkok_traffy.csv"  # Adjust path as needed
    df_result = processor.run_pipeline(input_file)

    # Optionally write to Parquet for other tools
    output_parquet = "data/processed/traffy_processed.parquet"
    logger.info(f"Saving to Parquet: {output_parquet}")
    df_result.write.mode("overwrite").parquet(output_parquet)

    processor.spark.stop()


if __name__ == "__main__":
    main()
