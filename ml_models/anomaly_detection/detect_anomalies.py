"""
Anomaly Detection System for Urban Complaints
Identifies unusual patterns in complaint data using multiple methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintAnomalyDetector:
    """Multi-method anomaly detection for complaint data"""

    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = None
        self.anomaly_scores = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection"""
        logger.info("Preparing features for anomaly detection...")

        features = pd.DataFrame()

        # Temporal features
        features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        features['month'] = pd.to_datetime(df['timestamp']).dt.month

        # Geospatial features (if available)
        if 'lat' in df.columns and 'lon' in df.columns:
            features['lat'] = df['lat']
            features['lon'] = df['lon']

        # Complaint characteristics
        if 'solve_days' in df.columns:
            features['solve_days'] = df['solve_days'].fillna(0)

        # Category encoding (one-hot for main types)
        if 'type' in df.columns:
            features['is_flood'] = df['type'].str.contains('น้ำท่วม', na=False).astype(int)
            features['is_traffic'] = df['type'].str.contains('จราจร|ถนน', na=False).astype(int)
            features['is_waste'] = df['type'].str.contains('ความสะอาด|ขยะ', na=False).astype(int)

        # Aggregated features (complaints per district per day)
        if 'district' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            district_daily = df.groupby(['district', 'date']).size().reset_index(name='daily_count')

            # Merge back
            df_with_count = df.merge(district_daily, on=['district', 'date'], how='left')
            features['district_daily_count'] = df_with_count['daily_count']

        logger.info(f"Prepared {len(features.columns)} features: {list(features.columns)}")
        return features

    def detect_isolation_forest(self, features: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        logger.info("Running Isolation Forest anomaly detection...")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )

        # Predict (-1 for anomalies, 1 for normal)
        predictions = self.iso_forest.fit_predict(features_scaled)
        anomaly_scores = self.iso_forest.score_samples(features_scaled)

        n_anomalies = (predictions == -1).sum()
        logger.info(f"Isolation Forest detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")

        return predictions, anomaly_scores

    def detect_statistical(self, df: pd.DataFrame, column: str = 'solve_days',
                          threshold: float = 3.0) -> np.ndarray:
        """Detect anomalies using statistical Z-score method"""
        logger.info(f"Running statistical anomaly detection on '{column}'...")

        if column not in df.columns:
            logger.warning(f"Column '{column}' not found")
            return np.zeros(len(df))

        values = df[column].fillna(df[column].median())

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(values))

        # Anomalies are points with |z-score| > threshold
        anomalies = z_scores > threshold

        n_anomalies = anomalies.sum()
        logger.info(f"Statistical method detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")

        return anomalies.astype(int)

    def detect_spatial_clusters(self, df: pd.DataFrame, eps=0.01, min_samples=5) -> np.ndarray:
        """Detect spatial anomalies using DBSCAN clustering"""
        logger.info("Running spatial anomaly detection with DBSCAN...")

        if 'lat' not in df.columns or 'lon' not in df.columns:
            logger.warning("Latitude/Longitude not found")
            return np.zeros(len(df))

        # Extract coordinates
        coords = df[['lat', 'lon']].dropna()

        if len(coords) == 0:
            return np.zeros(len(df))

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(coords)

        # Points with label -1 are noise/anomalies
        anomalies = (clusters == -1).astype(int)

        n_anomalies = anomalies.sum()
        logger.info(f"DBSCAN detected {n_anomalies} spatial anomalies ({n_anomalies/len(coords)*100:.2f}%)")

        # Align with original dataframe
        result = np.zeros(len(df))
        result[coords.index] = anomalies

        return result

    def detect_temporal_spikes(self, df: pd.DataFrame, window: int = 7,
                              threshold_std: float = 3.0) -> np.ndarray:
        """Detect temporal spikes in complaint volume"""
        logger.info("Running temporal spike detection...")

        # Group by date
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby('date').size()

        # Calculate rolling statistics
        rolling_mean = daily_counts.rolling(window=window, center=True).mean()
        rolling_std = daily_counts.rolling(window=window, center=True).std()

        # Detect spikes
        z_scores = (daily_counts - rolling_mean) / (rolling_std + 1e-10)
        spike_dates = daily_counts[np.abs(z_scores) > threshold_std].index

        logger.info(f"Detected {len(spike_dates)} days with anomalous complaint volumes")

        # Map back to original dataframe
        anomalies = df['date'].isin(spike_dates).astype(int)

        return anomalies.values

    def combine_methods(self, anomalies_dict: dict, weights: dict = None) -> np.ndarray:
        """Combine multiple anomaly detection methods using weighted voting"""
        if weights is None:
            # Equal weights by default
            weights = {method: 1.0 for method in anomalies_dict.keys()}

        logger.info(f"Combining {len(anomalies_dict)} detection methods with weights: {weights}")

        combined_score = np.zeros(len(list(anomalies_dict.values())[0]))

        for method, anomalies in anomalies_dict.items():
            weight = weights.get(method, 1.0)
            # Convert -1/1 labels to 0/1 if needed
            anomalies_binary = (anomalies == -1).astype(int) if anomalies.min() < 0 else anomalies
            combined_score += weight * anomalies_binary

        # Normalize
        max_score = sum(weights.values())
        combined_score /= max_score

        # Threshold for final anomaly classification
        final_anomalies = (combined_score > 0.5).astype(int)

        n_final = final_anomalies.sum()
        logger.info(f"Combined method identified {n_final} final anomalies ({n_final/len(final_anomalies)*100:.2f}%)")

        return final_anomalies, combined_score

    def visualize_anomalies(self, df: pd.DataFrame, anomalies: np.ndarray,
                           method_name: str = "Combined"):
        """Visualize detected anomalies"""
        logger.info(f"Creating visualizations for {method_name} method...")

        # Create output directory
        Path("ml_models/anomaly_detection/outputs").mkdir(parents=True, exist_ok=True)

        # 1. Time series plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby('date').size()

        # Get anomaly dates
        df['anomaly'] = anomalies
        anomaly_dates = df[df['anomaly'] == 1]['date'].unique()

        # Plot daily complaints
        axes[0].plot(daily_counts.index, daily_counts.values, label='Daily Complaints', alpha=0.7)
        axes[0].scatter(anomaly_dates,
                       daily_counts.loc[anomaly_dates],
                       color='red', s=100, label='Anomalies', zorder=5)
        axes[0].set_title(f'Daily Complaint Volume with Anomalies ({method_name})')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Number of Complaints')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Spatial plot (if coordinates available)
        if 'lat' in df.columns and 'lon' in df.columns:
            normal = df[df['anomaly'] == 0]
            anomalous = df[df['anomaly'] == 1]

            axes[1].scatter(normal['lon'], normal['lat'],
                          c='blue', alpha=0.1, s=10, label='Normal')
            axes[1].scatter(anomalous['lon'], anomalous['lat'],
                          c='red', alpha=0.6, s=50, label='Anomalies', edgecolors='black')
            axes[1].set_title(f'Spatial Distribution of Anomalies ({method_name})')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'ml_models/anomaly_detection/outputs/anomalies_{method_name.lower()}.png', dpi=300)
        logger.info(f"Visualization saved for {method_name}")

    def generate_report(self, df: pd.DataFrame, anomalies: np.ndarray) -> pd.DataFrame:
        """Generate detailed anomaly report"""
        logger.info("Generating anomaly report...")

        df['anomaly_flag'] = anomalies
        anomaly_df = df[df['anomaly_flag'] == 1].copy()

        # Summary statistics
        report = {
            'total_complaints': len(df),
            'total_anomalies': len(anomaly_df),
            'anomaly_rate': len(anomaly_df) / len(df) * 100,
            'anomaly_dates': anomaly_df['timestamp'].dt.date.unique().tolist()[:10]  # First 10
        }

        if 'district' in anomaly_df.columns:
            report['top_districts'] = anomaly_df['district'].value_counts().head(5).to_dict()

        if 'type' in anomaly_df.columns:
            report['top_types'] = anomaly_df['type'].value_counts().head(5).to_dict()

        logger.info("\n" + "=" * 80)
        logger.info("ANOMALY DETECTION REPORT")
        logger.info("=" * 80)
        for key, value in report.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 80)

        return anomaly_df

    def save_results(self, anomalies: np.ndarray, scores: np.ndarray = None,
                    filename: str = "anomaly_results.csv"):
        """Save anomaly detection results"""
        results = pd.DataFrame({
            'anomaly_flag': anomalies
        })

        if scores is not None:
            results['anomaly_score'] = scores

        output_path = f"ml_models/anomaly_detection/outputs/{filename}"
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main anomaly detection pipeline"""
    logger.info("=" * 80)
    logger.info("Complaint Anomaly Detection System")
    logger.info("=" * 80)

    # Create output directories
    Path("ml_models/anomaly_detection/outputs").mkdir(parents=True, exist_ok=True)

    # Load data (simulated for demonstration)
    logger.info("Loading data...")
    np.random.seed(42)
    date_range = pd.date_range(start='2021-08-01', end='2025-01-31', freq='H')

    # Simulate normal pattern with some anomalies
    n_samples = len(date_range)
    normal_data = np.random.poisson(lam=50, size=n_samples)

    # Inject anomalies (spikes)
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    normal_data[anomaly_indices] += np.random.randint(100, 300, size=len(anomaly_indices))

    df = pd.DataFrame({
        'timestamp': date_range,
        'complaint_count': normal_data,
        'district': np.random.choice(['ปทุมวัน', 'ห้วยขวาง', 'ดินแดง', 'คลองเตย'], n_samples),
        'type': np.random.choice(['น้ำท่วม', 'จราจร', 'ความสะอาด', 'ถนน'], n_samples),
        'lat': 13.7563 + np.random.uniform(-0.1, 0.1, n_samples),
        'lon': 100.5018 + np.random.uniform(-0.1, 0.1, n_samples),
        'solve_days': np.random.gamma(shape=2, scale=15, size=n_samples)
    })

    # Initialize detector
    detector = ComplaintAnomalyDetector(contamination=0.05)

    # Prepare features
    features = detector.prepare_features(df)

    # Run multiple detection methods
    anomalies_dict = {}

    # 1. Isolation Forest
    iso_predictions, iso_scores = detector.detect_isolation_forest(features)
    anomalies_dict['isolation_forest'] = iso_predictions

    # 2. Statistical method
    stat_anomalies = detector.detect_statistical(df, column='solve_days', threshold=3.0)
    anomalies_dict['statistical'] = stat_anomalies

    # 3. Spatial clustering
    spatial_anomalies = detector.detect_spatial_clusters(df, eps=0.01, min_samples=5)
    anomalies_dict['spatial'] = spatial_anomalies

    # 4. Temporal spikes
    temporal_anomalies = detector.detect_temporal_spikes(df, window=7, threshold_std=3.0)
    anomalies_dict['temporal'] = temporal_anomalies

    # Combine methods
    final_anomalies, combined_scores = detector.combine_methods(
        anomalies_dict,
        weights={
            'isolation_forest': 2.0,
            'statistical': 1.0,
            'spatial': 1.0,
            'temporal': 1.5
        }
    )

    # Visualize results
    detector.visualize_anomalies(df, final_anomalies, method_name="Combined")

    # Generate report
    anomaly_report = detector.generate_report(df, final_anomalies)

    # Save results
    detector.save_results(final_anomalies, combined_scores)

    # Save individual method results
    for method_name, anomalies in anomalies_dict.items():
        detector.visualize_anomalies(df, (anomalies == -1).astype(int) if anomalies.min() < 0 else anomalies,
                                    method_name=method_name)

    logger.info("=" * 80)
    logger.info("Anomaly detection pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
