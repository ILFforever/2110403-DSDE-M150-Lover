"""
LSTM Time-Series Forecasting Model for Urban Complaint Prediction
Predicts complaint volumes by category and location
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintForecaster:
    """LSTM-based forecasting system for urban complaints"""

    def __init__(self, lookback_days=30, forecast_horizon=7):
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'complaint_count'):
        """Prepare time-series data for LSTM training"""
        logger.info(f"Preparing data with lookback={self.lookback_days}, horizon={self.forecast_horizon}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Resample to daily counts
        daily_counts = df.resample('D').size().to_frame(name=target_col)

        # Fill missing dates
        date_range = pd.date_range(start=daily_counts.index.min(),
                                   end=daily_counts.index.max(),
                                   freq='D')
        daily_counts = daily_counts.reindex(date_range, fill_value=0)

        # Normalize data
        scaled_data = self.scaler.fit_transform(daily_counts.values.reshape(-1, 1))

        return scaled_data, daily_counts

    def create_sequences(self, data):
        """Create input-output sequences for LSTM"""
        X, y = [], []

        for i in range(len(data) - self.lookback_days - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback_days])
            y.append(data[i + self.lookback_days:i + self.lookback_days + self.forecast_horizon])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build Bidirectional LSTM architecture"""
        logger.info("Building LSTM model architecture...")

        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),

            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),

            LSTM(32, return_sequences=False),
            Dropout(0.2),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(self.forecast_horizon)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )

        logger.info(model.summary())
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the LSTM model"""
        logger.info(f"Training model for {epochs} epochs...")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'ml_models/forecasting/checkpoints/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed!")
        return self.history

    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Inverse transform to original scale
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_original = self.scaler.inverse_transform(predictions_reshaped)
        return predictions_original.reshape(predictions.shape)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")

        predictions = self.predict(X_test)
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        # Calculate metrics
        mae = mean_absolute_error(y_test_original, predictions)
        mse = mean_squared_error(y_test_original, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test_original, predictions) * 100

        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAPE: {mape:.2f}%")

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': predictions,
            'actuals': y_test_original
        }

    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('ml_models/forecasting/outputs/training_history.png', dpi=300)
        logger.info("Training history plot saved")

    def plot_predictions(self, actuals, predictions, dates=None):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(15, 6))

        # Plot first 100 forecasts
        n_samples = min(100, len(actuals))

        for i in range(n_samples):
            if i == 0:
                plt.plot(range(self.forecast_horizon), actuals[i], 'b-', alpha=0.3, label='Actual')
                plt.plot(range(self.forecast_horizon), predictions[i], 'r--', alpha=0.3, label='Predicted')
            else:
                plt.plot(range(self.forecast_horizon), actuals[i], 'b-', alpha=0.1)
                plt.plot(range(self.forecast_horizon), predictions[i], 'r--', alpha=0.1)

        plt.title(f'Complaint Volume Forecasts ({self.forecast_horizon}-day ahead)')
        plt.xlabel('Days Ahead')
        plt.ylabel('Number of Complaints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ml_models/forecasting/outputs/predictions.png', dpi=300)
        logger.info("Predictions plot saved")

    def save_model(self, path='ml_models/forecasting/models/lstm_model.h5'):
        """Save trained model and scaler"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        joblib.dump(self.scaler, path.replace('.h5', '_scaler.pkl'))
        logger.info(f"Model saved to {path}")

    def load_model(self, path='ml_models/forecasting/models/lstm_model.h5'):
        """Load trained model and scaler"""
        self.model = keras.models.load_model(path)
        self.scaler = joblib.load(path.replace('.h5', '_scaler.pkl'))
        logger.info(f"Model loaded from {path}")


class ProphetForecaster:
    """Prophet-based forecasting (alternative approach)"""

    def __init__(self):
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            logger.warning("Prophet not installed. Install with: pip install prophet")
            self.Prophet = None

    def train_and_forecast(self, df: pd.DataFrame, periods=30):
        """Train Prophet model and make forecasts"""
        if self.Prophet is None:
            logger.error("Prophet not available")
            return None

        logger.info("Training Prophet model...")

        # Prepare data for Prophet
        df_prophet = df.reset_index()
        df_prophet.columns = ['ds', 'y']

        # Initialize and fit model
        model = self.Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        model.fit(df_prophet)

        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        logger.info(f"Prophet forecast completed for {periods} days")

        return model, forecast


def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("LSTM Complaint Forecasting Model Training")
    logger.info("=" * 80)

    # Create output directories
    Path("ml_models/forecasting/models").mkdir(parents=True, exist_ok=True)
    Path("ml_models/forecasting/outputs").mkdir(parents=True, exist_ok=True)
    Path("ml_models/forecasting/checkpoints").mkdir(parents=True, exist_ok=True)

    # Load data (you'll need to adjust the path)
    logger.info("Loading data...")
    # Simulated data for demonstration
    date_range = pd.date_range(start='2021-08-01', end='2025-01-31', freq='D')
    np.random.seed(42)

    # Simulate complaint pattern with trend and seasonality
    trend = np.linspace(100, 150, len(date_range))
    seasonality = 30 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365)
    noise = np.random.normal(0, 10, len(date_range))
    complaint_counts = trend + seasonality + noise

    df = pd.DataFrame({
        'timestamp': date_range,
        'complaint_count': complaint_counts
    })

    # Initialize forecaster
    forecaster = ComplaintForecaster(lookback_days=30, forecast_horizon=7)

    # Prepare data
    scaled_data, daily_counts = forecaster.prepare_data(df)

    # Create sequences
    X, y = forecaster.create_sequences(scaled_data)
    logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")

    # Split data (80-10-10)
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")

    # Train model
    forecaster.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Evaluate
    results = forecaster.evaluate(X_test, y_test)

    # Plot results
    forecaster.plot_training_history()
    forecaster.plot_predictions(results['actuals'], results['predictions'])

    # Save model
    forecaster.save_model()

    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
