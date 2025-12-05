import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================
PARTIAL_DATA = False
ITERATIONS = 50
MODELS_DIR = "models"  # Update with your actual path
MODEL_PATH = f"{MODELS_DIR}\\DMTNN_{ITERATIONS}_{'PARTIAL' if PARTIAL_DATA else 'FULL'}"

# ============================================================================
# LOAD DATA
# ============================================================================
def load_test_data():
    """Load and prepare test data."""
    # Import here to avoid circular imports
    import TransCalc as transCalc
    from sklearn.model_selection import train_test_split
    
    print("Loading dataset...")
    dataSet = transCalc.readData(partial=PARTIAL_DATA)
    
    # Prepare features and target
    X = dataSet[['Hatch', 'Charge', 'Species', 'Level', 'Clone']]
    y = dataSet['Experience']
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# LOAD MODEL AND SCALERS
# ============================================================================
def load_trained_model():
    """Load the trained model and scalers."""
    print(f"Loading model from: {MODEL_PATH}.keras")
    model = load_model(f"{MODEL_PATH}.keras")
    
    print(f"Loading scalers from: {MODEL_PATH}.scalers")
    x_scaler, y_scaler = joblib.load(f"{MODEL_PATH}.scalers")
    
    print("Model summary:")
    model.summary()
    
    return model, x_scaler, y_scaler

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def make_predictions(model, x_scaler, y_scaler, X_test):
    """Make predictions on test data."""
    # Scale input
    X_scaled = x_scaler.transform(X_test)
    
    # Make prediction
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    
    # Inverse transform to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    return y_pred

# ============================================================================
# EVALUATION METRICS
# ============================================================================
def calculate_metrics(y_true, y_pred):
    """Calculate and display evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("="*60)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_predictions_vs_actual(y_test, y_pred, sample_size=100):
    """Plot predicted vs actual values."""
    # Sample data for cleaner plot
    if len(y_test) > sample_size:
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        y_test_sample = y_test.iloc[indices].values if hasattr(y_test, 'iloc') else y_test[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_test_sample = y_test.values if hasattr(y_test, 'iloc') else y_test
        y_pred_sample = y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Scatter plot: Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_sample, y_pred_sample, alpha=0.6, edgecolors='black', s=50)
    
    # Perfect prediction line
    min_val = min(y_test_sample.min(), y_pred_sample.min())
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Experience', fontsize=12)
    plt.ylabel('Predicted Experience', fontsize=12)
    plt.title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_test_sample - y_pred_sample
    plt.scatter(y_pred_sample, residuals, alpha=0.6, edgecolors='black', s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_test, y_pred):
    """Plot distribution of prediction errors."""
    errors = y_test - y_pred
    percentage_errors = ((y_test - y_pred) / y_test) * 100
    percentage_errors = percentage_errors[np.isfinite(percentage_errors)]  # Remove inf values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram of absolute errors
    axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution by magnitude
    error_bins = np.linspace(errors.min(), errors.max(), 20)
    bin_centers = (error_bins[:-1] + error_bins[1:]) / 2
    error_counts, _ = np.histogram(errors, bins=error_bins)
    
    axes[1].bar(bin_centers, error_counts, width=bin_centers[1]-bin_centers[0], 
                alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Error Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution by Magnitude', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Percentage error distribution
    axes[2].hist(percentage_errors, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Percentage Error (%)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_by_feature(X_test, y_test, y_pred):
    """Plot predictions grouped by different features."""
    features = ['Level', 'Clone', 'Charge', 'Species', 'Hatch']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features[:5]):
        if feature in X_test.columns:
            # Group by feature values
            unique_vals = X_test[feature].unique()
            if len(unique_vals) > 10:
                # Take top 10 most frequent values
                value_counts = X_test[feature].value_counts()
                unique_vals = value_counts.head(10).index
            
            avg_actual = []
            avg_predicted = []
            
            for val in unique_vals:
                mask = X_test[feature] == val
                if mask.sum() > 0:
                    avg_actual.append(y_test[mask].mean())
                    avg_predicted.append(y_pred[mask].mean())
            
            x_pos = np.arange(len(unique_vals))
            width = 0.35
            
            axes[idx].bar(x_pos - width/2, avg_actual, width, label='Actual', 
                         alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].bar(x_pos + width/2, avg_predicted, width, label='Predicted', 
                         alpha=0.7, color='lightcoral', edgecolor='black')
            axes[idx].set_xlabel(feature, fontsize=12)
            axes[idx].set_ylabel('Average Experience', fontsize=12)
            axes[idx].set_title(f'Predictions by {feature}', fontsize=13, fontweight='bold')
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels([str(v) for v in unique_vals], rotation=45)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Hide empty subplot if needed
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_top_worst_predictions(y_test, y_pred, X_test=None, top_n=10):
    """Plot the predictions with largest errors."""
    errors = np.abs(y_test - y_pred)
    sorted_indices = np.argsort(errors)[::-1]  # Descending order
    
    # Take top N worst predictions
    worst_indices = sorted_indices[:top_n]
    
    plt.figure(figsize=(14, 6))
    
    x_positions = np.arange(top_n)
    actual_values = y_test.iloc[worst_indices].values if hasattr(y_test, 'iloc') else y_test[worst_indices]
    predicted_values = y_pred[worst_indices]
    
    width = 0.35
    plt.bar(x_positions - width/2, actual_values, width, label='Actual', 
            alpha=0.7, color='skyblue', edgecolor='black')
    plt.bar(x_positions + width/2, predicted_values, width, label='Predicted', 
            alpha=0.7, color='lightcoral', edgecolor='black')
    
    # Add error values on top of bars
    for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        error = abs(actual - predicted)
        plt.text(i, max(actual, predicted) + (max(actual_values) * 0.05), 
                f'{error:.2f}', ha='center', fontsize=9)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Experience', fontsize=12)
    plt.title(f'Top {top_n} Worst Predictions (Largest Errors)', fontsize=14, fontweight='bold')
    plt.xticks(x_positions, [str(idx) for idx in worst_indices])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def print_sample_predictions(X_test, y_test, y_pred, num_samples=10):
    """Print sample predictions for inspection."""
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Select random samples
    indices = np.random.choice(len(y_test), min(num_samples, len(y_test)), replace=False)
    
    print(f"{'Index':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'% Error':<10}")
    print("-" * 60)
    
    for idx in indices:
        actual = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        predicted = y_pred[idx]
        error = actual - predicted
        pct_error = (error / actual) * 100 if actual != 0 else 0
        
        print(f"{idx:<8} {actual:<10.2f} {predicted:<10.2f} {error:<10.2f} {pct_error:<10.2f}%")

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================
def run_model_testing():
    """Main function to run all tests and visualizations."""
    print("="*70)
    print("NEURAL NETWORK MODEL TESTING")
    print("="*70)
    
    # 1. Load data
    X_train, X_test, y_train, y_test = load_test_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 2. Load model and scalers
    model, x_scaler, y_scaler = load_trained_model()
    
    # 3. Make predictions
    print("\nMaking predictions on test data...")
    y_pred = make_predictions(model, x_scaler, y_scaler, X_test)
    
    # 4. Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # 5. Plot visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Predictions vs Actual
    plot_predictions_vs_actual(y_test, y_pred)
    
    # Plot 2: Error distribution
    plot_error_distribution(y_test, y_pred)
    
    # Plot 3: Predictions by feature
    plot_predictions_by_feature(X_test, y_test, y_pred)
    
    # Plot 4: Top worst predictions
    plot_top_worst_predictions(y_test, y_pred, X_test)
    
    # 6. Print sample predictions
    print_sample_predictions(X_test, y_test, y_pred)
    
    # 7. Additional statistics
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    print(f"Average Actual Experience: {y_test.mean():.2f}")
    print(f"Average Predicted Experience: {y_pred.mean():.2f}")
    print(f"Std of Actual Experience: {y_test.std():.2f}")
    print(f"Std of Predicted Experience: {y_pred.std():.2f}")
    print(f"Min Actual Experience: {y_test.min():.2f}")
    print(f"Max Actual Experience: {y_test.max():.2f}")
    
    # 8. Error analysis
    absolute_errors = np.abs(y_test - y_pred)
    print(f"\nError Analysis:")
    print(f"Median Absolute Error: {np.median(absolute_errors):.2f}")
    print(f"90th Percentile Error: {np.percentile(absolute_errors, 90):.2f}")
    print(f"Maximum Error: {absolute_errors.max():.2f}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

# ============================================================================
# RUN THE TESTING
# ============================================================================
if __name__ == "__main__":
    run_model_testing()