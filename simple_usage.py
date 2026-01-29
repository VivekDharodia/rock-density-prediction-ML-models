"""
Simple Usage Script for Rock Density Predictor
Demonstrates how to use the trained model to make predictions
"""

from rock_density_predictor import RockDensityPredictor
import numpy as np

def main():
    print("\n" + "="*70)
    print("ROCK DENSITY PREDICTOR - USAGE EXAMPLE")
    print("="*70)
    
    # Option 1: Load a pre-trained model
    print("\n1. Loading the pre-trained model...")
    predictor = RockDensityPredictor(model_type='svr')  # SVR was the best model
    predictor.load_model('best_rock_density_model.pkl')
    
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)
    
    # Single prediction
    print("\nSingle Prediction:")
    signal = 45.5  # nanohertz
    density = predictor.predict(signal)[0]
    print(f"Signal Strength: {signal} nHz")
    print(f"Predicted Density: {density:.4f} kg/m³")
    
    # Multiple predictions
    print("\n\nMultiple Predictions:")
    signals = [5, 15, 30, 45, 60, 75, 90, 95]
    densities = predictor.predict(signals)
    
    print(f"\n{'Signal (nHz)':<15} {'Predicted Density (kg/m³)':<30}")
    print("-" * 45)
    for signal, density in zip(signals, densities):
        print(f"{signal:<15.1f} {density:<30.4f}")
    
    # Prediction range
    print("\n\nPrediction Range (0-100 nHz):")
    signal_range = np.linspace(0, 100, 11)  # 11 points from 0 to 100
    density_range = predictor.predict(signal_range)
    
    print(f"\n{'Signal (nHz)':<15} {'Predicted Density (kg/m³)':<30}")
    print("-" * 45)
    for signal, density in zip(signal_range, density_range):
        print(f"{signal:<15.1f} {density:<30.4f}")
    
    print("\n" + "="*70)
    print("TRAINING A NEW MODEL (Optional)")
    print("="*70)
    
    # Option 2: Train a new model from scratch
    print("\nTo train a new model with your own data:")
    print("1. Prepare your CSV file with two columns:")
    print("   - Column 1: Signal Strength (nHz)")
    print("   - Column 2: Rock Density (kg/m³)")
    print("\n2. Use this code:")
    print("""
    from rock_density_predictor import RockDensityPredictor
    
    # Initialize predictor (choose model type)
    predictor = RockDensityPredictor(model_type='svr')  # or 'random_forest', 'knn', etc.
    
    # Load your data
    X, y = predictor.load_data('your_data.csv')
    
    # Train the model
    results = predictor.train(X, y, test_size=0.2)
    
    # Visualize results
    predictor.plot_results(results, 'my_results.png')
    
    # Save the model
    predictor.save_model('my_model.pkl')
    
    # Make predictions
    predictions = predictor.predict([10, 20, 30, 40, 50])
    """)
    
    print("\n" + "="*70)
    print("AVAILABLE MODEL TYPES")
    print("="*70)
    print("""
    - 'svr' : Support Vector Regression (Best for this dataset - R² = 0.84)
    - 'knn' : K-Nearest Neighbors (R² = 0.81)
    - 'random_forest' : Random Forest (R² = 0.78)
    - 'gradient_boosting' : Gradient Boosting (R² = 0.76)
    - 'neural_network' : Neural Network (R² = 0.68)
    - 'polynomial' : Polynomial Regression
    - 'linear' : Linear Regression
    - 'ridge' : Ridge Regression
    """)
    
    print("\n" + "="*70)
    print("✓ Usage demonstration completed!")
    print("="*70)


if __name__ == "__main__":
    main()
