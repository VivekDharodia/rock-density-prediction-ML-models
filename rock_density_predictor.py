"""
Rock Density Prediction Model
Predicts rock density based on rebound signal strength (nanohertz)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


class RockDensityPredictor:
    """
    A comprehensive predictor for rock density based on nanohertz signal strength.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use. Options:
            - 'linear': Linear Regression
            - 'polynomial': Polynomial Regression (degree 2)
            - 'ridge': Ridge Regression
            - 'random_forest': Random Forest (default)
            - 'gradient_boosting': Gradient Boosting
            - 'svr': Support Vector Regression
            - 'neural_network': Neural Network
            - 'knn': K-Nearest Neighbors
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._get_model()
        self.poly_features = None
        self.feature_names = None
        
    def _get_model(self):
        """Select the appropriate model based on model_type."""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=15, 
                                                   random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, 
                                                          learning_rate=0.1, 
                                                          max_depth=5, 
                                                          random_state=42),
            'svr': SVR(kernel='rbf', C=100, gamma='scale'),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                          max_iter=1000, 
                                          random_state=42),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'polynomial': LinearRegression()
        }
        
        if self.model_type not in models:
            raise ValueError(f"Invalid model type. Choose from: {list(models.keys())}")
        
        return models[self.model_type]
    
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        tuple : (X, y) where X is signal strength and y is rock density
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Extract features and target
        X = df.iloc[:, 0].values.reshape(-1, 1)  # Signal strength
        y = df.iloc[:, 1].values  # Rock density
        
        self.feature_names = [df.columns[0]]
        
        print(f"\nFeature: {df.columns[0]}")
        print(f"Target: {df.columns[1]}")
        print(f"Number of samples: {len(X)}")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2):
        """
        Preprocess and split data.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # For polynomial regression, create polynomial features
        if self.model_type == 'polynomial':
            print("\nCreating polynomial features (degree 2)...")
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_train = self.poly_features.fit_transform(X_train)
            X_test = self.poly_features.transform(X_test)
        
        # Standardize features
        print("Standardizing features...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        dict : Training results and metrics
        """
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y, test_size)
        
        # Train model
        print(f"\n{'='*70}")
        print(f"Training {self.model_type.upper().replace('_', ' ')} model...")
        print('='*70)
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation score
        if self.model_type == 'polynomial':
            X_cv = self.poly_features.transform(X)
            X_cv = self.scaler.transform(X_cv)
        else:
            X_cv = self.scaler.transform(X)
        
        cv_scores = cross_val_score(self.model, X_cv, y, cv=5, 
                                    scoring='r2', n_jobs=-1)
        
        # Print results
        print("\n" + "="*70)
        print("MODEL PERFORMANCE")
        print("="*70)
        print(f"\nTraining Metrics:")
        print(f"  R² Score:  {train_r2:.4f}")
        print(f"  RMSE:      {train_rmse:.4f} kg/m³")
        print(f"  MAE:       {train_mae:.4f} kg/m³")
        
        print(f"\nTesting Metrics:")
        print(f"  R² Score:  {test_r2:.4f}")
        print(f"  RMSE:      {test_rmse:.4f} kg/m³")
        print(f"  MAE:       {test_mae:.4f} kg/m³")
        
        print(f"\nCross-Validation (5-fold):")
        print(f"  Mean R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print("\n⚠️  Warning: Model may be overfitting (large gap between train and test scores)")
        elif test_r2 > 0.8:
            print("\n✓ Excellent model performance!")
        elif test_r2 > 0.6:
            print("\n✓ Good model performance!")
        else:
            print("\n⚠️  Model could be improved with more data or feature engineering")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'X_train': X_train,
            'y_train': y_train,
            'y_train_pred': y_train_pred
        }
    
    def predict(self, signal_strength):
        """
        Predict rock density for new signal strength values.
        
        Parameters:
        -----------
        signal_strength : float or array-like
            Signal strength in nanohertz
            
        Returns:
        --------
        array : Predicted rock density in kg/m³
        """
        if isinstance(signal_strength, (int, float)):
            signal_strength = np.array([[signal_strength]])
        else:
            signal_strength = np.array(signal_strength).reshape(-1, 1)
        
        # Apply transformations
        if self.model_type == 'polynomial':
            signal_strength = self.poly_features.transform(signal_strength)
        
        signal_strength = self.scaler.transform(signal_strength)
        
        predictions = self.model.predict(signal_strength)
        return predictions
    
    def plot_results(self, results, save_path='model_results.png'):
        """
        Create visualizations of model performance.
        
        Parameters:
        -----------
        results : dict
            Results from training
        save_path : str
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted (Test set)
        axes[0, 0].scatter(results['y_test'], results['y_test_pred'], 
                          alpha=0.6, edgecolors='black', s=50)
        min_val = min(results['y_test'].min(), results['y_test_pred'].min())
        max_val = max(results['y_test'].max(), results['y_test_pred'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Rock Density (kg/m³)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Rock Density (kg/m³)', fontsize=11)
        axes[0, 0].set_title(f'Test Set: Actual vs Predicted (R² = {results["test_r2"]:.4f})', 
                           fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals (Test set)
        residuals = results['y_test'] - results['y_test_pred']
        axes[0, 1].scatter(results['y_test_pred'], residuals, 
                          alpha=0.6, edgecolors='black', s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Rock Density (kg/m³)', fontsize=11)
        axes[0, 1].set_ylabel('Residuals (kg/m³)', fontsize=11)
        axes[0, 1].set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of residuals
        axes[1, 0].hist(residuals, bins=20, color='skyblue', 
                       edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals (kg/m³)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Metrics comparison
        metrics = ['R² Score', 'RMSE', 'MAE']
        train_values = [results['train_r2'], results['train_rmse'], results['train_mae']]
        test_values = [results['test_r2'], results['test_rmse'], results['test_mae']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, train_values, width, label='Training', 
                      color='skyblue', edgecolor='black')
        axes[1, 1].bar(x + width/2, test_values, width, label='Testing', 
                      color='lightcoral', edgecolor='black')
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results visualization saved to '{save_path}'")
    
    def save_model(self, filepath='rock_density_model.pkl'):
        """Save the trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'poly_features': self.poly_features,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to '{filepath}'")
    
    def load_model(self, filepath='rock_density_model.pkl'):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.poly_features = model_data['poly_features']
        self.feature_names = model_data['feature_names']
        print(f"✓ Model loaded from '{filepath}'")


def compare_models(filepath):
    """
    Compare different model types and find the best one.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    """
    print("\n" + "="*70)
    print("COMPARING DIFFERENT MODELS")
    print("="*70)
    
    model_types = [
        'linear', 'polynomial', 'ridge', 
        'random_forest', 'gradient_boosting', 
        'svr', 'neural_network', 'knn'
    ]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Testing {model_type.upper().replace('_', ' ')} model...")
        print('='*70)
        
        try:
            predictor = RockDensityPredictor(model_type=model_type)
            X, y = predictor.load_data(filepath)
            result = predictor.train(X, y, test_size=0.2)
            
            results[model_type] = {
                'test_r2': result['test_r2'],
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae'],
                'cv_mean': result['cv_scores'].mean()
            }
        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")
            results[model_type] = None
    
    # Summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'CV Mean':<12}")
    print("-" * 70)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    for model_type, metrics in sorted(valid_results.items(), 
                                     key=lambda x: x[1]['test_r2'], 
                                     reverse=True):
        print(f"{model_type:<20} {metrics['test_r2']:<12.4f} "
              f"{metrics['test_rmse']:<12.4f} {metrics['test_mae']:<12.4f} "
              f"{metrics['cv_mean']:<12.4f}")
    
    # Best model
    best_model = max(valid_results.items(), key=lambda x: x[1]['test_r2'])
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model[0].upper().replace('_', ' ')}")
    print(f"   R² Score: {best_model[1]['test_r2']:.4f}")
    print("="*70)
    
    return best_model[0]


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROCK DENSITY PREDICTION MODEL")
    print("="*70)
    
    # You can specify the path to your CSV file here
    csv_file = '/mnt/user-data/uploads/rock_density_xray.csv'
    
    # Option 1: Compare all models and find the best
    print("\nRunning model comparison...")
    best_model_type = compare_models(csv_file)
    
    # Option 2: Train the best model
    print(f"\n\n{'='*70}")
    print(f"TRAINING BEST MODEL: {best_model_type.upper().replace('_', ' ')}")
    print('='*70)
    
    predictor = RockDensityPredictor(model_type=best_model_type)
    X, y = predictor.load_data(csv_file)
    results = predictor.train(X, y, test_size=0.2)
    
    # Create visualizations
    predictor.plot_results(results, 'rock_density_results.png')
    
    # Save model
    predictor.save_model('best_rock_density_model.pkl')
    
    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    test_signals = [10, 25, 50, 75, 90]
    
    print(f"\n{'Signal Strength (nHz)':<25} {'Predicted Density (kg/m³)':<30}")
    print("-" * 55)
    
    for signal in test_signals:
        density = predictor.predict(signal)[0]
        print(f"{signal:<25.1f} {density:<30.4f}")
    
    print("\n" + "="*70)
    print("✓ Model training completed successfully!")
    print("="*70)
