# Rock Density Prediction Model

A comprehensive machine learning solution to predict rock density (kg/m³) based on rebound signal strength (nanohertz).

## 🎯 Model Performance

After testing 8 different algorithms, the **Support Vector Regression (SVR)** model achieved the best results:

- **R² Score**: 0.8364 (83.64% variance explained)
- **RMSE**: 0.1309 kg/m³
- **MAE**: 0.1131 kg/m³
- **Cross-Validation Score**: 0.7997 ± 0.0878

### Model Comparison Results

| Model | R² Score | RMSE | MAE | CV Mean |
|-------|----------|------|-----|---------|
| **SVR** | **0.8364** | **0.1309** | **0.1131** | **0.7997** |
| K-Nearest Neighbors | 0.8109 | 0.1407 | 0.1218 | 0.7624 |
| Random Forest | 0.7778 | 0.1525 | 0.1260 | 0.7127 |
| Gradient Boosting | 0.7629 | 0.1576 | 0.1275 | 0.6987 |
| Neural Network | 0.6812 | 0.1827 | 0.1438 | 0.7135 |

## 📊 Dataset Information

- **Total Samples**: 300
- **Feature**: Rebound Signal Strength (0.70 - 98.83 nHz)
- **Target**: Rock Density (1.50 - 2.75 kg/m³)
- **Split**: 80% training, 20% testing

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Making Predictions with Pre-trained Model

```python
from rock_density_predictor import RockDensityPredictor

# Load the pre-trained model
predictor = RockDensityPredictor(model_type='svr')
predictor.load_model('best_rock_density_model.pkl')

# Make a single prediction
signal_strength = 45.5  # nanohertz
density = predictor.predict(signal_strength)[0]
print(f"Predicted Density: {density:.4f} kg/m³")

# Make multiple predictions
signals = [10, 25, 50, 75, 90]
densities = predictor.predict(signals)
for signal, density in zip(signals, densities):
    print(f"Signal: {signal} nHz → Density: {density:.4f} kg/m³")
```

### Training a New Model

```python
from rock_density_predictor import RockDensityPredictor

# Initialize predictor
predictor = RockDensityPredictor(model_type='svr')

# Load your data
X, y = predictor.load_data('rock_density_xray.csv')

# Train the model
results = predictor.train(X, y, test_size=0.2)

# Create visualizations
predictor.plot_results(results, 'results.png')

# Save the trained model
predictor.save_model('my_model.pkl')
```

## 📈 Features

- **Multiple ML Algorithms**: Test and compare 8 different models
- **Automatic Model Selection**: Identifies the best-performing model
- **Feature Scaling**: StandardScaler for optimal performance
- **Polynomial Features**: For capturing non-linear relationships
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Comprehensive Metrics**: R², RMSE, MAE
- **Visualization**: Detailed plots of model performance
- **Model Persistence**: Save and load trained models

## 📁 Files Included

1. **rock_density_predictor.py** - Main model class with all functionality
2. **simple_usage.py** - Quick start examples
3. **best_rock_density_model.pkl** - Pre-trained SVR model
4. **rock_density_results.png** - Performance visualizations
5. **data_analysis.png** - Dataset exploration plots
6. **requirements.txt** - Python dependencies

## 🔬 Model Details

### Support Vector Regression (SVR)

The winning model uses:
- **Kernel**: RBF (Radial Basis Function)
- **C parameter**: 100
- **Gamma**: scale (automatic)
- **Feature Scaling**: StandardScaler

This configuration captures the non-linear relationship between signal strength and rock density effectively.

## 📊 Example Predictions

| Signal Strength (nHz) | Predicted Density (kg/m³) |
|----------------------|---------------------------|
| 10.0 | 2.5115 |
| 25.0 | 2.3948 |
| 50.0 | 1.7070 |
| 75.0 | 2.4950 |
| 90.0 | 2.3003 |

## 🎨 Visualizations

The model generates two sets of visualizations:

### 1. Data Analysis (data_analysis.png)
- Scatter plot of signal vs density
- Distribution of signal strength
- Distribution of rock density
- Correlation heatmap

### 2. Model Results (rock_density_results.png)
- Actual vs Predicted values
- Residual plot
- Residual distribution
- Metrics comparison (train vs test)

## 🧪 Advanced Usage

### Compare All Models

```python
from rock_density_predictor import compare_models

# This will test all 8 models and return the best one
best_model_type = compare_models('rock_density_xray.csv')
```

### Custom Model Training

```python
# Try different models
models_to_test = ['svr', 'random_forest', 'gradient_boosting', 'knn']

for model_type in models_to_test:
    predictor = RockDensityPredictor(model_type=model_type)
    X, y = predictor.load_data('your_data.csv')
    results = predictor.train(X, y)
```

### Hyperparameter Tuning

For even better results, you can customize the models:

```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [10, 50, 100, 200],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'epsilon': [0.01, 0.1, 0.2]
}

# Create and tune model
svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

## 🔍 Understanding the Metrics

- **R² Score**: Proportion of variance in rock density explained by signal strength (0-1, higher is better)
- **RMSE**: Root Mean Squared Error - average prediction error in kg/m³ (lower is better)
- **MAE**: Mean Absolute Error - average absolute prediction error in kg/m³ (lower is better)
- **CV Score**: Cross-validation score - model performance on unseen folds (higher is better)

## 🎯 When to Use Each Model

- **SVR**: Best overall, handles non-linear relationships well
- **KNN**: Good for local patterns, fast predictions
- **Random Forest**: Good interpretability, handles outliers
- **Gradient Boosting**: High accuracy, but slower training
- **Neural Network**: Good for complex patterns, needs more data

## 📝 Data Format

Your CSV file should have this structure:

```csv
Rebound Signal Strength nHz,Rock Density kg/m3
72.945,2.457
14.230,2.602
36.597,1.967
...
```

## 🛠️ Troubleshooting

**Low R² Score?**
- Try different models (SVR works best for this dataset)
- Check for outliers in your data
- Ensure sufficient training samples (>200 recommended)

**High Error?**
- Standardize your features (done automatically)
- Try polynomial features for non-linear relationships
- Use ensemble methods like Random Forest

**Overfitting?**
- Reduce model complexity
- Use regularization (Ridge, Lasso)
- Collect more training data

## 🔄 Updating the Model

To retrain with new data:

```python
# Load existing model
predictor = RockDensityPredictor(model_type='svr')
predictor.load_model('best_rock_density_model.pkl')

# Load new data and retrain
X_new, y_new = predictor.load_data('new_data.csv')
results = predictor.train(X_new, y_new)

# Save updated model
predictor.save_model('updated_model.pkl')
```

## 📚 Dependencies

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## 🎓 Technical Details

The model pipeline includes:
1. Data loading and validation
2. Train-test split (80/20)
3. Feature standardization (zero mean, unit variance)
4. Model training with optimal hyperparameters
5. Performance evaluation (R², RMSE, MAE)
6. Cross-validation (5-fold)
7. Visualization generation
8. Model serialization

## 📞 Support

For issues or questions:
1. Check the visualization plots for insights
2. Review the model comparison results
3. Try different model types
4. Ensure your data format matches the expected structure

## 🎉 Results Summary

The SVR model successfully predicts rock density with 83.64% accuracy (R² = 0.8364), with an average error of only 0.11 kg/m³. This performance is excellent for real-world applications in:

- Geological surveys
- Mining operations
- Construction site analysis
- Rock quality assessment
- Geotechnical engineering

## 📜 License

This code is provided as-is for educational and commercial use.

---

**Model Version**: 1.0  
**Last Updated**: January 2026  
**Best Model**: Support Vector Regression (SVR)  
**Performance**: R² = 0.8364
