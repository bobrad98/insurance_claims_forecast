# Claims Forecasting Model

This code was developed as part of an interview assignment for an insurance company. It provides an end-to-end process for forecasting the number of insurance claims, using a combination of data analysis, visualization, and model training with Generalized Linear Models (GLMs) and XGBoost.

## Overview

### Data
The dataset contains policy-level information, including claims, exposure periods, and policyholder and vehicle characteristics.

### Key Steps
1. **Data Exploration**: Initial data loading, cleaning, and exploration to understand variable types, distributions, and relationships.
2. **Data Visualization**: Univariate and multivariate analyses, exploring claim frequency across key features such as driver age, vehicle power, and region.
3. **Feature Engineering**: One-hot encoding for categorical variables, grouping features, and creating exposure-weighted frequencies.
4. **Modeling**
   - **GLM**: Trains a Tweedie regression model for claim forecasting, with feature importance visualizations.
   - **XGBoost**: Trains an XGBoost regressor, including cross-validation and hyperparameter tuning to optimize performance.

### Results
Evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and accuracy for both models. The code demonstrates the strengths of XGBoost in handling categorical variables directly, yielding more interpretable feature importances and higher predictive accuracy.
