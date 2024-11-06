"""Forecasting number of claims."""

"""
Data contains 11 columns + IDpol (policy ID, links to the claims dataset):
    ClaimNb - Number of claims during the exposure period
    Exposure - The exposure period
    Area - The area code
    VehPower - The power of the car (ordered categorical)
    VehAge - The vehicle age, in years
    DrivAge - The driver age, in years
    BonusMalus - Between 50 and 350: <100 means bonus, >100 means malus
    VehBrand - The car brand (unknown categories)
    VehGas The car gas, diesel or regular
    Density - The density of inhabitants (number of inhabitants per km2) in 
    the city the driver of the car lives in
    Region - The policy regions
"""

# %% Import libraries and adjust settings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import xgboost as xgb
pd.set_option('display.max_columns', None)
sns.set_style("darkgrid")


# %% Data exploration


# Loading data #
data = pd.read_csv('freMTPL2freq.csv')
data = data.drop('IDpol', axis=1)  # used to link to database, not relevant

# Exploring basic info #
print(data.info(), end='\n\n')
print(data.dtypes, end='\n\n')
# Area, VehPower, VehBrand, VehGas and Region are categorical variables
# For now, they'll just be converted to categorical type
for column in ['Area', 'VehPower', 'VehBrand', 'VehGas', 'Region']:
    data[column] = data[column].astype('category')
print(data.dtypes, end='\n\n')

# Taking a quick glance at the data #
print(data.head(5), end='\n\n')
print(data.describe(), end='\n\n')

# Exploring target variable #
print(data.groupby('ClaimNb')['ClaimNb'].count(), end='\n\n')
# ClaimsNb (target) columns takes small values, with the maximum being 16

# Checking for missing values #
print('Missing data:')
print(data.isnull().sum())
# There is no missing data - no need to make any substitutions

'''
To-do:
    - Be more strict when looking at the min/max values and other outliers in 
    the data. If confident in knowledge of the dataset and the area which it 
    comes from, outliers can be removed in this stage of the analysis
'''

# %% Data visualisation - Univariate analysis

# Distribution of numerical data #
fig, ax = plt.subplots(3, 2, figsize=(10, 5))
num_cols = data.select_dtypes(include=np.number)
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        sns.histplot(num_cols[num_cols.columns[2*i+j]], ax=ax[i][j],
                     stat='density',
                     kde_kws=dict(cut=3))
plt.tight_layout()
# Exposure times up to 1 year
# Significant number with value < 0.5 - expecting a large difference between
# exposure-weighted and unweighted claims frequency

# Bar plots of categorical data #
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
cat_cols = data.select_dtypes(include='category')
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        if i == 2 and j == 1:
            continue
        ax[i, j].set_xlabel(cat_cols.columns[2*i+j])
        ax[i, j].set_ylabel('Frequency')
        col = cat_cols[cat_cols.columns[2*i+j]].value_counts()
        x, y = col.index, col.values
        ax[i, j].bar(x, y)
plt.tight_layout()

plt.figure(figsize=(10, 5))
reg = cat_cols['Region'].value_counts()
x_reg, y_reg = reg.index, reg.values
plt.bar(x_reg, y_reg)
plt.xlabel('Region')
plt.ylabel('Frequency')


# %% Data visualisation - Multivariate analysis

# Heatmap of correlation between numerical variables #
sns.heatmap(data=data.corr(), cmap='viridis', annot=True)
# High correlation between exposure and car/driver age, as well as between
# bonus/malus and driver age

# Frequency claims by area #
area_data = data[['Area', 'Exposure', 'ClaimNb']].groupby('Area',
                                                          as_index=False).sum()
area_freq = area_data['ClaimNb'] / area_data['Exposure']
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.barplot(area_data, x='Area', y='Exposure', ax=ax1, color='b')
sns.barplot(area_data, x='Area', y=area_freq, ax=ax2, color='b')
ax2.set(ylabel='Claim frequency')
plt.tight_layout()

# Frequency claims in the context of the car being used and its features #
car_data = data[['VehPower', 'VehAge', 'VehBrand', 'VehGas', 'ClaimNb',
                 'Exposure']]
# Grouping cars based on their age: < 5 (1), 5-10 (2), > 10 (3)
car_data.loc[car_data['VehAge'] < 5, 'VehAge'] = 1
car_data.loc[(car_data['VehAge'] >= 5) & (
    car_data['VehAge'] < 10), 'VehAge'] = 2
car_data.loc[car_data['VehAge'] >= 10, 'VehAge'] = 3

car_data = car_data.groupby(['VehPower', 'VehAge', 'VehBrand', 'VehGas']).agg(
    ClaimNb=pd.NamedAgg(column="ClaimNb", aggfunc='sum'),
    Exposure=pd.NamedAgg(column="Exposure", aggfunc='sum')).reset_index()

car_freq = car_data['ClaimNb'] / car_data['Exposure']

# Relationship between VehGas, VehBrand, VehAge and claim frequency
rel_plt_brand = sns.relplot(data=car_data,
                            x='VehBrand',
                            y=car_freq,
                            col='VehGas',
                            hue='VehAge',
                            kind='scatter')
rel_plt_brand.set_ylabels('Claim frequency')

# Relationship between VehGas, VehPower, VehAge and claim frequency
rel_plt_power = sns.relplot(data=car_data,
                            x='VehPower',
                            y=car_freq,
                            col='VehGas',
                            hue='VehAge',
                            kind='scatter')
rel_plt_power.set_ylabels('Claim frequency')

# Frequency claims by driver age #
driv_age_data = data[['DrivAge', 'Exposure', 'ClaimNb']]
# Create new column to group drivers by age (20, 25, 30 etc.)
driv_age_data['DrivAgeGr'] = np.round(
    driv_age_data['DrivAge']/5, 0).astype(int) * 5

driv_age_data = driv_age_data.groupby('DrivAgeGr', as_index=False).sum()
driv_age_freq = driv_age_data['ClaimNb'] / driv_age_data['Exposure']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
sns.barplot(driv_age_data, x='DrivAgeGr', y='Exposure', ax=ax1, color='b')
# ax1.set_xticks(x_tick_pos, x_tick_label)
sns.barplot(driv_age_data, x='DrivAgeGr', y=driv_age_freq, ax=ax2, color='b')
# ax2.set_xticks(x_tick_pos, x_tick_label)
ax2.set(ylabel='Claim frequency')
plt.tight_layout()


# Frequency claims by BonusMalus #
bonus_data = data[['BonusMalus', 'Exposure', 'ClaimNb']]
# Since this data is mostly smaller than 150, it will be set as a limit during
# visualisation. Values larger than 150 will be grouped together
bonus_data['BonusMalus'] = np.minimum(bonus_data['BonusMalus'], 150)
bonus_data['BonusMalusGr'] = np.round(
    bonus_data['BonusMalus']/5, 0).astype(int) * 5

bonus_data = bonus_data.groupby('BonusMalusGr', as_index=False).sum()
bonus_data_freq = bonus_data['ClaimNb'] / bonus_data['Exposure']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
sns.barplot(bonus_data, x='BonusMalusGr', y='Exposure', ax=ax1, color='b')
sns.barplot(bonus_data, x='BonusMalusGr',
            y=bonus_data_freq, ax=ax2, color='b')
ax2.set(ylabel='Claim frequency')
plt.tight_layout()

# Frequency claims by density #
density_data = data[['Density', 'ClaimNb', 'Exposure']].groupby(
    'Density', as_index=False).sum()

density_data_freq = density_data['ClaimNb'] / density_data['Exposure']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(density_data, x='Density', y='Exposure', ax=ax1, color='b')
sns.scatterplot(density_data, x='Density',
                y=density_data_freq, ax=ax2, color='b')
ax2.set(ylabel='Claim frequency')
plt.tight_layout()

'''
To-do:
    - explore connections between core features in more detail (for car models as
    well as other features)
    - geospatial analysis based on region
'''


# %% Algorithms - General Linear Models, Tweedie regression
data_GLM = data.copy()

# Conversion of categorical values #
# One-hot encoding will be used, since this dataset doesn't have many dimensions
# In other cases, depending on the data, different conversion method can/should
# be used.
'''
To-do:
    - Other possible solution, before proceeding, is to create subgroups in some
    features, similarly to how it was done during visualisation. Proper
    utilization of this usually requires real-life knowledge about the features,
    although visualisation can be really helpful.
'''
data_GLM = pd.get_dummies(data_GLM, drop_first=False, dtype='int')
print(data_GLM.shape)

# Split data into training and testing sets #
X, y = data_GLM.drop(['ClaimNb', 'Exposure'], axis=1), data_GLM['ClaimNb']
w = data_GLM['Exposure']
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, random_state=1)

# Define and fit model to training data #
glm_model = TweedieRegressor(power=1.5, alpha=1e-4, solver="newton-cholesky")
glm_model.fit(X_train, y_train, sample_weight=w_train)
'''
To-do:
    - By switching values for power parameter in the Tweedie regressor, different
    distributions are used. Optimal distribution can be derived by performing 
    grid search over values of power
    - Here, link function used is the default one (log-link), variation of this
    parameter can be performed to further optimize the model
    - Cross-validation should be performed 
'''

# Model evaluation - metrics #
y_pred = glm_model.predict(X_test)
mse_GLM = mean_squared_error(y_test, y_pred)
mae_GLM = mean_absolute_error(y_test, y_pred)
# Since counts (integers) are being modeled, accuracy score (used for
# classification) can be calculated by rounding predicted value to the nearest
# integer
acc_GLM = accuracy_score(y_test, np.rint(y_pred),)
print('Evaluation of GLM model')
print(f'Mean absolute error: {mae_GLM}')
print(f'Mean squared error: {mse_GLM}')
print(f'Accuracy score (%): {acc_GLM*100}')

# Feature importance - coefficients #
coef_df = pd.DataFrame({'Feature': X_train.columns,
                        'Coefficient': glm_model.coef_})
# inter_df = pd.DataFrame({'Feature': ['Intercept'],
#                          'Coefficient': [glm_model.intercept_]})
# coef_df = pd.concat([coef_df, inter_df], ignore_index=True)
coef_df.sort_values(by='Coefficient',
                    key=lambda x: np.abs(x),
                    ascending=False,
                    inplace=True)
fig, ax = plt.subplots()
sns.barplot(coef_df.head(10), x='Coefficient', y='Feature', color='b', ax=ax)
ax.set_title(f'GLM (intercept coefficient={np.round(glm_model.intercept_,4)})')

'''
To-do:
    - Modeling of claims frequency and comparison with the claim counts model
'''


# %% Algorithms - XGBoost
data_xgb = data.copy()

# Extract features and target #
X, y = data.drop(['ClaimNb', 'Exposure'], axis=1), data['ClaimNb']
w = data['Exposure']

# # Split data (0.25 test size) #
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, random_state=1)

# Define model and perform cross-validation/hyperparameter tuning #
n_estimators = 1000
xgb_model = xgb.XGBRegressor(objective='count:poisson',
                             n_estimators=n_estimators,
                             early_stopping_rounds=10,
                             enable_categorical=True)
param_grid = {'learning_rate': [0.05, 0.1, 0.3],
              'max_depth': [4, 6],
              'gamma': [0],
              'lambda': [1]}
cv_xgb_model = GridSearchCV(xgb_model,
                            param_grid=param_grid,
                            cv=5)
cv_xgb_model.fit(X_train,
                 y_train,
                 sample_weight=w_train,
                 eval_set=[(X_train, y_train), (X_test, y_test)])
best_xgb_model = cv_xgb_model.best_estimator_

# Fit model to training data #
best_xgb_model.fit(X_train,
                   y_train,
                   sample_weight=w_train,
                   eval_set=[(X_train, y_train), (X_test, y_test)])

# Model evaluation #
y_pred = best_xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred)
mae_xgb = mean_absolute_error(y_test, y_pred)
# Since counts (integers) are being modeled, accuracy score (used for
# classification) can be calculated by rounding predicted value to the nearest
# integer
acc_xgb = accuracy_score(y_test, np.rint(y_pred))
print('Evaluation of XGBoost model')
print(f'Mean absolute error: {mae_xgb}')
print(f'Mean squared error: {mse_xgb}')
print(f'Accuracy score (%): {acc_xgb*100}')

# Feature importance #
fig, ax = plt.subplots()
feat_imp = pd.Series(data=best_xgb_model.feature_importances_,
                     index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feat_imp.values, y=feat_imp.index, color='b')
ax.set_xlabel('Importance')
plt.tight_layout()

'''
To-do:
    - More exaustive hyperparameter tuning
    - Modeling of claims frequency and comparison with the claim counts model
'''

# %% Conclusion

'''
- Both models have high accuracy and low error metrics, although XGBoost per-
forms a bit better. It needs to be noted that the GLM used wasn't cross-validated
and it's hyperparameters weren't tuned.

- XGBoost's most important features make much more sense when compared to the
ones that GLM returns. It's evident that GLMs feature importance is heavily
influenced by the conversion of categorical variables and the increased dimens-
ionality - more attention should be paid to the conversion and possible grouping
of those values. XGBoost has an inherent advantage here since it can work with
categorical variables natively.

- For further analysis the overfitting should be measured and inspected in
detail when performing cross-validation and hyperparameter optimization.

- This problem can be assesed by modeling claims frequency, which was heavily
used in visualisation to infer relationships between the variables. The code
itself wouldn't be changed much, but the new results would need to be evaluated
independently.

'''
