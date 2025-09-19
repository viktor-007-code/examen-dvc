import os
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

if y_train.shape[1] == 1:
    y_train = y_train.values.ravel()

rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [2, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4]
}

# Grid Search with 5-fold cross validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
print("Best parameters:", best_parameters)

joblib.dump(best_parameters, 'models/best_parameters.pkl')