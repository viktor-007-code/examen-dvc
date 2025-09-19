import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor


X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

if y_train.shape[1] == 1:
    y_train = y_train.values.ravel()

best_parameters = joblib.load('models/best_parameters.pkl')

model = RandomForestRegressor(random_state=42, **best_parameters)
model.fit(X_train, y_train)

joblib.dump(model, 'models/trained_model.pkl')
print("Model training completed")