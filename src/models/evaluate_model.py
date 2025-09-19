import os
import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score


X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

if y_test.shape[1] == 1:
    y_test = y_test.values.ravel()

model = joblib.load('models/trained_model.pkl')

y_pred = model.predict(X_test)

# Save predictions
predictions = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
predictions.to_csv('data/processed_data/predictions.csv', index=False)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics
scores = {
    "MSE": mse, 
    "R2": r2
}
with open('metrics/scores.json', "w") as f:
    json.dump(scores, f, indent=4)

print("Eval complete")
print("Scores:", scores)