import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load data
cols = ['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']
df = pd.read_csv('abalone.data', names=cols)
X, y = df[['Sex','Length','Diameter','Height','Whole_weight','Shell_weight']], df['Rings']
# One-hot encode
ohe = OneHotEncoder(sparse_output=False)
sex_df = pd.DataFrame(ohe.fit_transform(X[['Sex']]), columns=[f'Sex_{c}' for c in ohe.categories_[0]], index=X.index)
X = pd.concat([X.drop('Sex', axis=1), sex_df], axis=1)
# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num = ['Length','Diameter','Height','Whole_weight','Shell_weight']
scaler = StandardScaler()
X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
X_train_scaled[num] = scaler.fit_transform(X_train[num])
X_test_scaled[num] = scaler.transform(X_test[num])
# Train & predict
model = LinearRegression().fit(X_train_scaled, y_train)
y_tr_pred, y_te_pred = model.predict(X_train_scaled), model.predict(X_test_scaled)
# Metrics
tr_mae, te_mae = mean_absolute_error(y_train, y_tr_pred), mean_absolute_error(y_test, y_te_pred)
tr_rmse, te_rmse = np.sqrt(mean_squared_error(y_train, y_tr_pred)), np.sqrt(mean_squared_error(y_test, y_te_pred))
tr_r2, te_r2 = r2_score(y_train, y_tr_pred), r2_score(y_test, y_te_pred)
# Print results
print("="*60 + "\nMODEL PERFORMANCE METRICS\n" + "="*60)
print(f"\n{'Metric':<15} {'Training':<15} {'Test':<15}\n{'-'*45}")
print(f"{'MAE':<15} {tr_mae:<15.4f} {te_mae:<15.4f}")
print(f"{'RMSE':<15} {tr_rmse:<15.4f} {te_rmse:<15.4f}")
print(f"{'R²':<15} {tr_r2:<15.4f} {te_r2:<15.4f}")
print("\n" + "="*60 + "\nFEATURE COEFFICIENTS\n" + "="*60)
feature_names = num + [f'Sex_{c}' for c in ohe.categories_[0]]
print(pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False).to_string(index=False))
print(f"\nIntercept: {model.intercept_:.4f}")
print("\n" + "="*60 + "\nADDITIONAL INFORMATION\n" + "="*60)
print(f"Total samples: {len(df)}\nTraining samples: {len(X_train)}\nTest samples: {len(X_test)}")
print(f"Unique Sex categories: {ohe.categories_[0].tolist()}")
print(f"\nOverfitting check:\nTraining R² - Test R² difference: {tr_r2 - te_r2:.4f}")
print("Warning: Possible overfitting detected!" if tr_r2 - te_r2 > 0.1 else "No significant overfitting detected.")