import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing import X_train, X_test, y_train, y_test
from features import build_feature_matrix
from sklearn.metrics import classification_report

# 1) استخراج الميزات من البيانات
X_train_features = build_feature_matrix(X_train)
X_test_features  = build_feature_matrix(X_test)

# 2) Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled  = scaler.transform(X_test_features)

# 3) Model training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# 4) Evaluation
pred = model.predict(X_test_scaled)
print(classification_report(y_test, pred))

# 5) Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
