import joblib
from sklearn.neighbors import KNeighborsClassifier
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
print("Random Forest: ")
print(classification_report(y_test, pred))

# 5) Save
joblib.dump(model, "model.pkl")   
joblib.dump(scaler, "scaler.pkl")  


k_range = range(1, 5)
for k in k_range:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train_scaled, y_train)
        y_pred = knn_model.predict(X_test_scaled)
        print(f"KNN K = {k}:")
        print(classification_report(y_test, y_pred))

    

print("Model saved successfully!")

