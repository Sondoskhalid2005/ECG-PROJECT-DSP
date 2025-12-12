from features import build_feature_matrix
from preprocessing import X_train, X_test, y_train, y_test

# تحويل الإشارات إلى ميزات
X_train_features = build_feature_matrix(X_train)
X_test_features  = build_feature_matrix(X_test)

# مجرد طباعة للتأكد
print("Raw X_train shape:", X_train.shape)
print("Feature X_train shape:", X_train_features.shape)

print("Raw X_test shape:", X_test.shape)
print("Feature X_test shape:", X_test_features.shape)
