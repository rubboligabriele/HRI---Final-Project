import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import joblib

# Load feature and label data
X = np.load("dataset_1/labeled_features.npy")
y = np.load("dataset_1/labels.npy")

# Check basic info
print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")
print(f"Classes: {set(y)}")

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nClass distribution:")
print("  Train:", Counter(y_train))
print("  Test: ", Counter(y_test))

# Define parameter grid for SVM (RBF)
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1]
}

# Setup stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
print("\n Starting Grid Search...")
grid_search = GridSearchCV(
    SVC(kernel="rbf", probability=True),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model from grid search
model = grid_search.best_estimator_
print("\n Best parameters:", grid_search.best_params_)

# Save trained model
joblib.dump(model, "svm_cognitive_state.joblib")
print("\n Model saved as 'svm_cognitive_state.joblib'")