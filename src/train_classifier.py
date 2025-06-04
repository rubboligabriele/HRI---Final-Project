import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load labeled feature data
X = np.load("labeled_features.npy")
y = np.load("labels.npy")

# Sanity check
print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")
print(f"Classes: {set(y)}")

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the SVM classifier
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(model, "svm_cognitive_state.joblib")
print("\n✅ Model saved as 'svm_cognitive_state.joblib'")