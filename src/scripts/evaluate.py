import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import joblib

# Load dataset
X = np.load("dataset_1/labeled_features.npy")
y = np.load("dataset_1/labels.npy")
feature_names = [
    "eye_openness_left", "eye_openness_right",
    "eyebrow_distance_left", "eyebrow_distance_right",
    "mouth_openness", "head_yaw_asymmetry",
    "shoulder_distance", "head_tilt_vertical"
]
colors = {'attentive': 'green', 'confused': 'orange', 'distracted': 'red'}

# Encode labels numerically (needed for contourf)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
label_names = encoder.classes_

# Load final trained model
model = joblib.load("svm_cognitive_state.joblib")

# ------------------ 1. PCA + Decision Boundary ------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
svm_pca = SVC(kernel="rbf")
svm_pca.fit(X_pca, y_encoded)

plt.figure(figsize=(8, 6))

# Plot data points
for i, label in enumerate(label_names):
    idxs = np.where(y_encoded == i)
    plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], label=label, alpha=0.6, color=colors[label])

# Compute decision boundaries
xx, yy = np.meshgrid(
    np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 500),
    np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 500)
)
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
cmap = ListedColormap([colors[l] for l in label_names])
plt.contourf(xx, yy, Z, alpha=0.2, levels=len(colors), cmap=cmap)

# Highlight support vectors
sv_idxs_pca = svm_pca.support_
plt.scatter(X_pca[sv_idxs_pca, 0], X_pca[sv_idxs_pca, 1],
            s=120, facecolors='none', edgecolors='black',
            linewidths=1.5, label="Support Vectors")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SVM Decision Regions with Support Vectors (PCA Space)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------ 2. Support Vector Info ------------------
print(f"\n🧷 Number of support vectors (final model): {len(model.support_)}")
print("Support vector indices:", model.support_)

# ------------------ 3. Permutation Feature Importance ------------------
# Train/test split for permutation importance
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
perm = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

plt.figure(figsize=(10, 5))
sorted_idx = perm.importances_mean.argsort()[::-1]
sns.barplot(
    x=perm.importances_mean[sorted_idx],
    y=np.array(feature_names)[sorted_idx],
    palette="coolwarm"
)
plt.title("Permutation Feature Importance (SVM)")
plt.xlabel("Mean Importance")
plt.tight_layout()
plt.show()