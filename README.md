# HRI---Final-Project: Adaptive Elderly Assistant

This project presents an adaptive Human-Robot Interaction (HRI) assistant designed to support elderly users by detecting their cognitive state via webcam and dynamically adjusting its conversational style accordingly. The system integrates computer vision, affective computing, and machine learning techniques to personalize interactions based on user attention, confusion, or distraction.

## 🎯 Project Goals

- Detect user cognitive states (attentive, confused, distracted) in real-time.
- Collect personalized conversation preferences based on the detected state.
- Adapt the assistant’s communication style automatically to match the user’s emotional and cognitive needs.
- Provide transparent feedback through visualizations and model interpretability.

---


## 🧠 Cognitive States

The system supports three user states:

- `attentive`
- `confused`
- `distracted`

Each state triggers a different conversation strategy depending on user preferences.

---

## 🖼️ Feature Extraction

From webcam frames, the following visual features are extracted:

- Eye openness (left/right)
- Eyebrow distance (left/right)
- Mouth openness
- Head yaw asymmetry
- Head vertical tilt
- Shoulder distance

These features are used to train a Support Vector Machine (SVM) classifier to distinguish user states.

---

## 🧪 Model Training & Evaluation

- **Classifier**: Support Vector Machine (RBF Kernel)
- **Hyperparameter tuning**: Grid Search with Stratified 5-Fold Cross-Validation
- **Performance metrics**: Accuracy, confusion matrix, permutation-based feature importance
- **Dimensionality reduction**: PCA for visualization of decision boundaries

Model is saved to: `svm_cognitive_state.joblib`

---

## 🧍 Real-time Assistant (run_assistant.py)

The assistant performs the following loop:

1. Observes user through webcam and classifies cognitive state.
2. Loads conversation preferences for that state.
3. Displays a styled response and asks for user input.
4. Observes the post-response reaction.
5. If the state worsens, prompts the user to change communication style.
6. Logs each interaction with state, message, and response.

---

## 📈 Visual Analytics

- PCA decision boundaries with support vectors
- Permutation feature importance chart
- Confusion matrix heatmap (saved as PDF)

---

## 🔧 Installation

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

This project includes the following main scripts:

---

### 🟡 1. Data Collection

Collect labeled facial data from webcam for each cognitive state:

```bash
ipython src/scripts/collect_data.py --
```

While the script is running and you are in the webcam:

- Press `A` for Attentive  
- Press `C` for Confused  
- Press `D` for Distracted  
- Press `Q` to stop  

Collected data is saved to:

- `dataset/labeled_features.npy`  
- `dataset/labels.npy`  
- `dataset/dataset.csv`

---

### 🟢 2. Model Training

Train an SVM model with hyperparameter tuning on the collected dataset:

```bash
ipython src/scripts/train_classifier.py --
```

This script performs:

- Train/test split (stratified)
- Grid search with cross-validation
- Best model saving as: `svm_cognitive_state.joblib`

---

### 🔵 3. Visual Analysis

To generate PCA visualization, feature importance, and confusion matrix:

```bash
ipython src/scripts/evaluate.py --
```

Outputs:

- PCA decision boundaries (with support vectors)
- Permutation feature importance chart
- Confusion matrix

---

### 🟣 4. Run the Adaptive Assistant

Start the interactive webcam-based assistant:

```bash
ipython src/scripts/main.py --
```

Features:

- Detects cognitive state in real-time
- Load preferences from memory
- Generates personalized responses
- Observes post-response changes
- Update user style preferences
- Logs full conversation history

---

## 📓 Logs and Style Learning

User interactions and style preferences are saved across sessions. The assistant adapts based on how the user's state evolves after each message, and asks whether to update communication style if engagement decreases.

---

## 🧑‍💻 Authors

Developed as a final project for the Human-Robot Interaction course, by Gabriele Rubboli Petroselli and Maaike Looijenga.
