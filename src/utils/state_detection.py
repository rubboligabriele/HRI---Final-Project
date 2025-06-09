import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe modules
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def extract_features(frame):
    """
    Extracts normalized facial and body posture features from a webcam frame using MediaPipe.

    Returns:
        - features: np.array of 8 values
        - feature_names: list of 8 corresponding feature names

    Features:
        - Left eye openness
        - Right eye openness
        - Left eyebrow raise
        - Right eyebrow raise
        - Mouth openness
        - Head yaw asymmetry
        - Shoulder distance
        - Head tilt (vertical)
    """
    with mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True) as face_mesh, \
         mp_pose.Pose(static_image_mode=False) as pose:

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        features = []
        feature_names = []

        face_scale = None  # Used for normalization based on face size

        # --- Facial features ---
        if face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0]
            get = lambda i: np.array([face.landmark[i].x, face.landmark[i].y])

            # Estimate face height as reference scale (to normalize distances)
            top_face = get(10)     # forehead center
            bottom_face = get(152) # chin
            face_scale = np.linalg.norm(top_face - bottom_face)
            face_scale = max(face_scale, 1e-6)  # Prevent division by zero

            # Eye openness
            left_eye = np.linalg.norm(get(159) - get(145)) / face_scale
            right_eye = np.linalg.norm(get(386) - get(374)) / face_scale
            features += [left_eye, right_eye]
            feature_names += ["left_eye_openness", "right_eye_openness"]

            # Eyebrow raise
            left_eyebrow = np.linalg.norm(get(105) - get(159)) / face_scale
            right_eyebrow = np.linalg.norm(get(334) - get(386)) / face_scale
            features += [left_eyebrow, right_eyebrow]
            feature_names += ["left_eyebrow_raise", "right_eyebrow_raise"]

            # Mouth openness
            mouth_open = np.linalg.norm(get(13) - get(14)) / face_scale
            features.append(mouth_open)
            feature_names.append("mouth_openness")

            # Head yaw asymmetry
            nose = get(1)
            left_cheek = get(234)
            right_cheek = get(454)
            yaw = (np.linalg.norm(nose - left_cheek) - np.linalg.norm(nose - right_cheek)) / face_scale
            features.append(yaw)
            feature_names.append("head_yaw_asymmetry")
        else:
            features += [0] * 6
            feature_names += [
                "left_eye_openness", "right_eye_openness",
                "left_eyebrow_raise", "right_eyebrow_raise",
                "mouth_openness", "head_yaw_asymmetry"
            ]

        # --- Body posture features ---
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y])

            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            head_tilt = landmarks[7].y - landmarks[8].y

            features += [shoulder_dist, head_tilt]
            feature_names += ["shoulder_distance", "head_tilt"]
        else:
            features += [0, 0]
            feature_names += ["shoulder_distance", "head_tilt"]

        return np.array(features), feature_names