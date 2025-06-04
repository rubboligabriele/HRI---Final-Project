import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe modules
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def extract_features(frame):
    """
    Extracts normalized facial and body posture features from a webcam frame using MediaPipe.

    Returns an 8-dimensional vector containing:
        - Left eye openness
        - Right eye openness
        - Left eyebrow raise
        - Right eyebrow raise
        - Mouth openness
        - Head yaw asymmetry
        - Shoulder distance
        - Head tilt (vertical)
    If no face or pose is detected, default values of 0 are returned for the missing features.
    """
    with mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True) as face_mesh, \
         mp_pose.Pose(static_image_mode=False) as pose:

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        features = []
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

            # Eye openness (vertical gap)
            left_eye = np.linalg.norm(get(159) - get(145)) / face_scale
            right_eye = np.linalg.norm(get(386) - get(374)) / face_scale
            features += [left_eye, right_eye]

            # Eyebrow raise (distance from eyebrow to eye)
            left_eyebrow = np.linalg.norm(get(105) - get(159)) / face_scale
            right_eyebrow = np.linalg.norm(get(334) - get(386)) / face_scale
            features += [left_eyebrow, right_eyebrow]

            # Mouth openness (vertical gap between lips)
            mouth_open = np.linalg.norm(get(13) - get(14)) / face_scale
            features.append(mouth_open)

            # Head yaw asymmetry (difference in distance from nose to cheeks)
            nose = get(1)
            left_cheek = get(234)
            right_cheek = get(454)
            yaw = (np.linalg.norm(nose - left_cheek) - np.linalg.norm(nose - right_cheek)) / face_scale
            features.append(yaw)
        else:
            # If face not detected, fill with zeros
            features += [0] * 7

        # --- Body posture features ---
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y])

            # Shoulder distance (used as rough proxy for body alignment/distance to camera)
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

            # Head tilt (difference in height between left and right ear)
            head_tilt = landmarks[7].y - landmarks[8].y

            features += [shoulder_dist, head_tilt]
        else:
            # If pose not detected, fill with zeros
            features += [0, 0]

        return np.array(features)