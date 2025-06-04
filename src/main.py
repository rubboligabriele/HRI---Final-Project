import cv2
import joblib
import time
from state_detection import extract_features
from memory import load_preference, update_preference
from response_generator import generate_response
from conversation_saving import log_conversation

# Load trained SVM model for cognitive state prediction
model = joblib.load("svm_cognitive_state.joblib")

# Initialize webcam
cap = cv2.VideoCapture(0)

user_id = "user_1"
turn_id = 1

print("\n--- Adaptive Elderly Assistant ---")
print("Press 'q' in the webcam window to quit.")
print("Type a message after each detection to interact.\n")

def response_did_not_work(before, after):
    """
    Heuristic to detect if the response failed to improve or maintain the user's state.
    """
    return before == after or (before == "attentive" and after in ["confused", "frustrated"])

# Initial assistant greeting (neutral state)
initial_response = generate_response(
    state="neutral",
    style="empathetic",
    example="Hello, how are you feeling today?",
    user_input="Please greet the user and ask if they would like to chat or do something together."
)
print(f"\nAssistant: {initial_response}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features from current frame
    features = extract_features(frame)

    # If no valid features, skip this frame
    if features is None or all(f == 0 for f in features):
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Predict user's cognitive state
    state = model.predict([features])[0]

    # Load the user's style preferences for the current state
    style, example = load_preference(user_id, state)

    # Display the webcam feed and detected state
    cv2.imshow("Webcam", frame)
    print(f"\nDetected state: [{state.upper()}] (style: {style})")

    # Wait for user input
    user_input = input("You: ").strip()

    # Generate assistant response adapted to user state and preference
    response = generate_response(state, style, example, user_input=user_input)
    print(f"Assistant: {response}")

    # Log the interaction
    log_conversation(state, user_input, response, user_id, turn_id)
    turn_id += 1

    # Pause and capture a post-response frame to observe reaction
    time.sleep(1.5)
    ret, post_frame = cap.read()
    post_features = extract_features(post_frame)
    post_state = model.predict([post_features])[0]

    # If the state got worse or did not improve, offer to change style
    if response_did_not_work(state, post_state):
        print(f"\n🤖 Assistant: You still seem {post_state}.")
        print("Would you like to try a different explanation style?")
        confirm = input("Change style? (yes/no): ").strip().lower()

        if confirm == "yes":
            new_style = input("Enter a new preferred style (e.g., calm, motivational, step-by-step): ").strip()
            new_example = input("Give an example response in that style: ").strip()
            update_preference(user_id, state, new_style, new_example)
            print("Preferences updated.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()