import cv2
import joblib
import time
from src.utils.state_detection import extract_features
from src.utils.memory import load_preference, update_preference
from src.utils.response_generator import generate_response
from src.utils.conversation_saving import log_conversation

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
    if before == "attentive" and after in ["confused", "distracted"]:
        return True
    elif before != after and after in ["confused", "distracted"]:
        return True
    elif before == after and before in ["confused", "distracted"]:
        return True
    else:
        return False

# Initial assistant greeting (neutral state)
initial_response = generate_response(
    state="attentive",
    style="empathetic",
    example="Hello, how are you feeling today?",
    user_input="Please greet the user and ask if they would like to chat or do something together."
)
print(f"\nAssistant: {initial_response}")

while True:
    for i in range(3, 0, -1):
        print(f"⏳ Observing your reaction in {i}...")
        time.sleep(1)
    
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

    print(f"\nDetected state: [{state.upper()}] (style: {style})")

    # Wait for user input
    user_input = input("You: ").strip()

    # Generate assistant response adapted to user state and preference
    response = generate_response(state, style, example, user_input=user_input)
    print(f"Assistant: {response}")

    # Log the interaction
    log_conversation(state, user_input, response, user_id, turn_id)
    turn_id += 1

    # Countdown before observing user's reaction
    for i in range(3, 0, -1):
        print(f"⏳ Observing your reaction in {i}...")
        time.sleep(1)

    ret, post_frame = cap.read()
    post_features = extract_features(post_frame)
    post_state = model.predict([post_features])[0]

    print(f"📸 Observed post-response state: [{post_state.upper()}]")

    # If the state got worse or did not improve, offer to change style
    if response_did_not_work(state, post_state):
        print(f"\n🤖 Assistant: You seem {post_state}.")
        print("Would you like to try a different explanation style?")
        confirm = input("Change style? (yes/no): ").strip().lower()

        if confirm == "yes":
            new_style = input(f"Enter a new preferred style for when you are {state} (e.g., calm, motivational, step-by-step): ").strip()
            new_example = input("Give an example response in that style: ").strip()
            update_preference(user_id, state, new_style, new_example)
            print("Preferences updated.")
        else:
            new_style, new_example = style, example

        response = generate_response(post_state, new_style, new_example, user_input="Continue the conversation.")
        print(f"Assistant: {response}")
        log_conversation(post_state, "System trigger after style check", response, user_id, turn_id)
        turn_id += 1

        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()