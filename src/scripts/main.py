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
        print(f" Observing your reaction in {i}...")
        time.sleep(1)

    ret, frame = cap.read()
    if not ret:
        break

    # Extract features from current frame
    features, feature_names = extract_features(frame)
    try:
        state = model.predict([features])[0]
    except ValueError as e:
        print("\n Prediction failed due to shape mismatch.")
        print(f"Expected {model.n_features_in_} features but got {len(features)}.\n")
        print(" Feature breakdown:")
        for name, value in zip(feature_names, features):
            print(f" - {name}: {value:.4f}")
        raise e

    style, example = load_preference(user_id, state)

    print(f"\nDetected state: [{state.upper()}] (style: {style})")

    user_input = input("You: ").strip()

    response = generate_response(state, style, example, user_input=user_input)

    style_requested = "[STYLE_CHANGE_REQUESTED]" in response
    response = response.replace("[STYLE_CHANGE_REQUESTED]", "")

    print(f"Assistant: {response}")

    log_conversation(state, user_input, response, user_id, turn_id)
    turn_id += 1

    if not style_requested:
        for i in range(10, 0, -1):
            print(f" Observing your reaction in {i}...")
            time.sleep(1)

        ret, post_frame = cap.read()
        post_features, feature_names = extract_features(post_frame)
        try:
            post_state = model.predict([post_features])[0]
        except ValueError as e:
            print("\n Post-response prediction failed.")
            print(f"Expected {model.n_features_in_} features but got {len(post_features)}.\n")
            print(" Feature breakdown:")
            for name, value in zip(feature_names, post_features):
                print(f" - {name}: {value:.4f}")
            raise e

        print(f" Observed post-response state: [{post_state.upper()}]")

    if response_did_not_work(state, post_state) or style_requested:
        assistant_question = f"You seem {post_state}. Would you like to try a different explanation style for when you are {state}?"
        print(f"\n Assistant: {assistant_question}")
        log_conversation(post_state, "System trigger", assistant_question, user_id, turn_id)
        turn_id += 1

        confirm = input("Change style? (yes/no): ").strip().lower()
        log_conversation(post_state, "User", f"Change style? → {confirm}", user_id, turn_id)
        turn_id += 1

        if confirm == "yes":
            new_style = input(f"Enter a new preferred style for when you are {state} (e.g., calm, motivational, step-by-step): ").strip()
            new_example = input("Give an example response in that style: ").strip()
            update_preference(user_id, state, new_style, new_example)
            print("Preferences updated.")

            style_change_input = f"[User updated preferred style to '{new_style}' for state '{state}']"
            assistant_ack = "Preferences updated and stored. I'll use this style from now on when you seem like that."
            log_conversation(state, style_change_input, assistant_ack, user_id, turn_id)
            turn_id += 1
        else:
            new_style, new_example = style, example

        response = generate_response(post_state, new_style, new_example, user_input="Continue the conversation.")
        print(f"Assistant: {response}")
        log_conversation(post_state, "System trigger after style check", response, user_id, turn_id)
        turn_id += 1

        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
