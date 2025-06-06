import datetime
import json

# File paths for logs
log_txt_path = "conversation_log.txt"
log_jsonl_path = "conversation_log.jsonl"

def log_conversation(state, user_input, response, user_id, turn_id):
    """
    Logs each conversation turn in both plain text and JSONL formats.
    
    Args:
        state (str): Detected cognitive state (e.g., 'attentive').
        user_input (str): The user's input message.
        response (str): The assistant's generated reply.
        user_id (str): Identifier for the current user.
        turn_id (int): Turn number in the conversation.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write to human-readable TXT log
    with open(log_txt_path, "a", encoding="utf-8") as f_txt:
        f_txt.write(f"[{timestamp}] Turn {turn_id} — User: {user_id} — State: {state.upper()}\n")
        f_txt.write(f"User: {user_input}\n")
        f_txt.write(f"Assistant: {response}\n\n")

    # Write structured JSONL log (one JSON object per line)
    with open(log_jsonl_path, "a", encoding="utf-8") as f_jsonl:
        json.dump({
            "timestamp": timestamp,
            "turn_id": turn_id,
            "user_id": user_id,
            "state": state,
            "user_input": user_input,
            "assistant_response": response
        }, f_jsonl)
        f_jsonl.write("\n")