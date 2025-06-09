import json

MEMORY_FILE = "user_memory.json"

def load_preference(user_id, state):
    """
    Loads the user's preferred response style and example for a given cognitive state.
    
    Args:
        user_id (str): The identifier of the user.
        state (str): The cognitive state (e.g., "confused", "attentive").
    
    Returns:
        tuple: (preferred_style, example_response)
    """
    with open(MEMORY_FILE, 'r') as f:
        memory = json.load(f)
    
    user_data = memory.get(user_id, {}).get(state, {})
    
    # Return default values if no preferences are stored
    return user_data.get("preferred_style", "neutral"), user_data.get("example_response", "Let's keep going!")

def update_preference(user_id, state, new_style, new_example):
    """
    Updates the user's preferred response style and example for a specific cognitive state.
    Creates entries if they don't already exist.
    
    Args:
        user_id (str): The identifier of the user.
        state (str): The cognitive state to update.
        new_style (str): New preferred communication style (e.g., "calm", "step-by-step").
        new_example (str): Example response in that style.
    """
    with open(MEMORY_FILE, 'r') as f:
        memory = json.load(f)

    # Ensure user section exists
    if user_id not in memory:
        memory[user_id] = {}

    # Update the preference for the given state
    memory[user_id][state] = {
        "preferred_style": new_style,
        "example_response": new_example,
    }

    # Save updated memory
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)