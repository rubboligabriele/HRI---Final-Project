import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create Gemini model instance
model = genai.GenerativeModel('models/gemini-1.5-flash')

# Initial system prompt to instruct the assistant's personality and communication style
system_prompt = (
    "You are a friendly, patient, and supportive AI assistant designed "
    "to help elderly users feel comfortable, understood, and engaged. "
    "You always use slow, clear, and simple language, speak with warmth and kindness, "
    "and adapt your tone to the user's cognitive state if given. "
    "Always close your messages with a gentle question, like: 'Shall we continue?' or 'Is that okay?'"
)

# Memory of previous turns
conversation_history = [ { "role": "user", "parts": [system_prompt] } ]


def generate_response(state, style, example, user_input=None):
    """
    Generates a response from Gemini, adapting to the user's cognitive state and interaction style.
    """
    # Compose the user input based on state and preference
    user_prompt = (
        f"The user appears {state}. Their preferred response style is: {style}.\n"
        f"Example response: \"{example}\"."
    )
    if user_input:
        user_prompt += f"\nThey just said: \"{user_input}\""

    # Add to conversation
    conversation_history.append({ "role": "user", "parts": [user_prompt] })

    # Generate response
    response = model.generate_content(conversation_history)

    # Extract response
    reply = response.text.strip()

    # Add assistant's response to memory
    conversation_history.append({ "role": "model", "parts": [reply] })

    return reply