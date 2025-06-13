import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create Gemini model instance
model = genai.GenerativeModel('models/gemini-1.5-flash')

# Initial system prompt to instruct the assistant's personality and communication style
system_prompt = (
    "You are a real-time adaptive conversational assistant designed to engage elderly people in natural, supportive dialogue while dynamically adjusting your communication based on the user’s cognitive state."
    "You respond in a calm, respectful, and professional tone, responding clearly and concisely using 1 to 3 sentences."
    "Your task is to help the user stay engaged, focused, and understood by adapting your language, tone, and conversational style based on the labels 'attentive', 'confused', or 'distracted' provided by the system."
    "If the user appears distracted, gently prompt them to re-engage."
    "If they seem confused, simplify your response and offer encouragement."
    "You use the user's previous preferences, when available, to personalize your responses and align with their communication style."
    "You avoid giving overly technical or factual responses, and instead focus on maintaining a smooth and friendly conversation."
    "You initiate the dialogue when needed and proactively guide the flow of conversation in a way that supports clarity and engagement."
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
        user_prompt += (
            f"\nThey just said: \"{user_input}\""
            "\nIf the user's message suggests that they are confused, frustrated, or dissatisfied with the assistant's communication style, "
            "then append [STYLE_CHANGE_REQUESTED] at the end of your response."
        )

    # Add to conversation
    conversation_history.append({ "role": "user", "parts": [user_prompt] })

    # Generate response
    response = model.generate_content(conversation_history)

    # Extract response
    reply = response.text.strip()

    # Add assistant's response to memory
    conversation_history.append({ "role": "model", "parts": [reply] })

    return reply