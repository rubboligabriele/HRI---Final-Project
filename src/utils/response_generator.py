import openai
import os

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initial system prompt to instruct the assistant's personality and communication style
system_prompt = {
    "role": "system",
    "content": (
        "You are a friendly, patient, and supportive AI assistant designed "
        "to help elderly users feel comfortable, understood, and engaged. "
        "You always use slow, clear, and simple language, speak with warmth and kindness, "
        "and adapt your tone to the user's cognitive state if given. "
        "Always close your messages with a gentle question, like: 'Shall we continue?' or 'Is that okay?'"
    )
}

# Conversation memory for maintaining context between turns
conversation_history = []

def generate_response(state, style, example, user_input=None):
    """
    Generates a response from the AI assistant, adapting to the user's cognitive state
    and preferred interaction style.

    Args:
        state (str): Detected cognitive state (e.g., 'confused', 'attentive').
        style (str): Preferred response style for this user in the given state.
        example (str): Example of a response the user reacted well to in the past.
        user_input (str, optional): Latest user message (if any).

    Returns:
        str: Assistant-generated reply.
    """

    # Context message describing the user's current state and preference
    context_msg = {
        "role": "user",
        "content": (
            f"The user appears {state}. Their preferred response style is: {style}.\n"
            f"Example response: \"{example}\"."
        )
    }

    # If a user message is provided, append it to the context
    if user_input:
        context_msg["content"] += f"\nThey just said: \"{user_input}\""

    # Build full conversation prompt
    messages = [system_prompt] + conversation_history + [context_msg]

    # Send request to the language model
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the reply from the response
    reply = completion.choices[0].message.content.strip()

    # Update conversation memory
    conversation_history.append(context_msg)
    conversation_history.append({"role": "assistant", "content": reply})

    return reply