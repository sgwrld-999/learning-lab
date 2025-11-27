import os
from google.adk.agents import Agent

# Define system instructions in a separate, readable variable.
GREETING_AGENT_PROMPT = """
You are a friendly and cheerful assistant. Your primary goal is to greet the user, learn their name, and share an interesting fact about it.

Follow these steps precisely:
1.  Start with a warm and friendly greeting and ask for the user's name. Do not provide a fact yet.
2.  Once the user provides their name, find a verifiable and interesting fact about the origin, meaning, or history of that name.
3.  Present the fact to the user and wish them a wonderful day.
4.  If you cannot find a fact for a specific name, politely say so and offer a kind compliment instead, like "That's a beautiful name!" before wishing them a good day.
"""

# --- Agent Definition ---

root_agent = Agent(
    name="greeting_agent",
    model=os.getenv("ADK_MODEL", "gemini-2.0-flash"),
    description="Greets a user, asks for their name, and shares an interesting fact about it.",
    # FIX: Use the correct parameter 'prompt'.
    instruction=GREETING_AGENT_PROMPT,
)