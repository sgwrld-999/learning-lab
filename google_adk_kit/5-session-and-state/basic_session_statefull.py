# This example demonstrates how to create a stateful session using the InMemorySessionService.

# standards imports
import uuid

# third party imports
from dotenv import load_dotenv # Since we are not using the ADK CLI, we need to load the .env file manually 
from google.adk.runners import Runner # Runner to execute the agent with session support 
from google.adk.sessions import InMemorySessionService # In-memory session service to store session state   
from google.genai import types # For creating message content
from email_agent_generator import root_agent as question_answering_agent # Import the agent you want to run

load_dotenv() 


# Create a new session service to store state
session_service_stateful = InMemorySessionService()

# This will give the initial state for the session, which basically the initial context for the agent, for performing the actions.
initial_state = {
    "user_name": "Siddhant Gond",
    "user_preferences": """
        - Favorite TV Show: Breaking Bad
        - Favorite Movie: Inception
        - Favorite Music Genre: Rock
        - Hobbies: Hiking, Photography, Cooking
        - Profession: AI Engineer
        - Work Experience: 5 years in software development and AI research
        - Skills: Python, Machine Learning, Data Analysis, Cloud Computing
        - Education: Master's degree in Computer Science
        - Time Zone: PST (UTC-8)
        - Location: Germany, Europe
        - Preferred Contact Method: Email
    """,
}

# Create a NEW session 
APP_NAME = "Siddhant Gond"
USER_ID = "brandon_hancock"
SESSION_ID = str(uuid.uuid4())

# Create a new session with the initial state and unique ID
stateful_session = session_service_stateful.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state,
)


print("CREATED NEW SESSION:")
print(f"\tSession ID: {SESSION_ID}")

# Create a runner to execute the agent with session support, runner will manage the session state automatically which contains two entries: Agents and session state.
runner = Runner(
    agent=question_answering_agent,
    app_name=APP_NAME,
    session_service=session_service_stateful,
)

# Simulate a conversation with the agent
print("==== Conversation Start ====")
new_message = types.Content(
    role="user", parts=[types.Part(text="Write a professional email to my manager about my work experience and skills.")],
)

# Run the agent with the new message and session details
for event in runner.run(
    user_id=USER_ID,
    session_id=SESSION_ID,
    new_message=new_message,
):
    
    if event.is_final_response():
        if event.content and event.content.parts:
            print(f"Final Response: {event.content.parts[0].text}")

print("==== Session Event Exploration ====")
session = session_service_stateful.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
)

# Log final Session state
print("=== Final Session State ===")
for key, value in session.state.items():
    print(f"{key}: {value}")