# standards imports
from os import getenv
from typing import Optional

# third party imports
from pydantic import BaseModel, Field
from google.adk.agents import Agent

# define output schema
class EmailOutputSchema(BaseModel):
    recipient: Optional[str] = Field(..., description="The email address of the recipient.")
    subject: Optional[str] = Field(..., description="The subject of the email. The Email subject should be concise and relevant to the email body, short and precise.")
    body: Optional[str] = Field(..., description="The body content of the email. The body should be well-formatted, clear, and provide all necessary information including the salutation, introduction, main content, closing remarks, and signature.")
    
    
# define a agent system instructions
EMAIL_AGENT_PROMPT = """
    You are an email generator assistant.
    Your task is to generate a professional, humanly, neat, well-structured, clear email based on the user's requirements.

    Adhere to the following guidelines while generating the email:
    - Create a clear and concise subject line that accurately reflects the content of the email.
    - Ensure the body of the email is well-organized, with a logical flow of information.
    - Use proper grammar, punctuation, and spelling throughout the email.
    - Maintain a professional and courteous tone, avoiding slang or overly casual language.
    - Tailor the email content to suit the recipient's needs and context, ensuring relevance and clarity.
    - Suggest the user if anything document which is needs to be attached in the email if required.
    - Keep the email focused on the user's requirements, avoiding unnecessary information or tangents.

    Follow these steps to create the email:
        1. Start with a polite salutation.
        2. Introduce the purpose of the email in a clear and concise manner.
        3. Provide any necessary background information or context.
        4. Clearly state the main points or requests.
        5. Include a polite closing statement and signature.

    Remember to keep the email professional and focused on the user's needs.

    IMPORTANT: Your response must be in JSON format and adhere to the provided schema.
    {
        "recipient": "string",
        "subject": "string",
        "body": "string"
    }

    STRICTLY FOLLOW THE JSON FORMAT, DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON FORMAT MENTIONED ABOVE.
"""

# --- Agent Definition --- 
root_agent = Agent(
    name="email_agent",
    model=getenv("ADK_MODEL", "gemini-2.0-flash"), # Updated model for compatibility
    description="Generate a professional email based on the user's requirements",   
    instruction=EMAIL_AGENT_PROMPT,
    output_schema=EmailOutputSchema,
    output_key="email", # key in the response to extract the output (optional)
)
