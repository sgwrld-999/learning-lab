# standards imports
from os import getenv
from datetime import datetime
from typing import Optional

# third party imports
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.tools import google_search


# tools 
def get_time() -> dict:
    """Get Time

    Args:
        

    Returns:
        str: current time in the specified location
    """
    return {
        "time": datetime.now().strftime("%H:%M:%S")
    }
    

def get_mean(numbers: list[float]) -> float:
    """Calculate the mean of a list of numbers.

    Args:
        numbers (list[float]): A list of numerical values.

    Returns:
        float: The mean of the provided numbers.
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


# Define system instructions in a separate, readable variable.
TOOL_AGENT_PROMPT = """
You are a tool agent. Your primary goal is to perform the actions with the help attached tools to complete the user query, learn their required things which is need to perform the certain actions, and deliver the results.

Follow these steps precisely:
1.  Start with a greeting the user and ask the user for what they want
2.  Once the user provides their requirements, using the tools perform the following action.
3.  After finishing done completion, validate your answer is this correct or not
4.  If you cannot perform the action as per the user request, inform the user, you are unable to do, because you don't have access to the tools.
"""

# --- Agent Definition ---

root_agent = Agent(
    name="tool_agent",
    model=getenv("ADK_MODEL", "gemini-2.0-flash"), # Updated model for compatibility
    description="Solve the user problem with the tools you have",   
    instruction=TOOL_AGENT_PROMPT,
    # tools=[google_search],  # List of tools the agent can use which are built-in tools
    tools=[get_time, get_mean] # custom tools can be added here (e.g., web_browser, python_executor
)


