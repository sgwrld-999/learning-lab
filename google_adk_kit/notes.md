# Notes for Google ADK Kit

## Basic

----

### Required Agent structure

    For ADK to discover and run your agents properly (especially with adk web), your project must follow a specific structure: 
```
    parent_folder/
    ├── agent_folder/
        __init__.py # this is the agent's package
        agent.py # this is the agent's main file # Must define the root_agent variable
        .env # this is the environment file (optional)
```
    - `parent_folder` can be any name you want, it is the folder that contains your agent's package.
    - `agent_folder` is the name of your agent's package, it must be a valid Python package name (no spaces, no special characters, etc.).
    - `__init__.py` can be an empty file, it just needs to be present to make Python treat the directory as a package.
    - `agent.py` is the main file of your agent, it must define a variable named `root_agent` which is an instance of `Agent`.
    - `.env` is an optional file where you can define environment variables for your agent.

----

### Key Components

    #### 1. Identity (`name` and `description`)
    - **`name` (Required):** A unique string identifier for your agent.
    - **`description` (Optional, but recommended):** A concise summary of the agent's capabilities, used by other agents to determine if they should route a task to this agent.

    #### 2. Model (`model`)
    - Specifies which LLM powers the agent (e.g., `"gemini-2.0-flash"`).
    - Affects the agent's capabilities, cost, and performance.

    #### 3. Instructions (`instruction`)
    The most critical parameter for shaping your agent's behavior. It defines:
    - Core task or goal
    - Personality or persona
    - Behavioral constraints
    - How to use available tools
    - Desired output format

    #### 4. Tools (`tools`)
    Optional capabilities beyond the LLM's built-in knowledge, allowing the agent to:
    - Interact with external systems
    - Perform calculations
    - Fetch real-time data
    - Execute specific actions

----

### Setting up the virtual environment for macOS

1. **Create a Virtual Environment:**
   Open your terminal and navigate to your project directory. Then run:
   ```bash
   python3 -m venv venv
   ```
    This command creates a new directory named `venv` that contains the virtual environment.

2. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

