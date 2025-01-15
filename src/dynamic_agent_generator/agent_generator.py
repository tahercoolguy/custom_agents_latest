from pathlib import Path
import os
from typing import List, Optional, Dict
from smolagents import CodeAgent, HfApiModel, Tool

class AgentCreationTool(Tool):
    """Tool for creating new agents."""
    
    name = "agent_creator"
    description = "Creates a new agent based on specifications"
    inputs = {
        "description": {
            "type": "string",
            "description": "Natural language description of the agent to create"
        },
        "save_path": {
            "type": "string",
            "description": "Directory where the agent should be saved"
        },
        "agent_name": {
            "type": "string",
            "description": "Optional name for the agent",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model=None, **kwargs):
        self.model = model  # Set model before super().__init__
        super().__init__(**kwargs)
        self.tool_generator = None  # Will be set by DynamicAgentGenerator

    def _create_agent(self, description: str, save_path: str, agent_name: Optional[str] = None) -> str:
        # Generate agent name if not provided
        if agent_name is None:
            messages = [
                {"role": "system", "content": "Generate a camelCase name for an agent."},
                {"role": "user", "content": f"Agent description: {description}"}
            ]
            agent_name = self.model(messages).content.strip()
            if not agent_name.isidentifier():
                agent_name = "CustomAgent"

        # Create directory structure
        agent_dir = Path(save_path) / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir(exist_ok=True)

        # Generate tools with implementations
        tool_specs = self.tool_generator(description=description)
        
        # Create tool files
        tool_paths = []
        for tool_spec in tool_specs:
            tool_code = f"""
from smolagents import Tool
from typing import Any, Dict

class {tool_spec['name'].title()}Tool(Tool):
    \"\"\"
    {tool_spec['description']}
    \"\"\"
    
    name = "{tool_spec['name']}"
    description = "{tool_spec['description']}"
    inputs = {tool_spec['inputs']}
    output_type = "{tool_spec['output_type']}"
    
    def forward(self, **kwargs) -> Any:
        {tool_spec['implementation']}
"""
            
            tool_path = tools_dir / f"{tool_spec['name']}.py"
            tool_paths.append(tool_path)
            
            with open(tool_path, "w") as f:
                f.write(tool_code)

        # Create tools/__init__.py
        with open(tools_dir / "__init__.py", "w") as f:
            for tool_spec in tool_specs:
                f.write(f"from .{tool_spec['name']} import {tool_spec['name'].title()}Tool\n")

        # Create agent.py
        agent_code = f"""
from smolagents import CodeAgent, HfApiModel
from .tools import *
from typing import Any, List

class {agent_name}(CodeAgent):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct", **kwargs):
        tools = [
            {', '.join(f"{t['name'].title()}Tool()" for t in tool_specs)}
        ]
        super().__init__(
            tools=tools,
            model=HfApiModel(model_id=model_id),
            description=self.__doc__,
            **kwargs
        )

    def run(self, query: str, **kwargs) -> str:
        \"\"\"Process the user query using available tools.\"\"\"
        messages = [
            {{"role": "system", "content": self.description}},
            {{"role": "user", "content": query}}
        ]
        response = self.model(messages)
        return response.content if hasattr(response, 'content') else str(response)
"""
        
        with open(agent_dir / "agent.py", "w") as f:
            f.write(agent_code)

        # Create main __init__.py
        with open(agent_dir / "__init__.py", "w") as f:
            f.write(f"from .agent import {agent_name}\n")
            f.write("from .tools import *\n")

        # Create requirements.txt
        with open(agent_dir / "requirements.txt", "w") as f:
            f.write("smolagents>=1.2.2\n")

        return str(agent_dir)

    def forward(self, description: str, save_path: str, agent_name: Optional[str] = None) -> str:
        return self._create_agent(description, save_path, agent_name)

class ToolGenerationTool(Tool):
    """Tool for generating agent tools."""
    
    name = "tool_generator"
    description = "Generates tools for an agent based on requirements"
    inputs = {
        "description": {
            "type": "string",
            "description": "Description of the agent's requirements"
        }
    }
    output_type = "object"

    def __init__(self, model=None, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def forward(self, description: str) -> List[Dict]:
        """Generate tool specifications based on description."""
        messages = [
            {
                "role": "system", 
                "content": """You are an AI that creates tool specifications for agents.
                You must respond with only a Python list of dictionaries.
                Each dictionary must have these exact keys: 'name', 'description', 'inputs', 'output_type', and 'implementation'.
                The response must be valid Python code that can be evaluated with eval().
                Do not include any explanatory text, just the Python list."""
            },
            {
                "role": "user",
                "content": f"""Create tools for an agent that: {description}

Example format:
[
    {{
        "name": "example_tool",
        "description": "What this tool does",
        "inputs": {{"input_name": {{"type": "string", "description": "what this input does"}}}},
        "output_type": "string",
        "implementation": "# Python code here\\nreturn processed_result"
    }}
]"""
            }
        ]
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                tool_specs = eval(response.content)
                
                # Validate response format
                if not isinstance(tool_specs, list):
                    raise ValueError("Response must be a list")
                
                required_keys = {'name', 'description', 'inputs', 'output_type', 'implementation'}
                for tool in tool_specs:
                    if not isinstance(tool, dict):
                        raise ValueError("Each tool must be a dictionary")
                    if not all(key in tool for key in required_keys):
                        raise ValueError(f"Tool missing required keys. Must have: {required_keys}")
                
                return tool_specs
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Add error context to the next attempt
                    messages.append({
                        "role": "user",
                        "content": f"""Previous attempt failed with error: {str(e)}
                        Please try again and ensure you return a valid Python list of tool specifications.
                        Remember to only return the Python list, no other text."""
                    })
                continue
        
        print(f"Failed to generate tools after {max_retries} attempts. Last error: {last_error}")
        # Fallback to basic calculator tool if all retries fail
        return [{
            "name": "calculator",
            "description": "Performs basic mathematical calculations",
            "inputs": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "output_type": "string",
            "implementation": """
                try:
                    # Safely evaluate the mathematical expression
                    result = eval(kwargs['expression'], {"__builtins__": {}}, {})
                    return str(result)
                except Exception as e:
                    return f"Error: {str(e)}"
            """
        }]

class DynamicAgentGenerator(CodeAgent):
    """An agent that generates new smolagents agents based on user requirements."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        output_dir: Optional[str] = None,
        **kwargs
    ):
        # Create model instance
        model = HfApiModel(model_id=model_id)
        
        # Initialize tools with the model
        self.agent_creator = AgentCreationTool(model=model)
        self.tool_generator = ToolGenerationTool(model=model)
        
        # Set up tool references
        self.agent_creator.tool_generator = self.tool_generator
        
        super().__init__(
            tools=[self.agent_creator, self.tool_generator],
            model=model,
            **kwargs
        )
        self.output_dir = output_dir or os.getcwd()

    def create_agent(
        self,
        description: str,
        save_path: str,
        agent_name: Optional[str] = None
    ) -> str:
        """Create a new agent using the tools."""
        # First, generate the tools needed for the agent
        tool_specs = self.tool_generator(description=description)
        
        # Then create the agent with the generated tools
        agent_path = self.agent_creator(
            description=description,
            save_path=save_path,
            agent_name=agent_name
        )
        
        return agent_path

    def run(self, **kwargs) -> str:
        """Run the agent generator with the given inputs."""
        return self.create_agent(**kwargs) 