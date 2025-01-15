from pathlib import Path
import os
from typing import List, Optional, Dict
from smolagents import CodeAgent, HfApiModel, Tool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from smolagents import CodeAgent, HfApiModel, Tool
from .tools import *
from typing import Any, List

class {agent_name}(CodeAgent):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        **kwargs
    ):
        # Initialize tools
        tools = [
            {', '.join(f"{t['name'].title()}Tool()" for t in tool_specs)}
        ]
        
        # Initialize model
        model = HfApiModel(model_id=model_id)
        
        super().__init__(
            tools=tools,
            model=model,
            **kwargs
        )
    
    def process_request(self, request: str) -> str:
        \"\"\"Process a user request using CodeAgent's run method.\"\"\"
        return self.run(request)
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
    description = "Generates tool specifications based on agent requirements"
    inputs = {
        "description": {
            "type": "string",
            "description": "Description of the agent's requirements"
        }
    }
    output_type = "object"

    def forward(self, description: str) -> List[Dict]:
        """Generate tool specifications based on description."""
        logger.info(f"Starting tool generation for description: {description}")
        
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
                "content": f"Create tools for an agent that: {description}"
            }
        ]
        
        try:
            response = self.model(messages)
            tool_specs = eval(response.content)
            return tool_specs
        except Exception as e:
            logger.error(f"Failed to generate tools: {e}")
            return self._get_fallback_tool()

    def _get_fallback_tool(self) -> List[Dict]:
        """Get a fallback tool if generation fails."""
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
        
        # Create tools
        agent_creator = AgentCreationTool(model=model)
        tool_generator = ToolGenerationTool(model=model)
        
        # Set up tool reference
        agent_creator.tool_generator = tool_generator
        
        # Initialize CodeAgent with tools
        super().__init__(
            tools=[agent_creator, tool_generator],
            model=model,
            **kwargs
        )
        self.output_dir = output_dir or os.getcwd()

    def create_agent(self, description: str, save_path: str = None) -> str:
        """Create a new agent using CodeAgent's run method."""
        if save_path is None:
            save_path = self.output_dir
            
        # First, get the tools using tool_generator
        tool_specs = self.run(f"""Use tool_generator to create tools for this agent: {description}
        Return only the tool specifications.""")
        
        # Then create the agent with those tools
        agent_path = self.run(f"""Use agent_creator to create an agent with these requirements:
        Description: {description}
        Save path: {save_path}""")
        
        return agent_path 