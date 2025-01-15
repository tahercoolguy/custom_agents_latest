from pathlib import Path
import os
from typing import List, Optional, Dict
from smolagents import CodeAgent, HfApiModel, Tool
from .utils.code_generator import generate_tool_code, generate_agent_code

class DynamicAgentGenerator(CodeAgent):
    """An agent that generates new smolagents agents based on user requirements."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the DynamicAgentGenerator.
        
        Args:
            model_id: The HuggingFace model ID to use for generation
            output_dir: Directory where generated agents will be saved
            **kwargs: Additional arguments passed to CodeAgent
        """
        super().__init__(
            tools=[],
            model=HfApiModel(model_id=model_id),
            **kwargs
        )
        self.output_dir = output_dir or os.getcwd()

    def _validate_tool_spec(self, tool_spec: Dict) -> None:
        """Validate tool specification."""
        required_fields = ["name", "description", "inputs", "output_type"]
        for field in required_fields:
            if field not in tool_spec:
                raise ValueError(f"Tool specification missing required field: {field}")
        
        if not isinstance(tool_spec["inputs"], dict):
            raise ValueError("Tool inputs must be a dictionary")

    def create_agent(
        self,
        agent_name: str,
        description: str,
        required_tools: List[Dict],
        save_path: str
    ) -> str:
        """Create a new agent with specified tools and capabilities.
        
        Args:
            agent_name: Name of the agent to create
            description: Description of what the agent should do
            required_tools: List of tool specifications
            save_path: Where to save the generated agent
            
        Returns:
            Path to the generated agent
        """
        if not agent_name.isidentifier():
            raise ValueError("Agent name must be a valid Python identifier")
        
        # Create output directory
        agent_dir = Path(save_path) / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tools directory
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        # Generate tools
        tool_paths = []
        for tool_spec in required_tools:
            self._validate_tool_spec(tool_spec)
            
            tool_code = generate_tool_code(
                tool_spec["name"],
                tool_spec["description"],
                tool_spec["inputs"],
                tool_spec["output_type"]
            )
            
            tool_path = tools_dir / f"{tool_spec['name']}.py"
            tool_paths.append(tool_path)
            
            with open(tool_path, "w") as f:
                f.write(tool_code)
        
        # Create tools/__init__.py
        with open(tools_dir / "__init__.py", "w") as f:
            for tool_spec in required_tools:
                f.write(f"from .{tool_spec['name']} import {tool_spec['name'].title()}Tool\n")
                
        # Generate agent code
        agent_code = generate_agent_code(
            agent_name,
            description,
            tool_paths
        )
        
        agent_path = agent_dir / "agent.py"
        with open(agent_path, "w") as f:
            f.write(agent_code)
            
        # Create main __init__.py
        with open(agent_dir / "__init__.py", "w") as f:
            f.write(f"from .agent import {agent_name}\n")
            f.write(f"from .tools import *\n")
            
        # Create requirements.txt
        with open(agent_dir / "requirements.txt", "w") as f:
            f.write("smolagents>=1.2.2\n")
            
        return str(agent_dir)

    def improve_agent(self, agent_path: str, feedback: str) -> str:
        """Improve an existing agent based on feedback.
        
        Args:
            agent_path: Path to the agent to improve
            feedback: Feedback on what to improve
            
        Returns:
            Path to the improved agent
        """
        # TODO: Implement agent improvement logic
        raise NotImplementedError("Agent improvement not yet implemented") 