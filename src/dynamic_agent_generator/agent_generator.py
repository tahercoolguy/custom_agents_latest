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

    def _get_completion(self, prompt: str) -> str:
        """Helper method to get completion from the model."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that generates code and specifications."},
            {"role": "user", "content": prompt}
        ]
        response = self.model(messages)
        # Extract content from ChatCompletionOutputMessage
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()

    def _determine_required_tools(self, description: str) -> List[Dict]:
        """Use LLM to determine required tools based on agent description."""
        prompt = f"""Based on the following agent description, determine the necessary tools needed.
        Return only the Python list of dictionaries, nothing else.
        
        Agent Description: {description}
        
        The tools should be in this exact format:
        [
            {{
                "name": "tool_name",
                "description": "what the tool does",
                "inputs": {{"param_name": {{"type": "type", "description": "param description"}}}},
                "output_type": "return_type"
            }}
        ]
        """
        
        response = self._get_completion(prompt)
        try:
            # Safely evaluate the response to get the list of tools
            tools = eval(response)
            if not isinstance(tools, list):
                raise ValueError("Response must be a list of tool specifications")
            return tools
        except Exception as e:
            # Fallback to default tools if parsing fails
            print(f"Warning: Failed to parse LLM response: {e}")
            return self._get_default_tools(description)

    def _get_default_tools(self, description: str) -> List[Dict]:
        """Provide default tools if LLM parsing fails."""
        return [{
            "name": "default_tool",
            "description": "Default tool for basic operations",
            "inputs": {
                "input": {"type": "string", "description": "Input to process"}
            },
            "output_type": "string"
        }]

    def create_agent(
        self,
        description: str,
        save_path: str,
        agent_name: Optional[str] = None
    ) -> str:
        """Create a new agent based on natural language description.
        
        Args:
            description: Natural language description of what the agent should do
            save_path: Where to save the generated agent
            agent_name: Optional name for the agent (will be generated if not provided)
            
        Returns:
            Path to the generated agent
        """
        # Generate agent name if not provided
        if agent_name is None:
            prompt = "Generate a single camelCase word to name an agent with this description: " + description
            try:
                agent_name = self._get_completion(prompt)
                if not agent_name.isidentifier():
                    raise ValueError("Invalid identifier")
            except:
                agent_name = "CustomAgent"
        
        # Determine required tools using LLM
        required_tools = self._determine_required_tools(description)
        
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

    def _validate_tool_spec(self, tool_spec: Dict) -> None:
        """Validate tool specification."""
        required_fields = ["name", "description", "inputs", "output_type"]
        for field in required_fields:
            if field not in tool_spec:
                raise ValueError(f"Tool specification missing required field: {field}")
        
        if not isinstance(tool_spec["inputs"], dict):
            raise ValueError("Tool inputs must be a dictionary")

    def improve_agent(self, agent_path: str, feedback: str) -> str:
        """Improve an existing agent based on feedback."""
        raise NotImplementedError("Agent improvement not yet implemented") 