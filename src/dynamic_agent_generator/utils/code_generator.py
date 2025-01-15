from typing import Dict, List
from pathlib import Path
import inspect
from ..templates import agent_template, tool_template

def generate_tool_code(
    name: str,
    description: str,
    inputs: Dict,
    output_type: str
) -> str:
    """Generate code for a new tool."""
    
    template = f'''
from typing import Dict, Any
from dynamic_agent_generator.templates.tool_template import BaseTool

class {name.title()}Tool(BaseTool):
    """
    {description}
    """
    name = "{name}"
    description = "{description}"
    inputs = {inputs}
    output_type = "{output_type}"
    
    def forward(self, **kwargs) -> Any:
        """Implementation of the {name} tool logic."""
        # TODO: Implement specific tool logic
        raise NotImplementedError("Tool logic needs to be implemented")
'''
    return template

def generate_agent_code(
    agent_name: str,
    description: str,
    tool_paths: List[Path]
) -> str:
    """Generate code for a new agent."""
    
    imports = []
    tool_inits = []
    
    for tool_path in tool_paths:
        tool_name = tool_path.stem
        imports.append(f"from .{tool_name} import {tool_name.title()}Tool")
        tool_inits.append(f"        self.{tool_name}_tool = {tool_name.title()}Tool()")
    
    template = f'''
from typing import List, Optional, Any
from dynamic_agent_generator.templates.agent_template import BaseAgent
{chr(10).join(imports)}

class {agent_name}(BaseAgent):
    """
    {description}
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        **kwargs
    ):
        # Initialize tools
{chr(10).join(tool_inits)}
        
        tools = [{", ".join(f"self.{t.stem}_tool" for t in tool_paths)}]
        
        super().__init__(
            tools=tools,
            model_id=model_id,
            description=self.__doc__,
            **kwargs
        )
    
    def run(self, **kwargs) -> Any:
        """Main execution method for {agent_name}."""
        # TODO: Implement agent-specific logic
        raise NotImplementedError("Agent run method needs to be implemented")
'''
    return template 