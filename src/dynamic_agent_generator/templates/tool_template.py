from smolagents import Tool
from typing import Any, Dict, Optional

class BaseTool(Tool):
    """Base template for generated tools."""
    
    name: str = ""
    description: str = ""
    inputs: Dict = {}
    output_type: str = "Any"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, **kwargs) -> Any:
        """Main execution method for the tool."""
        raise NotImplementedError("Subclasses must implement forward method") 