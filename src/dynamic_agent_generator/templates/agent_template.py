from smolagents import CodeAgent, HfApiModel
from typing import List, Optional

class BaseAgent(CodeAgent):
    """Base template for generated agents."""
    
    def __init__(
        self,
        tools: List,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        description: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            tools=tools,
            model=HfApiModel(model_id=model_id),
            description=description or self.__doc__,
            **kwargs
        )
    
    def run(self, **kwargs):
        """Main execution method for the agent."""
        raise NotImplementedError("Subclasses must implement run method") 