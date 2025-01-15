from .agent_generator import DynamicAgentGenerator

# Allow both names to be used
AgentGenerator = DynamicAgentGenerator

__version__ = "0.1.0"
__all__ = ["DynamicAgentGenerator", "AgentGenerator"] 