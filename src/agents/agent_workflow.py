from .agent_library import library
from typing import List, Dict, Any, Callable, Optional, Union, Type
import autogen
from autogen.cache import Cache
from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    register_function,
)
from collections import defaultdict
from functools import partial
from abc import ABC, abstractmethod
from ..toolkits import register_toolkits
from ..functional.rag import get_rag_function
from .agentic_utils import instruction_trigger, instruction_message, order_trigger, order_message
from .prompts import leader_system_message, role_system_message


class FinBuddy(AssistantAgent):
    """
    FinBuddy is a specialized assistant agent designed to assist in financial tasks.
    It utilizes the GroupChatManager to manage interactions and task assignments.
    """
    def __init__(
            self,
            agent_config: Union[str, Dict[str, Any]],
            system_message: Optional[str] = None,
            toolkits: Optional[List[Callable]] = None,
            proxy: Optional[UserProxyAgent] = None,
            **kwargs
    ):
        """
        Initializes the FinBuddy agent with the given configuration and parameters.

        Args:
            agent_config: Configuration for the agent, either as a string or dictionary.
            system_message: Optional system message to overwrite the default.
            toolkits: Optional list of toolkits to overwrite the default.
            proxy: Optional UserProxyAgent for communication.
            **kwargs: Additional keyword arguments for customization.
        """
        orig_name = " "
        if isinstance(agent_config, str):
            orig_name = agent_config
            name = orig_name.replace("_Shadow", "")
            assert name in library, f"FinBuddy {name} not found in agent library."
            agent_config = library[name]
        
        agent_config = self._preprocess_config(agent_config)

        assert agent_config, "agent_config is required."
        assert agent_config.get("name", ""), "name needs to be configured."

        name = orig_name if orig_name.strip() else agent_config["name"]
        default_system_message = agent_config.get("profile", None)
        default_toolkits = agent_config.get("toolkits", [])

        system_message = system_message or default_system_message
        self.toolkits = toolkits or default_toolkits

        name = name.replace(" ", "_").strip()

        super().__init__(
            name=name, 
            system_message=system_message, 
            description=agent_config["description"], 
            **kwargs
        )

        if proxy is not None:
            self.register_proxy(proxy)

    def _preprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent configuration to ensure proper formatting."""
        role_prompt, leader_prompt = "", ""
        responsibilities = ""

        if "responsibilities" in config:
            title = config.get("title", config.get("name", ""))
            if "name" not in config:
                config["name"] = title
                
            responsibilities = config["responsibilities"]
            formatted_responsibilities = (
                "\n".join([f" - {r}" for r in responsibilities])
                if isinstance(responsibilities, list)
                else responsibilities
            )
            role_prompt = role_system_message.format(
                title=title,
                responsibilities=formatted_responsibilities,
            )
        
        name = config.get("name", "")
        description = (
            f"Name: {name}\nResponsibility:\n{responsibilities}"
            if responsibilities
            else f"Name: {name}"
        )

        config["description"] = description.strip()

        if "group_desc" in config:
            group_desc = config["group_desc"]
            leader_prompt = leader_system_message.format(
                group_members=group_desc,
            )

        config["profile"] = (
            (role_prompt + "\n\n" if role_prompt else "") +
            (leader_prompt + "\n\n" if leader_prompt else "") +
            config.get("profile", "")
        ).strip()

        return config
    
    def register_proxy(self, proxy: UserProxyAgent) -> None:
        """Register available toolkits with the proxy agent."""
        register_toolkits(self.toolkits, self, proxy)


class SingleAssistantBase(ABC):
    """
    Base class for single assistant agents. This class defines the basic structure and methods
    that all single assistant agents should implement.
    """
    def __init__(
        self,
        agent_config: Union[str, Dict[str, Any]],
        llm_config: Dict[str, Any] = None,
    ):
        """
        Initialize the base single assistant.

        Args:
            agent_config: Configuration for the assistant agent
            llm_config: Language model configuration
        """
        self.assistant = FinBuddy(
            agent_config=agent_config,
            llm_config=llm_config or {},
            proxy=None
        )
        
    @abstractmethod
    def chat(self, message: str, use_cache: bool = False, **kwargs) -> None:
        """Start a chat with this assistant."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the assistant state."""
        pass


class SingleAssistant(SingleAssistantBase):
    """
    SingleAssistant is a specialized class that manages a single assistant agent.
    It provides methods for chatting and resetting the agent's state.
    """
    def __init__(
        self,
        agent_config: Union[str, Dict[str, Any]],
        llm_config: Dict[str, Any] = None,
        is_termination_msg: Callable = lambda x: x.get("content", ""),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_replies: int = 10,
        code_execution_config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initializes the SingleAssistant with the given configuration and parameters.

        Args:
            agent_config: Configuration for the agent, either as a string or dictionary.
            llm_config: Configuration for the LLM (Language Model).
            is_termination_msg: Function to determine if a message is a termination message.
            human_input_mode: Mode for human input interaction.
            max_consecutive_auto_replies: Maximum number of consecutive auto-replies.
            code_execution_config: Configuration for code execution.
            **kwargs: Additional keyword arguments.
        """
        default_code_config = {
            "work_dir": "coding",
            "use_docker": False,
        }
        
        super().__init__(agent_config, llm_config=llm_config or {})
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_replies=max_consecutive_auto_replies,
            code_execution_config=code_execution_config or default_code_config,
            **kwargs,
        )
        self.assistant.register_proxy(self.user_proxy)

    def chat(self, message: str, use_cache: bool = False, **kwargs) -> None:
        """
        Start a chat session with the assistant.

        Args:
            message: Initial message to start the chat
            use_cache: Whether to use caching
            **kwargs: Additional arguments for the chat
        """
        with Cache.disk() as cache:
            self.user_proxy.initiate_chat(
                self.assistant,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )
        
        print("Current chat finished. Resetting agents...")
        self.reset()

    def reset(self) -> None:
        """Reset both the user proxy and assistant states."""
        self.user_proxy.reset()
        self.assistant.reset()


class SingleAssistantRAG(SingleAssistant):
    """Single assistant with Retrieval-Augmented Generation capabilities."""
    
    def __init__(
        self,
        agent_config: Union[str, Dict[str, Any]],
        llm_config: Dict[str, Any] = None,
        is_termination_msg: Callable = lambda x: x.get("content", ""),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_replies: int = 10,
        code_execution_config: Dict[str, Any] = None,
        retrieve_config: Dict[str, Any] = None,
        rag_description: str = "",
        **kwargs
    ):
        """
        Initialize a RAG-enabled assistant.

        Args:
            agent_config: Configuration for the agent
            llm_config: Language model configuration
            is_termination_msg: Function to determine if a message is a termination message
            human_input_mode: Mode for human input interaction
            max_consecutive_auto_replies: Maximum number of consecutive auto-replies
            code_execution_config: Configuration for code execution
            retrieve_config: Configuration for the retrieval system
            rag_description: Description of the RAG functionality
            **kwargs: Additional arguments
        """
        super().__init__(
            agent_config,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_replies=max_consecutive_auto_replies,
            code_execution_config=code_execution_config,
            **kwargs,
        )

        assert retrieve_config, "retrieve_config cannot be empty for RAG agent."
        rag_func, rag_assistant = get_rag_function(retrieve_config, rag_description)
        self.rag_assistant = rag_assistant
        register_function(
            rag_func,
            caller=self.assistant,
            executor=self.user_proxy,
            description=rag_description if rag_description else rag_func.__doc__,
        )

    def reset(self) -> None:
        """Reset all agents including the RAG assistant."""
        super().reset()
        self.rag_assistant.reset()
    

class SingleAssistantShadow(SingleAssistant):
    """Single assistant with a shadow agent for specific tasks."""
    
    def __init__(
        self,
        agent_config: Union[str, Dict[str, Any]],
        llm_config: Dict[str, Any] = None,
        is_termination_msg: Callable = lambda x: x.get("content", "") and x.get("content", "").endswith("TERMINATE"),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_replies: int = 10,
        code_execution_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize a shadow assistant that handles specific subtasks.

        Args:
            agent_config: Configuration for the agent
            llm_config: Language model configuration
            is_termination_msg: Function to determine if a message is a termination message
            human_input_mode: Mode for human input interaction
            max_consecutive_auto_replies: Maximum number of consecutive auto-replies
            code_execution_config: Configuration for code execution
            **kwargs: Additional arguments
        """
        super().__init__(
            agent_config=agent_config,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_replies=max_consecutive_auto_replies,
            code_execution_config=code_execution_config,
            **kwargs,
        )

        if isinstance(agent_config, dict):
            agent_config_shadow = agent_config.copy()
            agent_config_shadow["name"] = agent_config["name"] + "_Shadow"
            agent_config_shadow["toolkits"] = []
        else:
            agent_config_shadow = agent_config + "_Shadow"
        
        self.assistant_shadow = FinBuddy(
            agent_config=agent_config_shadow,
            toolkits=[],
            llm_config=llm_config,
            proxy=None,
        )
        
        self.assistant.register_nested_chats(
            [
                {
                    "sender": self.assistant,
                    "recipient": self.assistant_shadow,
                    "message": instruction_message,
                    "summary_method": "last_msg",
                    "max_turn": 2,
                    "silent": True,
                }
            ],
            trigger=instruction_trigger,
        )

    def reset(self) -> None:
        """Reset both the main assistant and its shadow."""
        super().reset()
        self.assistant_shadow.reset()


"""
Multi-Agent Workflows
"""

class MultiAssistantBase(ABC):
    """Base class for multi-assistant setups."""

    def __init__(
        self,
        group_config: Union[str, Dict[str, Any]],
        agent_configs: List[Union[Dict[str, Any], ConversableAgent]] = None,
        llm_config: Dict[str, Any] = None,
        user_proxy: UserProxyAgent = None,
        is_termination_msg: Callable = lambda x: x.get("content", "") and x.get("content", "").endswith("TERMINATE"),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_replies: int = 10,
        code_execution_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize the multi-assistant base.

        Args:
            group_config: Configuration for the group
            agent_configs: Configurations for individual agents
            llm_config: Language model configuration
            user_proxy: User proxy agent
            is_termination_msg: Function to determine if a message is a termination message
            human_input_mode: Mode for human input interaction
            max_consecutive_auto_replies: Maximum number of consecutive auto-replies
            code_execution_config: Configuration for code execution
            **kwargs: Additional arguments
        """
        self.group_config = group_config
        self.llm_config = llm_config or {}
        
        if user_proxy is None:
            default_code_config = {
                "work_dir": "coding",
                "use_docker": False,
            }
            self.user_proxy = UserProxyAgent(
                name="User_Proxy",
                is_termination_msg=is_termination_msg,
                human_input_mode=human_input_mode,
                max_consecutive_auto_replies=max_consecutive_auto_replies,
                code_execution_config=code_execution_config or default_code_config,
                **kwargs,
            )
        else:
            self.user_proxy = user_proxy
            
        self.agent_configs = agent_configs or (
            group_config.get("agents", []) if isinstance(group_config, dict) else []
        )
        assert self.agent_configs, "agent_configs is required."
        
        self.agents = []
        self._init_agents()
        self.representative = self._get_representative()
    
    def _init_single_agent(self, agent_config: Union[Dict[str, Any], ConversableAgent]) -> ConversableAgent:
        """Initialize a single agent from config or return the agent if already instantiated."""
        if isinstance(agent_config, ConversableAgent):
            return agent_config
        else:
            return FinBuddy(
                agent_config=agent_config,
                llm_config=self.llm_config,
                proxy=self.user_proxy,
            )
    
    def _init_agents(self) -> None:
        """Initialize all agents for the multi-agent setup."""
        agent_dict = defaultdict(list)
        for c in self.agent_configs:
            agent = self._init_single_agent(c)
            agent_dict[agent.name].append(agent)

        # Add index indicator for duplicate name/title
        for name, agent_list in agent_dict.items():
            if len(agent_list) == 1:
                self.agents.append(agent_list[0])
            else:
                for idx, agent in enumerate(agent_list):
                    agent._name = f"{name}_{idx+1}"
                    self.agents.append(agent)

    @abstractmethod
    def _get_representative(self) -> ConversableAgent:
        """Get the representative agent for the group."""
        pass

    def chat(self, message: str, use_cache: bool = False, **kwargs) -> None:
        """
        Start a chat with the representative agent.

        Args:
            message: Initial message to start the chat
            use_cache: Whether to use caching
            **kwargs: Additional arguments for the chat
        """
        with Cache.disk() as cache:
            self.user_proxy.initiate_chat(
                self.representative,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )
        
        print("Current chat finished. Resetting agents...")
        self.reset()
    
    def reset(self) -> None:
        """Reset all agents in the multi-agent setup."""
        self.user_proxy.reset()
        self.representative.reset()
        for agent in self.agents:
            agent.reset()


class MultiAssistant(MultiAssistantBase):
    """
    MultiAssistant is a specialized class that manages multiple assistant agents.
    It provides methods for chatting and resetting the agents' state.
    """
    def _get_representative(self) -> GroupChatManager:
        """
        Get the representative for the group, which is a GroupChatManager in this case.
        """
        def custom_speaker_selection_func(
                last_speaker: autogen.Agent, groupchat: autogen.GroupChat
        ) -> autogen.Agent:
            """
            Custom function to select the speaker in a group chat.
            This function ensures that the last speaker is not selected again.
            """
            messages = groupchat.messages
            if len(messages) <= 1:
                return groupchat.agents[0]
            if last_speaker is self.user_proxy:
                return groupchat.agent_by_name(messages[-2]["name"])
            elif "tool_calls" in messages[-1] or messages[-1]["content"].endswith("TERMINATE"):
                return self.user_proxy
            else:
                return groupchat.next_agent(last_speaker, groupchat.agents[:-1])
        
        self.group_chat = GroupChat(
            agents=self.agents + [self.user_proxy],
            messages=[],
            speaker_selection_method=custom_speaker_selection_func,
            send_introduction=True,
        )

        manager_name = (
            (self.group_config.get("name", "") if isinstance(self.group_config, dict) else "") + 
            "_chat_manager"
        ).strip("_")
        manager = GroupChatManager(
            groupchat=self.group_chat, name=manager_name, llm_config=self.llm_config
        )
        return manager
    

class MultiAssistantWithLeader(MultiAssistantBase):
    """
    Leader based Workflow with multiple agents connected to a leader agent through nested chats.

    Group config has to follow the following structure:
    {
        "leader": {
            "title": "Leader Title",
            "responsibilities": ["responsibility 1", "responsibility 2"]
        },
        "agents": [
            {
                "title": "Employee Title",
                "responsibilities": ["responsibility 1", "responsibility 2"]
            }, ...
        ]
    }
    """
    def _get_representative(self) -> FinBuddy:
        """
        Get the leader agent as the representative.
        """
        assert (
            isinstance(self.group_config, dict) and
            "leader" in self.group_config and 
            "agents" in self.group_config
        ), "Leader and Agents must be explicitly defined in the group config."

        assert self.agent_configs, "At least one agent must be defined in the group config."
        
        # Determine if we need name suffixes (when all agents have the same title)
        need_suffix = False
        titles = set()
        for c in self.agent_configs:
            if isinstance(c, dict):
                title = c.get("title", c.get("name", ""))
                titles.add(title)
        need_suffix = len(titles) == 1 and len(self.agent_configs) > 1
                
        group_desc = ""
        for i, c in enumerate(self.agent_configs):
            if isinstance(c, ConversableAgent):
                group_desc += c.description + "\n\n"
            else:
                name = c.get("title", c.get("name", ""))
                name = name.replace(" ", "_").strip() + (
                    f"_{i+1}" if need_suffix else ""
                )
                
                if "responsibilities" in c and isinstance(c["responsibilities"], list):
                    responsibilities = "\n".join([f" - {r}" for r in c["responsibilities"]])
                else:
                    responsibilities = c.get("responsibilities", "")
                    
                group_desc += f"Name: {name}\nResponsibility:\n{responsibilities}\n\n"
        
        self.leader_config = self.group_config["leader"]
        self.leader_config["group_desc"] = group_desc.strip()

        # Initialize leader
        leader = self._init_single_agent(self.leader_config)

        # Register Leader - Agents connections
        for agent in self.agents:
            self.user_proxy.register_nested_chats(
                [
                    {
                        "sender": self.user_proxy,
                        "recipient": agent,
                        "message": partial(order_message, agent.name),
                        "summary_method": "reflection_with_llm",
                        "max_turn": 10,
                        "max_consecutive_auto_replies": 3,
                    }
                ],
                trigger=partial(
                    order_trigger, name=leader.name, pattern=f"[{agent.name}]"
                ),
            )
            
        return leader