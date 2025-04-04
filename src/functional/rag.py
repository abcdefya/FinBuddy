from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from typing import Annotated, Tuple, Dict, Any, Callable, Optional, Union, List

# Default prompt template for retrieval function
DEFAULT_RETRIEVAL_PROMPT = """
    Below is the context retrieved from the knowledge base based on your query.
    
    If the retrieved context doesn't adequately answer your question:
    1. Try refining your search query to be more specific
    2. Request additional context if needed
    3. Specify different document sources if available
    
    Your query: {input_query}
    
    Retrieved context: {intput_context}
"""


def create_knowledge_retriever(
    retrieval_config: Dict[str, Any], 
    description: str = ""
) -> Tuple[Callable, RetrieveUserProxyAgent]:
    """
    Creates a knowledge retrieval function and its associated agent.
    
    Args:
        retrieval_config: Configuration for the retrieval system
        description: Optional custom description for the retrieval function
        
    Returns:
        Tuple containing:
          - The knowledge retrieval function
          - The retrieval agent instance
    """
    # Define termination condition for the retrieval agent
    def is_termination_message(message: Dict[str, Any]) -> bool:
        """Check if a message indicates the retrieval task is complete."""
        if not isinstance(message, dict):
            return False
        
        content = str(message.get("content", ""))
        return content.strip().upper().endswith("TERMINATE")
    
    # Apply default prompt if not specified in config
    if "customized_prompt" not in retrieval_config:
        retrieval_config["customized_prompt"] = DEFAULT_RETRIEVAL_PROMPT
    
    # Initialize the retrieval agent
    knowledge_agent = RetrieveUserProxyAgent(
        name="Knowledge_Retriever",
        is_termination_msg=is_termination_message,
        human_input_mode="NEVER",
        default_auto_reply="Reply 'TERMINATE' if the retrieval task is complete.",
        max_consecutive_auto_reply=3,
        retrieve_config=retrieval_config,
        code_execution_config=False,
        description="Specialized agent with document retrieval capabilities for accessing relevant information",
    )

    def retrieve_knowledge(
        query: Annotated[
            str,
            "Specific query to retrieve relevant information from the knowledge base. "
            "Example queries: 'YoY comparisons of profit margin', 'risk factors of NVIDIA in Q4', "
            "'historical price trends for AAPL from 2020-2022'"
        ],
        result_count: Annotated[int, "Number of results to retrieve"] = 3,
    ) -> str:
        """
        Retrieve relevant information from the knowledge base.
        
        Args:
            query: The search query for retrieving information
            result_count: Number of results to return (default: 3)
            
        Returns:
            Retrieved content as formatted text
        """
        # Set the number of results to retrieve
        knowledge_agent.n_results = result_count

        # Check if context update is needed
        should_update_case1, should_update_case2 = knowledge_agent._check_update_context(query)
        
        if (should_update_case1 or should_update_case2) and knowledge_agent.update_context:
            # If updating existing context
            knowledge_agent.problem = (
                query if not hasattr(knowledge_agent, "problem") else knowledge_agent.problem
            )
            _, response_message = knowledge_agent._generate_retrieve_user_reply(query)
        else:
            # If generating new context
            context = {"problem": query, "n_results": result_count}
            response_message = knowledge_agent.message_generator(knowledge_agent, None, context)
        
        # Return the retrieved content or the original query if retrieval failed
        return response_message if response_message else query
    
    # Set the function documentation
    if description:
        retrieve_knowledge.__doc__ = description
    else:
        # Create default documentation with available document information
        base_doc = "Retrieve information from the knowledge base to answer questions or support analysis."
        docs_paths = retrieval_config.get("docs_path", [])
        
        if docs_paths:
            docs_list = docs_paths if isinstance(docs_paths, list) else [docs_paths]
            formatted_docs = "\n - ".join(docs_list)
            retrieve_knowledge.__doc__ = f"{base_doc}\n\nAvailable Documents:\n - {formatted_docs}"
        else:
            retrieve_knowledge.__doc__ = base_doc

    return retrieve_knowledge, knowledge_agent
