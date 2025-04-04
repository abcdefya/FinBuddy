import re
import os
from typing import Dict, List, Any
from .prompts import order_template

def instruction_trigger(sender: Any) -> bool:
    """
    Determines if a message contains an instruction trigger phrase.
    
    Args:
        sender: The entity that sent the message
        
    Returns:
        bool: True if the trigger phrase is found, False otherwise
    """
    return "intruction & resources saved to" in sender.last_message()["content"]

def instruction_message(recipient: Any, messages: List[Dict], sender: Any, config: Dict) -> str:
    """
    Processes an instruction message by reading the referenced file and appending 
    standard termination text.
    
    Args:
        recipient: The recipient of the instruction
        messages: The message history
        sender: The sender of the instruction
        config: Configuration parameters
        
    Returns:
        str: The processed instruction text
    """
    # Extract the path to the instruction text file
    full_order = recipient.chat_messages_for_summary(sender)[-1]["content"]
    txt_path = full_order.replace("instruction & resources saved to ", "").strip()
    
    # Validate file exists before reading
    if not os.path.exists(txt_path):
        return f"Error: Instruction file not found at {txt_path}"
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            instruction_text = f.read() + "\n\nReply with 'TERMINATE' when your assigned tasks are complete."
        return instruction_text
    except Exception as e:
        return f"Error reading instruction file: {str(e)}"

def order_trigger(sender: Any, name: str, pattern: str) -> bool:
    """
    Checks if an order matches the specified pattern from the expected sender.
    
    Args:
        sender: The entity that sent the message
        name: The expected name of the sender
        pattern: The pattern to match in the message
        
    Returns:
        bool: True if the sender matches and the pattern is found, False otherwise
    """
    return sender.name == name and pattern in sender.last_message()["content"]

def order_message(pattern: str, recipient: Any, messages: List[Dict], sender: Any, config: Dict) -> str:
    """
    Extracts and formats an order from a message based on the specified pattern.
    
    Args:
        pattern: The pattern to match for the order
        recipient: The recipient of the order
        messages: The message history
        sender: The sender of the order
        config: Configuration parameters
        
    Returns:
        str: The formatted order message
    """
    # Get the latest message content
    full_order = recipient.chat_messages_for_summary(sender)[-1]["content"]
    
    # Pattern to extract the relevant order section
    regex_pattern = rf"\[{pattern}\](?::)?\s*(.+?)(?=\n\[|$)"
    
    # Extract the order using regex
    match = re.search(regex_pattern, full_order, re.DOTALL)
    
    if match:
        order = match.group(1).strip()
    else:
        # If pattern doesn't match, use the full message as fallback
        order = full_order
    
    # Format the order using the template
    return order_template.format(order=order)