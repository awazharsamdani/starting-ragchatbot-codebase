import openai
import json
from typing import List, Optional, Dict, Any

# Module-level state
_client = None
_model = None

# Static system prompt
SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

# Base API parameters
BASE_PARAMS = {
    "temperature": 0,
    "max_tokens": 800
}

def initialize_ai_generator(api_key: str, model: str):
    """Initialize the AI generator with OpenAI client"""
    global _client, _model
    _client = openai.OpenAI(api_key=api_key)
    _model = model

def generate_response(query: str, conversation_history: Optional[str] = None,
                     tools: Optional[List] = None) -> str:
    """
    Generate AI response with optional tool usage and conversation context.

    Args:
        query: The user's question or request
        conversation_history: Previous messages for context
        tools: Available tools the AI can use

    Returns:
        Generated response as string
    """

    # Build system content efficiently
    system_content = (
        f"{SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        if conversation_history
        else SYSTEM_PROMPT
    )

    # Prepare messages array for OpenAI
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]

    # Prepare API call parameters efficiently
    api_params = {
        "model": _model,
        **BASE_PARAMS,
        "messages": messages
    }

    # Add tools if available
    if tools:
        api_params["tools"] = tools
        api_params["tool_choice"] = "auto"

    # Get response from OpenAI
    response = _client.chat.completions.create(**api_params)

    # Handle tool execution if needed
    if response.choices[0].finish_reason == "tool_calls":
        return _handle_tool_execution(response, messages, tools)

    # Return direct response
    return response.choices[0].message.content

def _handle_tool_execution(initial_response, messages: List[Dict[str, Any]], tools):
    """
    Handle execution of tool calls and get follow-up response.

    Args:
        initial_response: The response containing tool use requests
        messages: Existing message history
        tools: Tool definitions

    Returns:
        Final response text after tool execution
    """
    # Import here to avoid circular dependency
    import search_tools

    # Add assistant's tool call message to history
    assistant_message = {
        "role": "assistant",
        "content": initial_response.choices[0].message.content,
        "tool_calls": initial_response.choices[0].message.tool_calls
    }
    messages.append(assistant_message)

    # Execute all tool calls and add results
    for tool_call in initial_response.choices[0].message.tool_calls:
        # Parse function arguments (comes as JSON string)
        function_args = json.loads(tool_call.function.arguments)

        # Execute the tool
        tool_result = search_tools.execute_tool(
            tool_call.function.name,
            **function_args
        )

        # Add tool result to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

    # Prepare final API call
    final_params = {
        "model": _model,
        **BASE_PARAMS,
        "messages": messages
    }

    # Get final response
    final_response = _client.chat.completions.create(**final_params)
    return final_response.choices[0].message.content
