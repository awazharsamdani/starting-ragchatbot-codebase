from typing import Dict, Any, Optional, List, Callable
import vector_store

# Module-level state for tools
_tools = {}
_last_sources = []

# Tool definitions and execution

def get_course_search_tool_definition() -> Dict[str, Any]:
    """Return OpenAI tool definition for course search"""
    return {
        "type": "function",
        "function": {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    }

def execute_course_search(query: str, course_name: Optional[str] = None,
                         lesson_number: Optional[int] = None) -> str:
    """
    Execute the search tool with given parameters.

    Args:
        query: What to search for
        course_name: Optional course filter
        lesson_number: Optional lesson filter

    Returns:
        Formatted search results or error message
    """
    global _last_sources

    # Use the vector store's unified search interface
    results = vector_store.search(
        query=query,
        course_name=course_name,
        lesson_number=lesson_number
    )

    # Handle errors
    if results["error"]:
        return results["error"]

    # Handle empty results
    if vector_store.is_empty_results(results):
        filter_info = ""
        if course_name:
            filter_info += f" in course '{course_name}'"
        if lesson_number:
            filter_info += f" in lesson {lesson_number}"
        return f"No relevant content found{filter_info}."

    # Format and return results
    return _format_results(results)

def _format_results(results: dict) -> str:
    """Format search results with course and lesson context"""
    global _last_sources

    formatted = []
    sources = []  # Track sources for the UI

    for doc, meta in zip(results["documents"], results["metadata"]):
        course_title = meta.get('course_title', 'unknown')
        lesson_num = meta.get('lesson_number')

        # Build context header
        header = f"[{course_title}"
        if lesson_num is not None:
            header += f" - Lesson {lesson_num}"
        header += "]"

        # Track source for the UI
        source = course_title
        if lesson_num is not None:
            source += f" - Lesson {lesson_num}"
        sources.append(source)

        formatted.append(f"{header}\n{doc}")

    # Store sources for retrieval
    _last_sources = sources

    return "\n\n".join(formatted)

# Tool Manager functions

def register_tool(name: str, definition_func: Callable, execute_func: Callable):
    """Register a tool with its definition and execution function"""
    _tools[name] = {
        "definition": definition_func,
        "execute": execute_func
    }

def initialize_tools():
    """Initialize all available tools"""
    register_tool(
        "search_course_content",
        get_course_search_tool_definition,
        execute_course_search
    )

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get all tool definitions for OpenAI tool calling"""
    return [tool["definition"]() for tool in _tools.values()]

def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool by name with given parameters"""
    if tool_name not in _tools:
        return f"Tool '{tool_name}' not found"

    return _tools[tool_name]["execute"](**kwargs)

def get_last_sources() -> List[str]:
    """Get sources from the last search operation"""
    return _last_sources.copy()

def reset_sources():
    """Reset sources from all tools that track sources"""
    global _last_sources
    _last_sources = []
