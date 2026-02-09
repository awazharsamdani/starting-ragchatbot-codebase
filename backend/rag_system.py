from typing import List, Tuple, Optional, Dict
import os
import document_processor
import vector_store
import ai_generator
import session_manager
import search_tools

# Module initialization flag
_initialized = False

def initialize_rag_system(config_module):
    """Initialize the RAG system with all components"""
    global _initialized

    if _initialized:
        return

    # Initialize all components
    vector_store.initialize_vector_store(
        config_module.CHROMA_PATH,
        config_module.OPENAI_API_KEY,
        config_module.OPENAI_EMBEDDING_MODEL,
        config_module.MAX_RESULTS
    )

    ai_generator.initialize_ai_generator(
        config_module.OPENAI_API_KEY,
        config_module.OPENAI_MODEL
    )

    session_manager.initialize_session_manager(config_module.MAX_HISTORY)

    # Initialize search tools
    search_tools.initialize_tools()

    _initialized = True

def add_course_document(file_path: str, chunk_size: int, chunk_overlap: int) -> Tuple[Optional[dict], int]:
    """
    Add a single course document to the knowledge base.

    Args:
        file_path: Path to the course document
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Tuple of (Course dict, number of chunks created)
    """
    try:
        # Process the document
        course, course_chunks = document_processor.process_course_document(
            file_path, chunk_size, chunk_overlap
        )

        # Add course metadata to vector store for semantic search
        vector_store.add_course_metadata(course)

        # Add course content chunks to vector store
        vector_store.add_course_content(course_chunks)

        return course, len(course_chunks)
    except Exception as e:
        print(f"Error processing course document {file_path}: {e}")
        return None, 0

def add_course_folder(folder_path: str, chunk_size: int, chunk_overlap: int,
                     clear_existing: bool = False) -> Tuple[int, int]:
    """
    Add all course documents from a folder.

    Args:
        folder_path: Path to folder containing course documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        clear_existing: Whether to clear existing data first

    Returns:
        Tuple of (total courses added, total chunks created)
    """
    total_courses = 0
    total_chunks = 0

    # Clear existing data if requested
    if clear_existing:
        print("Clearing existing data for fresh rebuild...")
        vector_store.clear_all_data()

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return 0, 0

    # Get existing course titles to avoid re-processing
    existing_course_titles = set(vector_store.get_existing_course_titles())

    # Process each file in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
            try:
                # Check if this course might already exist
                course, course_chunks = document_processor.process_course_document(
                    file_path, chunk_size, chunk_overlap
                )

                if course and course["title"] not in existing_course_titles:
                    # This is a new course - add it to the vector store
                    vector_store.add_course_metadata(course)
                    vector_store.add_course_content(course_chunks)
                    total_courses += 1
                    total_chunks += len(course_chunks)
                    print(f"Added new course: {course['title']} ({len(course_chunks)} chunks)")
                    existing_course_titles.add(course["title"])
                elif course:
                    print(f"Course already exists: {course['title']} - skipping")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return total_courses, total_chunks

def query(query_text: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Process a user query using the RAG system with tool-based search.

    Args:
        query_text: User's question
        session_id: Optional session ID for conversation context

    Returns:
        Tuple of (response, sources list)
    """
    # Create prompt for the AI with clear instructions
    prompt = f"""Answer this question about course materials: {query_text}"""

    # Get conversation history if session exists
    history = None
    if session_id:
        history = session_manager.get_conversation_history(session_id)

    # Generate response using AI with tools
    response = ai_generator.generate_response(
        query=prompt,
        conversation_history=history,
        tools=search_tools.get_tool_definitions()
    )

    # Get sources from the search tool
    sources = search_tools.get_last_sources()

    # Reset sources after retrieving them
    search_tools.reset_sources()

    # Update conversation history
    if session_id:
        session_manager.add_exchange(session_id, query_text, response)

    # Return response with sources from tool searches
    return response, sources

def get_course_analytics() -> Dict:
    """Get analytics about the course catalog"""
    return {
        "total_courses": vector_store.get_course_count(),
        "course_titles": vector_store.get_existing_course_titles()
    }
