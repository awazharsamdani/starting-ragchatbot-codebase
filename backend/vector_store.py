import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import openai
import json

# Module-level state for vector store
_client = None
_embedding_function = None
_course_catalog = None
_course_content = None
_max_results = 5

# OpenAI embedding function
def create_embedding_function(api_key: str, model_name: str = "text-embedding-3-small"):
    """Create embedding function that uses OpenAI API"""
    client = openai.OpenAI(api_key=api_key)

    def embed_texts(input_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        # Replace newlines which can negatively affect performance
        input_texts = [text.replace("\n", " ") for text in input_texts]

        response = client.embeddings.create(
            model=model_name,
            input=input_texts
        )

        return [item.embedding for item in response.data]

    return embed_texts

# Search results container
def create_search_results(documents: List[str], metadata: List[Dict[str, Any]],
                         distances: List[float], error: Optional[str] = None) -> dict:
    """Create a search results dictionary"""
    return {
        "documents": documents,
        "metadata": metadata,
        "distances": distances,
        "error": error
    }

def search_results_from_chroma(chroma_results: Dict) -> dict:
    """Create SearchResults from ChromaDB query results"""
    return create_search_results(
        documents=chroma_results['documents'][0] if chroma_results['documents'] else [],
        metadata=chroma_results['metadatas'][0] if chroma_results['metadatas'] else [],
        distances=chroma_results['distances'][0] if chroma_results['distances'] else []
    )

def empty_search_results(error_msg: str) -> dict:
    """Create empty results with error message"""
    return create_search_results([], [], [], error_msg)

def is_empty_results(results: dict) -> bool:
    """Check if results are empty"""
    return len(results["documents"]) == 0

# Vector store initialization
def initialize_vector_store(chroma_path: str, openai_api_key: str,
                           embedding_model: str, max_results: int = 5):
    """Initialize the vector store with ChromaDB"""
    global _client, _embedding_function, _course_catalog, _course_content, _max_results

    _max_results = max_results

    # Initialize ChromaDB client
    _client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False)
    )

    # Set up OpenAI embedding function
    _embedding_function = create_embedding_function(
        api_key=openai_api_key,
        model_name=embedding_model
    )

    # Wrap the embedding function for ChromaDB compatibility
    class EmbeddingWrapper:
        def __init__(self, model_name):
            self.model_name = model_name

        def __call__(self, input: List[str]) -> List[List[float]]:
            return _embedding_function(input)

        def name(self) -> str:
            return f"openai_{self.model_name}"

    embedding_wrapper = EmbeddingWrapper(embedding_model)

    # Create collections for different types of data
    _course_catalog = _client.get_or_create_collection(
        name="course_catalog",
        embedding_function=embedding_wrapper
    )
    _course_content = _client.get_or_create_collection(
        name="course_content",
        embedding_function=embedding_wrapper
    )

def search(query: str, course_name: Optional[str] = None,
          lesson_number: Optional[int] = None, limit: Optional[int] = None) -> dict:
    """
    Main search interface that handles course resolution and content search.

    Args:
        query: What to search for in course content
        course_name: Optional course name/title to filter by
        lesson_number: Optional lesson number to filter by
        limit: Maximum results to return

    Returns:
        Search results dictionary with documents and metadata
    """
    # Step 1: Resolve course name if provided
    course_title = None
    if course_name:
        course_title = _resolve_course_name(course_name)
        if not course_title:
            return empty_search_results(f"No course found matching '{course_name}'")

    # Step 2: Build filter for content search
    filter_dict = _build_filter(course_title, lesson_number)

    # Step 3: Search course content
    search_limit = limit if limit is not None else _max_results

    try:
        results = _course_content.query(
            query_texts=[query],
            n_results=search_limit,
            where=filter_dict
        )
        return search_results_from_chroma(results)
    except Exception as e:
        return empty_search_results(f"Search error: {str(e)}")

def _resolve_course_name(course_name: str) -> Optional[str]:
    """Use vector search to find best matching course by name"""
    try:
        results = _course_catalog.query(
            query_texts=[course_name],
            n_results=1
        )

        if results['documents'][0] and results['metadatas'][0]:
            # Return the title (which is now the ID)
            return results['metadatas'][0][0]['title']
    except Exception as e:
        print(f"Error resolving course name: {e}")

    return None

def _build_filter(course_title: Optional[str], lesson_number: Optional[int]) -> Optional[Dict]:
    """Build ChromaDB filter from search parameters"""
    if not course_title and lesson_number is None:
        return None

    # Handle different filter combinations
    if course_title and lesson_number is not None:
        return {"$and": [
            {"course_title": course_title},
            {"lesson_number": lesson_number}
        ]}

    if course_title:
        return {"course_title": course_title}

    return {"lesson_number": lesson_number}

def add_course_metadata(course: dict):
    """Add course information to the catalog for semantic search"""
    course_text = course["title"]

    # Build lessons metadata and serialize as JSON string
    lessons_metadata = []
    for lesson in course["lessons"]:
        lessons_metadata.append({
            "lesson_number": lesson["lesson_number"],
            "lesson_title": lesson["title"],
            "lesson_link": lesson["lesson_link"]
        })

    _course_catalog.add(
        documents=[course_text],
        metadatas=[{
            "title": course["title"],
            "instructor": course["instructor"],
            "course_link": course["course_link"],
            "lessons_json": json.dumps(lessons_metadata),
            "lesson_count": len(course["lessons"])
        }],
        ids=[course["title"]]
    )

def add_course_content(chunks: List[dict]):
    """Add course content chunks to the vector store"""
    if not chunks:
        return

    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{
        "course_title": chunk["course_title"],
        "lesson_number": chunk["lesson_number"],
        "chunk_index": chunk["chunk_index"]
    } for chunk in chunks]
    # Use title with chunk index for unique IDs
    ids = [f"{chunk['course_title'].replace(' ', '_')}_{chunk['chunk_index']}" for chunk in chunks]

    _course_content.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

def clear_all_data():
    """Clear all data from both collections"""
    global _course_catalog, _course_content

    try:
        _client.delete_collection("course_catalog")
        _client.delete_collection("course_content")

        # Recreate collections
        class EmbeddingWrapper:
            def __init__(self, model_name):
                self.model_name = model_name

            def __call__(self, input: List[str]) -> List[List[float]]:
                return _embedding_function(input)

            def name(self) -> str:
                return f"openai_{self.model_name}"

        # We need the model name - get it from module state or use default
        embedding_wrapper = EmbeddingWrapper("text-embedding-3-small")

        _course_catalog = _client.get_or_create_collection(
            name="course_catalog",
            embedding_function=embedding_wrapper
        )
        _course_content = _client.get_or_create_collection(
            name="course_content",
            embedding_function=embedding_wrapper
        )
    except Exception as e:
        print(f"Error clearing data: {e}")

def get_existing_course_titles() -> List[str]:
    """Get all existing course titles from the vector store"""
    try:
        # Get all documents from the catalog
        results = _course_catalog.get()
        if results and 'ids' in results:
            return results['ids']
        return []
    except Exception as e:
        print(f"Error getting existing course titles: {e}")
        return []

def get_course_count() -> int:
    """Get the total number of courses in the vector store"""
    try:
        results = _course_catalog.get()
        if results and 'ids' in results:
            return len(results['ids'])
        return 0
    except Exception as e:
        print(f"Error getting course count: {e}")
        return 0

def get_all_courses_metadata() -> List[Dict[str, Any]]:
    """Get metadata for all courses in the vector store"""
    try:
        results = _course_catalog.get()
        if results and 'metadatas' in results:
            # Parse lessons JSON for each course
            parsed_metadata = []
            for metadata in results['metadatas']:
                course_meta = metadata.copy()
                if 'lessons_json' in course_meta:
                    course_meta['lessons'] = json.loads(course_meta['lessons_json'])
                    del course_meta['lessons_json']  # Remove the JSON string version
                parsed_metadata.append(course_meta)
            return parsed_metadata
        return []
    except Exception as e:
        print(f"Error getting courses metadata: {e}")
        return []

def get_course_link(course_title: str) -> Optional[str]:
    """Get course link for a given course title"""
    try:
        # Get course by ID (title is the ID)
        results = _course_catalog.get(ids=[course_title])
        if results and 'metadatas' in results and results['metadatas']:
            metadata = results['metadatas'][0]
            return metadata.get('course_link')
        return None
    except Exception as e:
        print(f"Error getting course link: {e}")
        return None

def get_lesson_link(course_title: str, lesson_number: int) -> Optional[str]:
    """Get lesson link for a given course title and lesson number"""
    try:
        # Get course by ID (title is the ID)
        results = _course_catalog.get(ids=[course_title])
        if results and 'metadatas' in results and results['metadatas']:
            metadata = results['metadatas'][0]
            lessons_json = metadata.get('lessons_json')
            if lessons_json:
                lessons = json.loads(lessons_json)
                # Find the lesson with matching number
                for lesson in lessons:
                    if lesson.get('lesson_number') == lesson_number:
                        return lesson.get('lesson_link')
        return None
    except Exception as e:
        print(f"Error getting lesson link: {e}")
        return None
