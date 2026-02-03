# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. The system uses ChromaDB for vector storage, Anthropic's Claude API with tool-calling for AI responses, and sentence transformers for embeddings. The frontend is vanilla HTML/CSS/JS served by FastAPI.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables (required)
# Create .env file with:
ANTHROPIC_API_KEY=your_key_here
```

### Running the Application
```bash
# Start the development server (from project root)
./run.sh

# Or manually from backend directory
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application runs on http://localhost:8000 with API docs at http://localhost:8000/docs.

### Running Python Modules
When running backend modules directly, use `uv run python` from the backend directory:
```bash
cd backend
uv run python -m module_name
```

## Architecture

### Core Data Flow

1. **Document Processing** → **Vector Storage** → **AI Query with Tools** → **Response**
   - Course documents (txt/pdf/docx) are processed into structured `Course` objects with `Lesson` metadata
   - Content is chunked and stored as `CourseChunk` objects in ChromaDB with two collections:
     - `course_catalog`: Course metadata (title, instructor, lessons) for semantic course name matching
     - `course_content`: Actual course content chunks with course_title and lesson_number filters

2. **Query Processing Uses Tool-Calling Pattern**:
   - User queries go to `RAGSystem.query()` → `AIGenerator.generate_response()`
   - Claude is provided with `search_course_content` tool via `ToolManager`
   - Claude decides when to search and extracts search parameters (query, course_name, lesson_number)
   - `CourseSearchTool` executes searches against `VectorStore` which handles:
     - Fuzzy course name matching via semantic search on course_catalog
     - Content search with optional course/lesson filtering
   - Results are formatted with metadata and returned to Claude for final response synthesis

### Key Component Responsibilities

**RAGSystem (rag_system.py)**: Main orchestrator that coordinates all components. Handles document ingestion via `add_course_folder()` and query processing via `query()` method using tool-based architecture.

**VectorStore (vector_store.py)**: Manages ChromaDB with dual collections. The `search()` method is the unified interface that handles course name resolution, filter building, and content search. Uses `SearchResults` dataclass for consistent result handling.

**AIGenerator (ai_generator.py)**: Wraps Anthropic API with tool-calling support. The `_handle_tool_execution()` method manages the multi-turn conversation pattern required for tool use (initial request → tool execution → final response). System prompt instructs Claude to use tools only for course-specific questions.

**DocumentProcessor (document_processor.py)**: Parses course files expecting format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson N: [lesson title]
Lesson Link: [url]
[lesson content...]
```
Chunks are sentence-based with configurable overlap. First chunk of each lesson gets "Lesson N content:" prefix.

**ToolManager & CourseSearchTool (search_tools.py)**: Implements tool pattern with `Tool` abstract base class. `CourseSearchTool.execute()` calls `VectorStore.search()` and formats results with course/lesson context headers. Tracks last_sources for UI display.

**SessionManager (session_manager.py)**: Maintains conversation history per session. History is formatted as text and passed to Claude's system prompt for context (limited to `MAX_HISTORY` exchanges).

### Configuration (config.py)

Key settings:
- `CHUNK_SIZE`: 800 chars (balance between context and precision)
- `CHUNK_OVERLAP`: 100 chars (maintains continuity across chunks)
- `MAX_RESULTS`: 5 results per search
- `MAX_HISTORY`: 2 conversation exchanges retained
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence transformers)

### Document Structure Expectations

Course files in `docs/` must follow this format:
- First 3 lines: Course metadata (Title, Link, Instructor)
- Lesson markers: "Lesson N: [title]" followed by optional "Lesson Link: [url]"
- Content between lesson markers becomes that lesson's content

Documents are loaded on startup via `app.py` startup event. Existing courses are skipped to prevent duplicates.

### Frontend Integration

The frontend (`frontend/`) is served as static files via FastAPI's `StaticFiles` with `DevStaticFiles` class adding no-cache headers for development. API endpoints:
- `POST /api/query`: Submit queries with optional session_id
- `GET /api/courses`: Get course catalog statistics

## Development Notes

- ChromaDB persists data in `backend/chroma_db/` directory
- The system avoids re-processing courses by checking existing titles before ingestion
- Tool-calling enables Claude to determine when semantic search is needed vs. answering from general knowledge
- The `SearchResults` dataclass provides consistent error handling across the vector store
- Course name matching uses semantic search for fuzzy matching ("MCP" can match "Introduction to MCP")
