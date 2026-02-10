# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. The system uses ChromaDB for vector storage, OpenAI's GPT-4o-mini API with tool-calling for AI responses, and OpenAI's text-embedding-3-small for embeddings. The frontend is vanilla HTML/CSS/JS served by FastAPI.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables (required)
# Create .env file with:
OPENAI_API_KEY=your_key_here
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

**Programming Paradigm**: The backend uses **procedural programming** with module-level state and functions rather than OOP classes. Each module maintains its own state and exposes functions for initialization and operations.

### Core Data Flow

1. **Document Processing** → **Vector Storage** → **AI Query with Tools** → **Response**
   - Course documents (txt/pdf/docx) are processed into structured course dictionaries with lesson metadata
   - Content is chunked and stored in ChromaDB with two collections:
     - `course_catalog`: Course metadata (title, instructor, lessons) for semantic course name matching
     - `course_content`: Actual course content chunks with course_title and lesson_number filters

2. **Query Processing Uses Tool-Calling Pattern**:
   - User queries go to `rag_system.query()` → `ai_generator.generate_response()`
   - GPT-4o-mini is provided with `search_course_content` tool definition
   - GPT-4o-mini decides when to search and extracts search parameters (query, course_name, lesson_number)
   - `search_tools.execute_course_search()` calls `vector_store.search()` which handles:
     - Fuzzy course name matching via semantic search on course_catalog
     - Content search with optional course/lesson filtering
   - Results are formatted with metadata and returned to GPT-4o-mini for final response synthesis

### Key Component Responsibilities

**rag_system.py**: Main orchestrator module that coordinates all components. Uses module-level initialization flag to ensure single setup. Key functions:
- `initialize_rag_system()`: Sets up all components (vector store, AI generator, session manager, search tools)
- `add_course_folder()`: Ingests documents from a directory
- `query()`: Processes user queries using tool-based architecture

**vector_store.py**: Manages ChromaDB with dual collections using module-level state (`_client`, `_course_catalog`, `_course_content`). Key functions:
- `initialize_vector_store()`: Sets up ChromaDB and embedding function
- `search()`: Unified interface that handles course name resolution, filter building, and content search
- Returns search results as dictionaries with documents, metadata, and distances

**ai_generator.py**: Wraps OpenAI API with tool-calling support using module-level state (`_client`, `_model`). Key functions:
- `initialize_ai_generator()`: Sets up OpenAI client
- `generate_response()`: Handles multi-turn conversation pattern for tool use (initial request → tool execution → final response)
- System prompt instructs GPT-4o-mini to use tools only for course-specific questions

**document_processor.py**: Parses course files expecting format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson N: [lesson title]
Lesson Link: [url]
[lesson content...]
```
Chunks are sentence-based with configurable overlap. First chunk of each lesson gets "Lesson N content:" prefix.

**search_tools.py**: Implements tool execution pattern using module-level state (`_tools`, `_last_sources`). Key functions:
- `get_course_search_tool_definition()`: Returns OpenAI tool definition for course search
- `execute_course_search()`: Calls `vector_store.search()` and formats results with course/lesson context headers
- `get_last_sources()`: Retrieves sources from last search for UI display
- Tracks last_sources globally for frontend integration

**session_manager.py**: Maintains conversation history per session using module-level state (`_sessions` dict). Key functions:
- `create_session()`: Generates new session ID
- `add_message()` / `add_exchange()`: Adds messages to session history
- `get_conversation_history()`: Returns formatted text history passed to GPT-4o-mini's system prompt
- History limited to `MAX_HISTORY` exchanges (auto-trimmed)

### Configuration (config.py)

Key settings:
- `CHUNK_SIZE`: 800 chars (balance between context and precision)
- `CHUNK_OVERLAP`: 100 chars (maintains continuity across chunks)
- `MAX_RESULTS`: 5 results per search
- `MAX_HISTORY`: 2 conversation exchanges retained
- `OPENAI_MODEL`: "gpt-4o-mini"
- `OPENAI_EMBEDDING_MODEL`: "text-embedding-3-small"

### Document Structure Expectations

Course files in `docs/` must follow this format:
- First 3 lines: Course metadata (Title, Link, Instructor)
- Lesson markers: "Lesson N: [title]" followed by optional "Lesson Link: [url]"
- Content between lesson markers becomes that lesson's content

Documents are loaded on startup via `app.py` startup event. Existing courses are skipped to prevent duplicates.

### Frontend Integration

The frontend (`frontend/`) is served as static files via FastAPI's `StaticFiles` with `DevStaticFiles` class adding no-cache headers for development.

**UI Layout:**
- Left sidebar with '+ New Chat' button, course statistics, and suggested questions
- Main chat area with message history and input field
- '+ New Chat' button clears conversation and resets session for fresh start

**API Endpoints:**
- `POST /api/query`: Submit queries with optional session_id
- `GET /api/courses`: Get course catalog statistics

## Development Notes

- **Procedural architecture**: All backend modules use module-level state and functions instead of OOP classes
- ChromaDB persists data in `backend/chroma_db/` directory
- The system avoids re-processing courses by checking existing titles before ingestion
- Tool-calling enables GPT-4o-mini to determine when semantic search is needed vs. answering from general knowledge
- Search results are returned as dictionaries for consistent error handling across the vector store
- Course name matching uses semantic search for fuzzy matching ("MCP" can match "Introduction to MCP")
- OpenAI embeddings (text-embedding-3-small) provide 1536 dimensions and improved quality over local models
- Frontend session management: Session IDs start as null and are created on first query; '+ New Chat' button resets to fresh session
