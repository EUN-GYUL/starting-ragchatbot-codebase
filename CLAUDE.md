# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start using the shell script
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add <package_name>
```

### Code Quality
```bash
# Format code with Black and sort imports
./scripts/format.sh

# Check code quality (dry-run)
./scripts/check.sh

# Run tests with quality checks
./scripts/test.sh

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Environment Setup
- Requires `.env` file with `ANTHROPIC_API_KEY`
- Application runs on port 8000 by default
- Access: http://localhost:8000 (web UI) and http://localhost:8000/docs (API docs)

## Architecture Overview

### Core RAG Pipeline
This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a modular, orchestrated architecture:

1. **RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates all components
2. **DocumentProcessor** (`document_processor.py`) - Parses course files with specific format expectations
3. **VectorStore** (`vector_store.py`) - ChromaDB integration for semantic search
4. **AIGenerator** (`ai_generator.py`) - Claude API integration with tool-calling capabilities
5. **SessionManager** (`session_manager.py`) - Conversation history management

### Document Processing Flow
- **Input Format**: Structured course files with metadata headers (`Course Title:`, `Course Link:`, `Course Instructor:`) followed by lesson markers (`Lesson X:`)
- **Chunking Strategy**: Sentence-based chunking with 800-character limit and 100-character overlap
- **Context Enhancement**: Each chunk prefixed with course/lesson context for better retrieval

### Key Configuration
All settings centralized in `config.py`:
- **AI Model**: `claude-sonnet-4-20250514`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Chunk Parameters**: 800 chars with 100 overlap
- **Search Limits**: 5 max results, 2 conversation history turns

### Vector Database Collections
The ChromaDB vector store maintains two specialized collections:

**course_catalog:**
- Stores course metadata for name resolution and semantic course discovery
- Documents: Course titles (used as searchable text)
- Metadata per course: `title`, `instructor`, `course_link`, `lesson_count`, `lessons_json` (JSON string containing array of lessons with lesson_number, lesson_title, lesson_link)
- IDs: Course titles (used as unique identifiers)

**course_content:**
- Stores text chunks for semantic content search
- Documents: Enhanced text chunks with course/lesson context prefixes
- Metadata per chunk: `course_title`, `lesson_number`, `chunk_index`
- IDs: `{course_title}_{chunk_index}` format

### Data Flow
1. Course documents → DocumentProcessor → structured Course/Lesson objects
2. Course metadata → course_catalog collection
3. Content chunks → course_content collection with metadata
4. User queries → semantic search (with optional course/lesson filtering) → relevant chunks + conversation history
5. Combined context → Claude API with tools → generated response

### Tool Integration
- **CourseSearchTool**: Semantic search within the ToolManager framework
- **Tool Calling**: AIGenerator handles Claude's tool-calling protocol
- Search results integrated into response generation context

### Frontend Integration
- FastAPI backend with CORS enabled
- Static file serving for frontend assets
- RESTful API endpoints: `/api/query` and `/api/courses`
- Startup document loading from `docs/` directory

### Session Management
- Stateful conversation tracking with session IDs
- Automatic session creation for new conversations
- History truncation to maintain performance (configurable via MAX_HISTORY)
- use uv to run python files or add any dependency
- The vector database has two collections:
- course_catalog:
stores course titles for name resolution
metadata for each course: title, instructor, course_link, lesson_count, lessons_json (list of lessons: lesson_number, lesson_title, lesson_link)
- course_content:
stores text chunks for semantic search
metadata for each chunk: course_title, lesson_number, chunk_index
