"""FastMCP server for Claude Code integration."""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("FastMCP not found. Install with: uv add mcp fastmcp")
    sys.exit(1)

from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import CodeEmbedder
from embeddings.llama_embedder import LlamaServerEmbedder
from search.indexer import CodeIndexManager
from search.searcher import IntelligentSearcher


# Initialize logging with more verbose output for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable FastMCP internal logging
logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("fastmcp").setLevel(logging.DEBUG)

# Initialize MCP server
mcp = FastMCP("Code Search")

# Global components (will be initialized when first needed)
_embedder = None
_index_manager = None
_searcher = None
_storage_dir = None
_current_project = None  # Track which project is currently active
_model_preload_task_started = False


def get_storage_dir() -> Path:
    """Get or create base storage directory."""
    global _storage_dir
    if _storage_dir is None:
        # Use a default location or environment variable
        storage_path = os.getenv('CODE_SEARCH_STORAGE', str(Path.home() / '.claude_code_search'))
        _storage_dir = Path(storage_path)
        _storage_dir.mkdir(parents=True, exist_ok=True)
    return _storage_dir


def get_project_storage_dir(project_path: str) -> Path:
    """Get or create project-specific storage directory."""
    base_dir = get_storage_dir()
    
    # Create a safe directory name from project path
    import hashlib
    from datetime import datetime
    
    project_path = Path(project_path).resolve()
    project_name = project_path.name
    project_hash = hashlib.md5(str(project_path).encode()).hexdigest()[:8]
    
    # Use project name + hash to ensure uniqueness and readability
    project_dir = base_dir / "projects" / f"{project_name}_{project_hash}"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Store project info
    project_info_file = project_dir / "project_info.json"
    if not project_info_file.exists():
        project_info = {
            "project_name": project_name,
            "project_path": str(project_path),
            "project_hash": project_hash,
            "created_at": datetime.now().isoformat()
        }
        with open(project_info_file, 'w') as f:
            json.dump(project_info, f, indent=2)
    
    return project_dir


def ensure_project_indexed(project_path: str) -> bool:
    """Check if project is indexed, auto-index if it's the current directory and has Python files."""
    try:
        project_dir = get_project_storage_dir(project_path)
        index_dir = project_dir / "index"
        
        # Check if already indexed
        if index_dir.exists() and (index_dir / "code.index").exists():
            return True
        
        # Auto-index current directory if it has Python files
        project_path_obj = Path(project_path)
        if project_path_obj == Path.cwd() and list(project_path_obj.glob("**/*.py")):
            logger.info(f"Auto-indexing current directory: {project_path}")
            result = index_directory(project_path)
            result_data = json.loads(result)
            return "error" not in result_data
        
        return False
        
    except Exception as e:
        logger.warning(f"Failed to check/auto-index project {project_path}: {e}")
        return False


def get_embedder():
    """Lazy initialization of embedder."""
    global _embedder
    if _embedder is None:
        # Check if llama-server should be used
        use_llama = os.getenv('USE_LLAMA_SERVER', 'true').lower() == 'true'
        llama_url = os.getenv('LLAMA_SERVER_URL', 'http://localhost:10101')

        if use_llama:
            # Get configurable timeout and batch size
            timeout = int(os.getenv('LLAMA_SERVER_TIMEOUT', os.getenv('CLAUDE_MCP_LLAMA_TIMEOUT', '60')))
            batch_size = int(os.getenv('LLAMA_SERVER_BATCH_SIZE', os.getenv('CLAUDE_MCP_LLAMA_BATCH', '32')))
            logger.info(
                f"Using llama-server embedder at {llama_url} (timeout: {timeout}s, batch_size: {batch_size})"
            )
            _embedder = LlamaServerEmbedder(base_url=llama_url, timeout=timeout, batch_size=batch_size)
        else:
            cache_dir = get_storage_dir() / "models"
            cache_dir.mkdir(exist_ok=True)
            _embedder = CodeEmbedder(cache_dir=str(cache_dir))
            logger.info("Using sentence-transformers embedder")
    return _embedder


def _maybe_start_model_preload() -> None:
    """Preload the embedding model in the background to avoid cold-start delays."""
    global _model_preload_task_started
    if _model_preload_task_started:
        return
    _model_preload_task_started = True

    async def _preload():
        try:
            logger.info("Starting background model preload")
            embedder = get_embedder()
            # Only preload if using sentence-transformers (llama-server doesn't need preload)
            if isinstance(embedder, CodeEmbedder):
                _ = embedder.model
            logger.info("Background model preload completed")
        except Exception as e:
            logger.warning(f"Background model preload failed: {e}")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_preload())
        else:
            loop.run_until_complete(_preload())
    except Exception as e:
        logger.debug(f"Model preload scheduling skipped: {e}")


def get_index_manager(project_path: str = None) -> CodeIndexManager:
    """Get index manager for specific project or current project."""
    global _index_manager, _current_project
    
    # If no project specified, use current project or default to cwd
    if project_path is None:
        if _current_project is None:
            # Auto-detect current working directory as default project
            project_path = os.getcwd()
            logger.info(f"No active project found. Using current working directory: {project_path}")
            # Try to auto-index if current directory has Python files
            ensure_project_indexed(project_path)
        else:
            project_path = _current_project
    
    # If switching projects, reset the index manager
    if _current_project != project_path:
        _index_manager = None
        _current_project = project_path
    
    if _index_manager is None:
        project_dir = get_project_storage_dir(project_path)
        index_dir = project_dir / "index"
        index_dir.mkdir(exist_ok=True)
        _index_manager = CodeIndexManager(str(index_dir))
        logger.info(f"Index manager initialized for project: {Path(project_path).name}")
    
    return _index_manager


def get_searcher(project_path: str = None) -> IntelligentSearcher:
    """Get searcher for specific project or current project."""
    global _searcher, _current_project
    
    # Auto-detect project path if not provided
    if project_path is None and _current_project is None:
        project_path = os.getcwd()
        logger.info(f"No active project found. Using current working directory: {project_path}")
        # Try to auto-index if current directory has Python files
        ensure_project_indexed(project_path)
    
    # If switching projects, reset the searcher
    if _current_project != project_path or _searcher is None:
        _searcher = IntelligentSearcher(
            get_index_manager(project_path),
            get_embedder()
        )
        logger.info(f"Searcher initialized for project: {Path(_current_project).name if _current_project else 'unknown'}")
    
    return _searcher


@mcp.tool()
def search_code(
    query: str,
    k: int = 5,
    search_mode: str = "auto",
    file_pattern: str = None,
    chunk_type: str = None,
    include_context: bool = True,
    auto_reindex: bool = True,
    max_age_minutes: float = 5
) -> str:
    """
    PREFERRED: Use this tool for code analysis and understanding tasks. Provides semantic search 
    using EmbeddingGemma-300m model for intelligent code discovery based on functionality rather 
    than just text patterns.

    WHEN TO USE:
    - Understanding how specific functionality is implemented
    - Finding similar patterns across the codebase  
    - Discovering related functions/classes by behavior
    - Searching for code that handles specific use cases
    - Analyzing architectural patterns and relationships

    WHEN NOT TO USE:
    - Simple exact text/pattern matching (use generic grep/search tools instead)
    - Searching non-Python files (this tool only works with Python codebases)
    - When the codebase hasn't been indexed yet (use index_directory first)

    Args:
        query: Natural language description of functionality you're looking for
               Examples: "error handling", "user authentication", "database connection"
        k: Number of results to return (default: 5, max recommended: 20)
        search_mode: Currently supports "semantic" mode only 
        file_pattern: Filter by filename/path pattern (e.g., "auth", "utils", "models")
        chunk_type: Filter by code structure - "function", "class", "method", or None for all
        include_context: Include similar chunks and relationships (default: True, recommended)
        auto_reindex: Automatically reindex if index is stale (default: True)
        max_age_minutes: Maximum age of index before auto-reindex (default: 5 minutes)

    Returns:
        JSON with semantically ranked results including similarity scores, file paths, 
        line numbers, code previews, semantic tags, and contextual relationships
    """
    try:
        logger.info(f"🔍 MCP REQUEST: search_code(query='{query}', k={k}, mode='{search_mode}', file_pattern={file_pattern}, chunk_type={chunk_type})")
        
        # Auto-reindex if enabled and index is stale
        if auto_reindex and _current_project:
            from search.incremental_indexer import IncrementalIndexer
            
            logger.info(f"Checking if index needs refresh (max age: {max_age_minutes} minutes)")
            
            # Initialize incremental indexer
            index_manager = get_index_manager(_current_project)
            embedder = get_embedder()
            chunker = MultiLanguageChunker(_current_project)
            
            incremental_indexer = IncrementalIndexer(
                indexer=index_manager,
                embedder=embedder,
                chunker=chunker
            )
            
            # Auto-reindex if needed (this is very fast if no changes)
            reindex_result = incremental_indexer.auto_reindex_if_needed(
                _current_project,
                max_age_minutes=max_age_minutes
            )
            
            if reindex_result.files_modified > 0 or reindex_result.files_added > 0:
                logger.info(f"Auto-reindexed: {reindex_result.files_added} added, {reindex_result.files_modified} modified, took {reindex_result.time_taken:.2f}s")
                # Refresh searcher after reindex
                global _searcher
                _searcher = None  # Reset to force reload
        
        searcher = get_searcher()
        logger.info(f"Current project: {_current_project}")
        
        # Debug: Check index stats
        index_stats = searcher.index_manager.get_stats()
        logger.info(f"Index contains {index_stats.get('total_chunks', 0)} chunks")
        
        # Build filters
        filters = {}
        if file_pattern:
            filters['file_pattern'] = [file_pattern]
        if chunk_type:
            filters['chunk_type'] = chunk_type
        
        logger.info(f"Search filters: {filters}")
        
        # Perform search
        context_depth = 1 if include_context else 0
        logger.info(f"Calling searcher.search with query='{query}', k={k}, mode={search_mode}")
        results = searcher.search(
            query=query,
            k=k,
            search_mode=search_mode,
            context_depth=context_depth,
            filters=filters if filters else None
        )
        logger.info(f"Search returned {len(results)} results")
        
        #
        # Previous verbose response structure (reference only)
        # {
        #   "query": str,
        #   "total_results": int,
        #   "results": [
        #     {
        #       "file_path": str,        # relative path
        #       "full_path": str,        # absolute path
        #       "lines": "start-end",
        #       "chunk_type": str,
        #       "name": str | null,
        #       "parent_name": str | null,
        #       "similarity_score": float,
        #       "content_preview": str,  # multi-line preview
        #       "docstring": str | null,
        #       "tags": [str],
        #       "folder_structure": str | null,
        #       "context": {             # only when include_context=True
        #         "similar_chunks": [
        #           { "chunk_id": str, "similarity": float, "name": str | null, "chunk_type": str }
        #         ],
        #         "file_context": { "total_chunks_in_file": int, "folder_path": str | null }
        #       }
        #     }
        #   ]
        # }
        #
        # Response structure (compact)
        # {
        #   "query": str,
        #   "results": [
        #     {
        #       "file": str,            # path relative to project root
        #       "lines": "start-end",  # 1-based inclusive line range
        #       "kind": str,            # chunk type: function | method | class | interface | enum | script | style | ...
        #       "score": float,         # similarity score in [0,1], rounded to 2 decimals
        #       "chunk_id": str,        # stable id: "relative_path:start-end:kind[:name]"
        #       "name": str,            # optional chunk name (function/class/interface)
        #       "snippet": str          # optional short signature/minisnippet (<=160 chars)
        #     }
        #   ]
        # }
        #
        # Notes:
        # - Fields intentionally omitted for token efficiency: full_path, folder_structure, parent_name, docstring, raw previews, context.
        # - Snippet is derived from the first non-empty line of content_preview with whitespace compressed.
        # - "file" and "lines" are sufficient for downstream precise file reading.
        #
        # Compact, token-efficient formatting with a short snippet
        def make_snippet(preview: Optional[str]) -> str:
            if not preview:
                return ""
            for line in preview.split('\n'):
                s = line.strip()
                if s:
                    # Compress whitespace and cap length
                    snippet = ' '.join(s.split())
                    return (snippet[:157] + '...') if len(snippet) > 160 else snippet
            return ""

        formatted_results = []
        for result in results:
            item = {
                'file': result.relative_path,
                'lines': f"{result.start_line}-{result.end_line}",
                'kind': result.chunk_type,
                'score': round(result.similarity_score, 2),
                'chunk_id': result.chunk_id
            }
            if result.name:
                item['name'] = result.name
            snippet = make_snippet(result.content_preview)
            if snippet:
                item['snippet'] = snippet
            formatted_results.append(item)

        response = {
            'query': query,
            'results': formatted_results
        }

        # Minified JSON to reduce tokens
        return json.dumps(response, separators=(",", ":"))
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


@mcp.tool()
def index_directory(
    directory_path: str,
    project_name: str = None,
    file_patterns: List[str] = None,
    incremental: bool = True
) -> str:
    """
    SETUP REQUIRED: Index a codebase for semantic search. Must run this before 
    using search_code on a new project. Supports Python, JavaScript, TypeScript, JSX, TSX, and Svelte.

    WHEN TO USE:
    - First time analyzing a new codebase
    - After significant code changes that might affect search results
    - When switching to a different project

    PROCESS:
    - Uses Merkle trees to detect file changes efficiently
    - Only reprocesses changed/new files (incremental mode)
    - Parses code files using AST (Python) and tree-sitter (JS/TS/JSX/TSX/Svelte)
    - Chunks code into semantic units (functions, classes, methods)
    - Generates 768-dimensional embeddings using EmbeddingGemma-300m
    - Builds FAISS vector index for fast similarity search
    - Stores metadata in SQLite database

    Args:
        directory_path: Absolute path to project root
        project_name: Optional name for organization (defaults to directory name)
        file_patterns: File patterns to include (default: all supported extensions)
        incremental: Use incremental indexing if snapshot exists (default: True)

    Returns:
        JSON with indexing statistics and success status

    Note: Incremental indexing is much faster for updates. Full reindex on first run.
    """
    try:
        from search.incremental_indexer import IncrementalIndexer
        
        # Start model preload early to overlap with Merkle/IO work
        _maybe_start_model_preload()

        directory_path = Path(directory_path).resolve()
        if not directory_path.exists():
            return json.dumps({"error": f"Directory does not exist: {directory_path}"})
        
        if not directory_path.is_dir():
            return json.dumps({"error": f"Path is not a directory: {directory_path}"})
        
        project_name = project_name or directory_path.name
        logger.info(f"Indexing directory: {directory_path} (incremental={incremental})")
        
        # Initialize incremental indexer
        index_manager = get_index_manager(str(directory_path))
        embedder = get_embedder()
        chunker = MultiLanguageChunker(str(directory_path))
        
        incremental_indexer = IncrementalIndexer(
            indexer=index_manager,
            embedder=embedder,
            chunker=chunker
        )
        
        # Perform indexing
        result = incremental_indexer.incremental_index(
            str(directory_path),
            project_name,
            force_full=not incremental
        )
        
        # Get updated statistics
        stats = incremental_indexer.get_indexing_stats(str(directory_path))
        
        response = {
            "success": result.success,
            "directory": str(directory_path),
            "project_name": project_name,
            "incremental": incremental and result.files_modified > 0,
            "files_added": result.files_added,
            "files_removed": result.files_removed,
            "files_modified": result.files_modified,
            "chunks_added": result.chunks_added,
            "chunks_removed": result.chunks_removed,
            "time_taken": round(result.time_taken, 2),
            "index_stats": stats
        }
        
        if result.error:
            response["error"] = result.error
        
        logger.info(f"Indexing completed. Added: {result.files_added}, Modified: {result.files_modified}, Time: {result.time_taken:.2f}s")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Indexing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


@mcp.tool()
def find_similar_code(
    chunk_id: str,
    k: int = 5
) -> str:
    """
    SPECIALIZED: Find code chunks functionally similar to a specific reference chunk.
    Use this when you want to discover code that does similar things to a known piece of code.

    WHEN TO USE:
    - Finding alternative implementations of the same functionality
    - Discovering code duplication or similar patterns
    - Understanding how a pattern is used throughout the codebase
    - Refactoring: finding related code that might need similar changes

    WORKFLOW: 
    1. First use search_code to find a reference chunk
    2. Use the chunk_id from search results with this tool
    3. Get ranked list of functionally similar code

    Args:
        chunk_id: ID from search_code results (format: "file:lines:type:name")
        k: Number of similar chunks to return (default: 5)

    Returns:
        JSON with reference chunk info and ranked similar chunks with similarity scores
    """
    try:
        searcher = get_searcher()
        results = searcher.find_similar_to_chunk(chunk_id, k=k)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                'file_path': result.relative_path,
                'lines': f"{result.start_line}-{result.end_line}",
                'chunk_type': result.chunk_type,
                'name': result.name,
                'similarity_score': round(result.similarity_score, 3),
                'content_preview': result.content_preview,
                'tags': result.tags
            })
        
        response = {
            'reference_chunk': chunk_id,
            'similar_chunks': formatted_results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Similar code search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


@mcp.tool()
def get_index_status() -> str:
    """
    Get current status and statistics of the search index.
    
    Returns:
        JSON string with index statistics and model information
    """
    try:
        # Get index stats (safe to call even if not initialized)
        index_manager = get_index_manager()
        stats = index_manager.get_stats()
        
        # Get model info if embedder is initialized
        model_info = {"status": "not_loaded"}
        if _embedder is not None:
            model_info = _embedder.get_model_info()
        
        response = {
            "index_statistics": stats,
            "model_information": model_info,
            "storage_directory": str(get_storage_dir())
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Status check failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


@mcp.tool()
def list_projects() -> str:
    """
    List all indexed projects with their information.
    
    Returns:
        JSON string with list of projects and their metadata
    """
    try:
        base_dir = get_storage_dir()
        projects_dir = base_dir / "projects"
        
        if not projects_dir.exists():
            return json.dumps({
                "projects": [],
                "count": 0,
                "message": "No projects indexed yet"
            })
        
        projects = []
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                info_file = project_dir / "project_info.json"
                if info_file.exists():
                    with open(info_file) as f:
                        project_info = json.load(f)
                    
                    # Add index statistics if available
                    stats_file = project_dir / "index" / "stats.json"
                    if stats_file.exists():
                        with open(stats_file) as f:
                            stats = json.load(f)
                        project_info["index_stats"] = stats
                    
                    projects.append(project_info)
        
        return json.dumps({
            "projects": projects,
            "count": len(projects),
            "current_project": _current_project
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def switch_project(project_path: str) -> str:
    """
    Switch to a different indexed project for searching.
    
    Args:
        project_path: Path to the project directory
    
    Returns:
        JSON string with switch result
    """
    try:
        global _current_project, _index_manager, _searcher
        
        project_path = Path(project_path).resolve()
        if not project_path.exists():
            return json.dumps({"error": f"Project path does not exist: {project_path}"})
        
        # Check if project is indexed
        project_dir = get_project_storage_dir(str(project_path))
        index_dir = project_dir / "index"
        
        if not index_dir.exists() or not (index_dir / "code.index").exists():
            return json.dumps({
                "error": f"Project not indexed: {project_path}",
                "suggestion": f"Run index_directory('{project_path}') first"
            })
        
        # Reset global state to switch projects
        _current_project = str(project_path)
        _index_manager = None
        _searcher = None
        
        # Get project info
        info_file = project_dir / "project_info.json"
        project_info = {}
        if info_file.exists():
            with open(info_file) as f:
                project_info = json.load(f)
        
        logger.info(f"Switched to project: {project_path.name}")
        
        return json.dumps({
            "success": True,
            "message": f"Switched to project: {project_path.name}",
            "project_info": project_info
        })
        
    except Exception as e:
        logger.error(f"Error switching project: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def index_test_project() -> str:
    """
    Index the built-in Python test project for demonstration purposes.
    
    This indexes a sample Python project with authentication, database, API, and utility modules
    to demonstrate the code search capabilities. Useful for trying out the system.
    
    Returns:
        JSON string with indexing results and statistics
    """
    try:
        logger.info("Indexing built-in test project")
        
        # Get the test project path
        server_dir = Path(__file__).parent
        test_project_path = server_dir.parent / "tests" / "test_data" / "python_project"
        
        if not test_project_path.exists():
            return json.dumps({
                "success": False,
                "error": "Test project not found. The sample project may not be available."
            })
        
        # Use the regular index_directory function
        result = index_directory(str(test_project_path))
        result_data = json.loads(result)
        
        # Add demo information
        if "error" not in result_data:
            result_data["demo_info"] = {
                "project_type": "Sample Python Project",
                "includes": [
                    "Authentication module (user login, password hashing)",
                    "Database module (connections, queries, transactions)", 
                    "API module (HTTP handlers, request validation)",
                    "Utilities (helpers, validation, configuration)"
                ],
                "sample_searches": [
                    "user authentication functions",
                    "database connection code",
                    "HTTP API handlers", 
                    "input validation",
                    "error handling patterns"
                ]
            }
        
        return json.dumps(result_data, indent=2)
        
    except Exception as e:
        logger.error(f"Error indexing test project: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@mcp.tool()
def clear_index() -> str:
    """
    Clear the entire search index and metadata for the current project.
    
    Returns:
        JSON string confirming the operation
    """
    try:
        # Use current project or raise error
        if _current_project is None:
            return json.dumps({"error": "No project is currently active. Use index_directory() to index a project first."})
        
        index_manager = get_index_manager()
        index_manager.clear_index()
        
        response = {
            "success": True,
            "message": "Search index cleared successfully"
        }
        
        logger.info("Search index cleared")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Clear index failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


@mcp.resource("search://stats")
def get_search_statistics() -> str:
    """
    Get detailed search index statistics.
    
    Returns:
        Detailed statistics about indexed files, chunks, and search performance
    """
    try:
        index_manager = get_index_manager()
        stats = index_manager.get_stats()
        
        return json.dumps(stats, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to get statistics: {str(e)}"})


@mcp.prompt()
def search_help() -> str:
    """
    Get help on how to use the code search tools effectively.
    
    Returns:
        Detailed help text with examples
    """
    help_text = """
# Code Search Tool Help

This tool provides semantic search capabilities for Python codebases using AI embeddings.

## Available Tools:

### 1. search_code(query, k=5, ...)
Search for code using natural language queries.

Examples:
- "Find authentication functions"
- "Show database connection code"
- "Find error handling patterns"
- "Look for API endpoint definitions"

### 2. index_directory(directory_path, ...)
Index a Python project for search.

Example:
- index_directory("/path/to/my/project")

### 3. get_index_status()
Check current index statistics and model status.

### 4. find_similar_code(chunk_id, k=5)
Find code similar to a specific chunk.

## Search Tips:

1. **Natural Language**: Use descriptive phrases
   - Good: "Find functions that handle user authentication"
   - Better: "authentication login user validation"

2. **Specific Terms**: Include technical terms
   - "database query connection"
   - "API endpoint route handler"

3. **Filters**: Use filters to narrow results
   - file_pattern: "auth" (files containing "auth")
   - chunk_type: "function", "class", "method"

## Getting Started:

1. First, index your codebase:
   ```
   index_directory("/path/to/your/python/project")
   ```

2. Then search:
   ```
   search_code("find authentication code", k=10)
   ```

The tool uses advanced AST parsing to understand code structure and creates intelligent chunks that preserve function and class boundaries.
"""
    
    return help_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Search MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Map "http" to the correct FastMCP transport name
    transport = "sse" if args.transport == "http" else args.transport
    
    if transport in ["sse", "streamable-http"]:
        logger.info(f"Starting HTTP server on {args.host}:{args.port}")
        mcp.run(transport=transport)
    else:
        mcp.run(transport=transport)
   