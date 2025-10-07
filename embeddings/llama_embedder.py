"""Llama-server wrapper for generating code embeddings via OpenAI-compatible API."""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import requests

from chunking.python_ast_chunker import CodeChunk


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: np.ndarray
    chunk_id: str
    metadata: Dict[str, Any]


class LlamaServerEmbedder:
    """Wrapper for llama-server to generate code embeddings via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:10101",
        model_name: str = "nomic-embed-text",
        api_key: Optional[str] = "-",
        timeout: int = 30,
        batch_size: int = 32
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self._logger = logging.getLogger(__name__)
        self._embedding_dim = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test if llama-server is accessible."""
        try:
            # Try to get server info
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self._logger.info(f"Connected to llama-server at {self.base_url}")
            else:
                self._logger.warning(f"llama-server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            self._logger.warning(f"Could not connect to llama-server at {self.base_url}: {e}")
            self._logger.warning("Make sure llama-server is running with --embedding flag")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Call llama-server embeddings API (OpenAI-compatible)."""
        # llama-server uses OpenAI-compatible /v1/embeddings endpoint
        url = f"{self.base_url}/v1/embeddings"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # OpenAI-compatible payload format
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Extract embeddings from OpenAI-compatible response format
            embeddings = []
            for item in sorted(data.get('data', []), key=lambda x: x.get('index', 0)):
                embedding = item.get('embedding')
                if not embedding:
                    raise ValueError("No embedding in response item")
                embeddings.append(embedding)

            # Cache embedding dimension
            if embeddings and self._embedding_dim is None:
                self._embedding_dim = len(embeddings[0])
                self._logger.info(f"Embedding dimension: {self._embedding_dim}")

            return embeddings

        except requests.exceptions.RequestException as e:
            self._logger.error(f"Failed to get embeddings from llama-server: {e}")
            raise RuntimeError(f"Embedding API request failed: {e}")

    def create_embedding_content(self, chunk: CodeChunk, max_chars: int = 6000) -> str:
        """Create clean content for embedding generation with size limits."""
        # Prepare clean content without fabricated headers
        content_parts = []

        # Add docstring if available (important context for code understanding)
        docstring_budget = 300
        if chunk.docstring:
            # Keep docstring but limit length to stay within token budget
            docstring = chunk.docstring[:docstring_budget] + "..." if len(chunk.docstring) > docstring_budget else chunk.docstring
            content_parts.append(f'"""{docstring}"""')

        # Calculate remaining budget for code content
        docstring_len = len(content_parts[0]) if content_parts else 0
        remaining_budget = max_chars - docstring_len - 10  # small buffer

        # Add the actual code content, truncating if necessary
        if len(chunk.content) <= remaining_budget:
            content_parts.append(chunk.content)
        else:
            # Smart truncation: try to keep function signature and important parts
            lines = chunk.content.split('\n')
            if len(lines) > 3:
                # Keep first few lines (signature) and last few lines (return/conclusion)
                head_lines = []
                tail_lines = []
                current_length = docstring_len

                # Add head lines (function signature, early logic)
                for i, line in enumerate(lines[:min(len(lines)//2, 20)]):
                    if current_length + len(line) + 1 > remaining_budget * 0.7:
                        break
                    head_lines.append(line)
                    current_length += len(line) + 1

                # Add tail lines (return statements, conclusions) if space remains
                remaining_space = remaining_budget - current_length - 20  # buffer for "..."
                for line in reversed(lines[-min(len(lines)//3, 10):]):
                    if len('\n'.join(tail_lines)) + len(line) + 1 > remaining_space:
                        break
                    tail_lines.insert(0, line)

                if tail_lines:
                    truncated_content = '\n'.join(head_lines) + '\n    # ... (truncated) ...\n' + '\n'.join(tail_lines)
                else:
                    truncated_content = '\n'.join(head_lines) + '\n    # ... (truncated) ...'
                content_parts.append(truncated_content)
            else:
                # For short chunks, just truncate at character limit
                content_parts.append(chunk.content[:remaining_budget] + "..." if len(chunk.content) > remaining_budget else chunk.content)

        return '\n'.join(content_parts)

    def embed_chunk(self, chunk: CodeChunk) -> EmbeddingResult:
        """Generate embedding for a single code chunk."""
        max_inputs = int(os.getenv('LLAMA_SERVER_MAX_INPUTS', os.getenv('CLAUDE_MCP_LLAMA_MAX_INPUTS', '2048')))
        content = self.create_embedding_content(chunk, max_chars=max_inputs)

        # Get embedding from llama-server
        embeddings = self._get_embeddings([content])
        embedding = np.array(embeddings[0], dtype=np.float32)

        # Create unique chunk ID
        chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
        if chunk.name:
            chunk_id += f":{chunk.name}"

        # Prepare metadata
        metadata = {
            'file_path': chunk.file_path,
            'relative_path': chunk.relative_path,
            'folder_structure': chunk.folder_structure,
            'chunk_type': chunk.chunk_type,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'name': chunk.name,
            'parent_name': chunk.parent_name,
            'docstring': chunk.docstring,
            'decorators': chunk.decorators,
            'imports': chunk.imports,
            'complexity_score': chunk.complexity_score,
            'tags': chunk.tags,
            'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        }

        return EmbeddingResult(
            embedding=embedding,
            chunk_id=chunk_id,
            metadata=metadata
        )

    def embed_chunks(self, chunks: List[CodeChunk], batch_size: Optional[int] = None) -> List[EmbeddingResult]:
        """Generate embeddings for multiple chunks with batching."""
        results = []

        effective_batch_size = self.batch_size if batch_size is None else max(1, batch_size)
        self._logger.info(
            f"Generating embeddings for {len(chunks)} chunks (batch_size={effective_batch_size})"
        )

        # Get max chars per chunk from env var
        max_chars_per_chunk = int(os.getenv('LLAMA_SERVER_MAX_INPUTS', os.getenv('CLAUDE_MCP_LLAMA_MAX_INPUTS', '2048')))

        # Process in batches for efficiency
        for i in range(0, len(chunks), effective_batch_size):
            batch = chunks[i:i + effective_batch_size]
            batch_contents = [self.create_embedding_content(chunk, max_chars=max_chars_per_chunk) for chunk in batch]

            # Respect llama-server API limit if provided via env var
            max_batch_inputs = max_chars_per_chunk
            if len(batch_contents) > max_batch_inputs:
                raise ValueError(
                    f"Batch size {len(batch_contents)} exceeds llama-server input limit ({max_batch_inputs})"
                )

            # Generate embeddings for batch
            try:
                batch_embeddings = self._get_embeddings(batch_contents)
            except RuntimeError as exc:
                if effective_batch_size == 1:
                    raise
                self._logger.warning(
                    "Embedding batch failed; retrying with batch_size=1 for remaining chunks: %s",
                    exc
                )
                return results + self.embed_chunks(chunks[i:], batch_size=1)

            # Create results
            for j, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
                if chunk.name:
                    chunk_id += f":{chunk.name}"

                metadata = {
                    'file_path': chunk.file_path,
                    'relative_path': chunk.relative_path,
                    'folder_structure': chunk.folder_structure,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'name': chunk.name,
                    'parent_name': chunk.parent_name,
                    'docstring': chunk.docstring,
                    'decorators': chunk.decorators,
                    'imports': chunk.imports,
                    'complexity_score': chunk.complexity_score,
                    'tags': chunk.tags,
                    'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                }

                results.append(EmbeddingResult(
                    embedding=np.array(embedding, dtype=np.float32),
                    chunk_id=chunk_id,
                    metadata=metadata
                ))

            if i + effective_batch_size < len(chunks):
                self._logger.info(f"Processed {i + effective_batch_size}/{len(chunks)} chunks")

        self._logger.info("Embedding generation completed")
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        embeddings = self._get_embeddings([query])
        return np.array(embeddings[0], dtype=np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the llama-server model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self._embedding_dim or "unknown (call embed first)",
            "base_url": self.base_url,
            "status": "connected"
        }

    def cleanup(self):
        """No cleanup needed for API-based embedder."""
        self._logger.info("Llama-server embedder cleanup (no-op)")

    def __del__(self):
        """Cleanup when object is destroyed."""
        pass
