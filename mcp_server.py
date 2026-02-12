#!/usr/bin/env python3
"""MCP server exposing Harry Potter RAG tools to AI clients.

Self-contained -- does not import from main.py. Uses the same ChromaDB
index that main.py creates, so run `python main.py --reingest` first if
the index doesn't exist yet.

Usage:
    python mcp_server.py                   # start MCP server (stdio)
    mcp dev mcp_server.py                  # interactive browser inspector
"""

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore
from mcp.server.fastmcp import FastMCP

# ── Paths (absolute so the server works regardless of cwd) ────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(PROJECT_DIR / "chroma_db")

# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
SIMILARITY_TOP_K = 10
COLLECTION_NAME = "harry_potter"

# ── Initialize LlamaIndex + ChromaDB ─────────────────────────────────────────

load_dotenv(PROJECT_DIR / ".env")

Settings.llm = Anthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

if chroma_collection.count() == 0:
    raise RuntimeError(
        "No chunks found in ChromaDB. Run `python main.py --reingest` first."
    )

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# ── Custom prompt (same as main.py) ──────────────────────────────────────────

QA_PROMPT = PromptTemplate("""\
You are a knowledgeable and enthusiastic Harry Potter expert. You have deep \
familiarity with all seven books and the wizarding world.

Below are relevant passages from the Harry Potter books:
-----
{context_str}
-----

Using these passages as your primary source, answer the following question. \
Be thorough and comprehensive -- look carefully through ALL provided passages \
for every relevant detail before answering. For questions asking you to list or \
enumerate things, make sure you find every instance across all passages. \
Synthesize information across multiple passages into a natural, conversational \
response. If the passages don't fully cover the answer, clearly state what is \
confirmed by the text and what isn't.

Question: {query_str}

Answer: """)

# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("Harry Potter RAG")


@mcp.tool()
def search_books(query: str) -> str:
    """Search through all 7 Harry Potter books using semantic search.

    Returns a synthesized answer from the most relevant passages.
    Use targeted, specific queries for best results.
    For broad questions, make multiple calls with different specific queries.
    """
    engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode="tree_summarize",
        text_qa_template=QA_PROMPT,
    )
    return str(engine.query(query))


@mcp.tool()
def retrieve_passages(query: str) -> str:
    """Retrieve raw text passages from the Harry Potter books with similarity scores.

    Use this when you need exact quotes, specific text, or details that
    search_books might summarize away.
    """
    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    nodes = retriever.retrieve(query)
    results = []
    for i, node in enumerate(nodes, 1):
        score = f"{node.score:.4f}" if node.score is not None else "N/A"
        source = node.node.metadata.get("file_name", "unknown")
        text = node.node.get_content()
        results.append(f"[{i}] Score: {score} | Source: {source}\n{text}")
    return "\n\n---\n\n".join(results)


@mcp.tool()
def collection_info() -> str:
    """Get metadata about the indexed book collection.

    Returns chunk count, embedding model, chunk size, and other settings.
    """
    info = {
        "collection_name": COLLECTION_NAME,
        "chunk_count": chroma_collection.count(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "similarity_top_k": SIMILARITY_TOP_K,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "claude-sonnet-4-5-20250929",
    }
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    mcp.run()
