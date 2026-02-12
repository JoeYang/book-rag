#!/usr/bin/env python3
"""Harry Potter RAG System -- Agentic RAG with multi-search capability."""

import argparse
import asyncio
import sys

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCallResult,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Config ───────────────────────────────────────────────────────────────────

load_dotenv()

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
SIMILARITY_TOP_K = 10
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "harry_potter"

# ── LlamaIndex Settings ─────────────────────────────────────────────────────

Settings.llm = Anthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

# ── Custom Prompt ────────────────────────────────────────────────────────────

QA_PROMPT_TMPL = """\
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

Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

AGENT_SYSTEM_PROMPT = """\
You are a Harry Potter expert assistant with access to the full text of all \
seven Harry Potter books. You can search through them to answer questions.

IMPORTANT: You have tools to search the books. For complex questions, break \
them into multiple targeted searches. For example:
- "Name everyone who destroyed a horcrux" -> search for each horcrux separately \
(diary, ring, locket, cup, diadem, Nagini, Harry)
- "Compare Snape and Dumbledore" -> search for each character separately, then combine
- "What happened in the Battle of Hogwarts?" -> search for different aspects \
(who fought, who died, key events)

Always use multiple searches when a single search might miss scattered information. \
After gathering enough evidence from your searches, synthesize a thorough, \
natural-sounding answer. Be specific and cite which book events come from when possible."""

# ── ChromaDB ─────────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


def ingest() -> VectorStoreIndex:
    """Load documents from data/, chunk, embed, and store in ChromaDB."""
    print(f"Loading documents from {DATA_DIR}/...")
    reader = SimpleDirectoryReader(DATA_DIR, required_exts=[".txt"])
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document(s).")

    # Clear existing collection before re-ingesting
    if chroma_collection.count() > 0:
        print("Clearing existing collection...")
        chroma_client.delete_collection(COLLECTION_NAME)
        new_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        new_vs = ChromaVectorStore(chroma_collection=new_collection)
        storage_context = StorageContext.from_defaults(vector_store=new_vs)
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking and embedding (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    # Re-read count from the actual collection on disk
    updated_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    print(f"Ingestion complete. Stored {updated_collection.count()} chunks.")
    return index


def load_existing() -> VectorStoreIndex:
    """Load index from existing ChromaDB collection."""
    count = chroma_collection.count()
    print(f"Found existing index with {count} chunks. Skipping ingestion.")
    print("(Run with --reingest to re-index.)")
    return VectorStoreIndex.from_vector_store(vector_store)


def build_agent(index: VectorStoreIndex) -> ReActAgent:
    """Build a ReAct agent with search tools over the HP index."""

    # Tool 1: Semantic search -- finds passages by meaning similarity
    search_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode="tree_summarize",
        text_qa_template=QA_PROMPT,
    )
    search_tool = QueryEngineTool(
        query_engine=search_engine,
        metadata=ToolMetadata(
            name="search_books",
            description=(
                "Search through all 7 Harry Potter books using semantic search. "
                "Input a specific search query and get a synthesized answer from "
                "the most relevant passages. Use targeted, specific queries for "
                "best results. For broad questions, make multiple calls with "
                "different specific queries."
            ),
        ),
    )

    # Tool 2: Direct passage retrieval -- returns raw chunks with scores
    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

    def retrieve_passages(query: str) -> str:
        """Retrieve raw text passages from the books with similarity scores."""
        nodes = retriever.retrieve(query)
        results = []
        for i, node in enumerate(nodes, 1):
            score = f"{node.score:.4f}" if node.score is not None else "N/A"
            source = node.node.metadata.get("file_name", "unknown")
            text = node.node.get_content()
            results.append(f"[{i}] Score: {score} | Source: {source}\n{text}")
        return "\n\n---\n\n".join(results)

    passage_tool = FunctionTool.from_defaults(
        fn=retrieve_passages,
        name="retrieve_passages",
        description=(
            "Retrieve raw text passages from the Harry Potter books. Returns "
            "the actual book text with similarity scores. Use this when you "
            "need to read the exact text from the books, verify specific "
            "quotes, or get details that the search_books tool might summarize "
            "away. Input a specific search query."
        ),
    )

    agent = ReActAgent(
        name="HarryPotterExpert",
        description="An expert on the Harry Potter book series",
        tools=[search_tool, passage_tool],
        llm=Settings.llm,
        verbose=True,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    return agent


async def run_agent_query(agent: ReActAgent, question: str) -> str:
    """Run a single agent query and stream the thinking process."""
    handler = agent.run(question)

    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            name = event.tool_name if hasattr(event, "tool_name") else "tool"
            print(f"  [Tool: {name}] done")
        elif isinstance(event, AgentOutput):
            if event.response and hasattr(event.response, "content"):
                pass  # final response handled below

    result = await handler
    return str(result)


def query_loop(agent: ReActAgent) -> None:
    """Interactive query loop with the agentic RAG system."""
    print("\n── Harry Potter RAG (Agentic Mode) ─────────────────────────")
    print("Ask a question about the Harry Potter books.")
    print("The agent will search the books multiple times if needed.")
    print("Commands: 'quit' (exit)")
    print("────────────────────────────────────────────────────────────\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\n  Thinking...\n")
        try:
            response = loop.run_until_complete(run_agent_query(agent, question))
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

    loop.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Harry Potter RAG System")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Clear and re-ingest all documents",
    )
    args = parser.parse_args()

    if args.reingest or chroma_collection.count() == 0:
        try:
            index = ingest()
        except ValueError as e:
            if "No files found" in str(e) or "is not a directory" in str(e):
                print(f"\nError: {e}")
                print(f"Place your Harry Potter .txt files in the '{DATA_DIR}/' directory.")
                sys.exit(1)
            raise
    else:
        index = load_existing()

    agent = build_agent(index)
    query_loop(agent)


if __name__ == "__main__":
    main()
