# Harry Potter RAG System -- How It Works

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances an LLM's answers by first *retrieving* relevant text passages from a document collection, then feeding those passages as context to the LLM so it can generate a grounded answer.

Without RAG, an LLM can only rely on its training data (which may be vague, outdated, or wrong about specifics). With RAG, the LLM gets the actual source text to reference, leading to more accurate and detailed answers.

## RAG Approaches: Naive vs Agentic

### Naive RAG (what we started with)

```
Question  -->  Embed  -->  Retrieve top-K chunks  -->  Send to LLM  -->  Answer
```

One search, one answer. Simple but limited -- if the answer is scattered across multiple parts of the books (e.g., "who destroyed each horcrux?"), a single retrieval pass will miss things.

### Agentic RAG (what we use now)

```
Question  -->  Agent thinks  -->  Search #1  -->  thinks again  -->  Search #2
          -->  Search #3  -->  ...  -->  Synthesize final answer
```

A **ReAct agent** (Reason + Act) sits on top of the retrieval system. Instead of one search, it:

1. **Thinks** about what information it needs
2. **Acts** by calling a search tool with a targeted query
3. **Observes** the results
4. **Decides** if it needs more information
5. **Repeats** steps 1-4 until it has enough evidence
6. **Synthesizes** a final answer from all gathered information

For the question "Name everyone who destroyed a horcrux," the agent autonomously runs ~7 separate searches (one per horcrux) and combines all results into a complete answer.

## How This System Works (Step by Step)

### 1. Ingestion (one-time setup)

```
.txt files  -->  Chunking  -->  Embedding  -->  ChromaDB
```

- **Load**: `SimpleDirectoryReader` reads all `.txt` files from `data/`.
- **Chunk**: The `SentenceSplitter` breaks each book into overlapping text chunks (like cutting a book into index cards). This is necessary because embedding models have token limits and smaller passages are more precise for retrieval.
- **Embed**: Each chunk is converted into a 384-dimensional vector (a list of numbers) by the HuggingFace embedding model. Text with similar meaning produces vectors that are close together in this vector space.
- **Store**: The vectors and their source text are saved into ChromaDB, a persistent vector database on disk.

### 2. Agent Setup

The `ReActAgent` is created with two tools it can call:

| Tool | What It Does | When the Agent Uses It |
|------|-------------|----------------------|
| `search_books` | Semantic search that returns a synthesized answer from the top-K most relevant passages | General questions, getting summarized information |
| `retrieve_passages` | Returns raw book text with similarity scores | Verifying exact quotes, reading specific passages, getting details that `search_books` might summarize away |

The agent also receives a **system prompt** that instructs it to break complex questions into multiple targeted searches rather than trying to answer everything in one pass.

### 3. Query (every question you ask)

```
User question
     |
     v
ReAct Agent (Claude Sonnet 4.5)
     |
     +-- Thought: "I need to find who destroyed each horcrux separately"
     |
     +-- Action: search_books("who destroyed Tom Riddle's diary")
     |-- Observation: "Harry Potter stabbed it with a basilisk fang..."
     |
     +-- Action: search_books("who destroyed the ring horcrux")
     |-- Observation: "Dumbledore destroyed the ring..."
     |
     +-- Action: search_books("who destroyed Slytherin's locket")
     |-- Observation: "Ron destroyed the locket with the sword..."
     |
     +-- ... (more searches as needed) ...
     |
     +-- Thought: "I now have information about all horcrux destroyers"
     |
     v
Final synthesized answer
```

The agent decides on its own how many searches to make and what to search for. For simple questions ("What house is Harry in?") it might do just one search. For complex questions, it can make up to 15 iterations.

### 4. Response Modes (used by the `search_books` tool internally)

The `response_mode` controls how retrieved chunks are fed to the LLM within each search:

- **`compact`** (default): Stuffs as many chunks as possible into a single LLM call. Fast but can miss details if there are many chunks.
- **`tree_summarize`** (what we use): Processes chunks in batches, generates partial answers, then merges them hierarchically. Better for comprehensive "find everything" type questions. Uses more API calls but produces more thorough answers.
- **`refine`**: Feeds chunks one at a time, iteratively refining the answer. Thorough but slow.

---

## Parameters Reference

### Ingestion Parameters (require `--reingest` to take effect)

| Parameter | Current Value | What It Does |
|-----------|--------------|--------------|
| `CHUNK_SIZE` | `1024` | Maximum number of tokens per text chunk. **Larger** = more context per chunk but less precise retrieval. **Smaller** = more precise matching but fragments may lack context. |
| `CHUNK_OVERLAP` | `200` | Number of tokens that overlap between adjacent chunks. Prevents information from being split awkwardly at chunk boundaries. E.g., if a sentence spans two chunks, the overlap ensures both chunks contain the full sentence. |

### Query Parameters (take effect immediately, no re-ingestion needed)

| Parameter | Current Value | What It Does |
|-----------|--------------|--------------|
| `SIMILARITY_TOP_K` | `10` | Number of chunks retrieved per tool call. The agent makes multiple tool calls, so the effective total chunks considered is `top_k * number_of_searches`. |
| `temperature` | `0.3` | Controls LLM output randomness. `0.0` = deterministic, always picks the most likely word. `1.0` = very creative/random. `0.3` is a good balance for factual-but-natural responses. |
| `response_mode` | `tree_summarize` | How chunks are fed to the LLM within each `search_books` call (see Response Modes above). |
| `text_qa_template` | Custom prompt | The prompt template that wraps retrieved passages and your question before sending to the LLM. This shapes the tone and behavior of the answers. |

### Agent Parameters

| Parameter | Current Value | What It Does |
|-----------|--------------|--------------|
| `AGENT_SYSTEM_PROMPT` | Custom prompt | Instructs the agent on how to behave -- break complex questions into sub-searches, be thorough, cite book sources. |
| `verbose` | `True` | Prints the agent's thinking process and tool calls so you can see what it's doing. |

### Infrastructure Parameters

| Parameter | Current Value | What It Does |
|-----------|--------------|--------------|
| `DATA_DIR` | `data` | Directory containing `.txt` source files. |
| `CHROMA_DIR` | `chroma_db` | Directory where ChromaDB stores the vector index on disk. Persists across runs so you don't re-embed every time. |
| `COLLECTION_NAME` | `harry_potter` | Name of the ChromaDB collection. You could create multiple collections for different document sets. |
| `model` | `claude-sonnet-4-5-20250929` | The Anthropic Claude model used for both the agent reasoning and answer generation. |
| `embed_model` | `all-MiniLM-L6-v2` | The HuggingFace model used for converting text to vectors. Runs locally (free, no API calls). Produces 384-dimensional vectors. |

---

## Tuning Guide

| Problem | Try |
|---------|-----|
| Answers are too vague / missing details | Increase `SIMILARITY_TOP_K` (e.g., 15-20) |
| Answers include irrelevant info | Decrease `SIMILARITY_TOP_K` (e.g., 5) |
| Agent doesn't search enough | Update `AGENT_SYSTEM_PROMPT` to be more explicit about breaking questions down |
| Agent searches too many times (slow/expensive) | Simplify the system prompt, or increase `SIMILARITY_TOP_K` so each search finds more |
| Chunks feel fragmented / cut mid-sentence | Increase `CHUNK_OVERLAP` (e.g., 300) |
| Retrieval misses relevant passages | Decrease `CHUNK_SIZE` (e.g., 512) for more precise matching |
| Answers lack surrounding context | Increase `CHUNK_SIZE` (e.g., 1500) |
| Answers feel robotic | Increase `temperature` (e.g., 0.5) or adjust the prompt |
| Answers hallucinate facts | Decrease `temperature` (e.g., 0.1) |
| API costs too high | Lower `SIMILARITY_TOP_K`, simplify system prompt to reduce agent iterations |

---

## Architecture Diagram

```
                           hp-rag/
                           main.py
                              |
                +-------------+-------------+
                |                           |
          INGESTION                   QUERYING (Agentic)
                |                           |
     data/*.txt files               User types question
                |                           |
     SimpleDirectoryReader                  v
                |                    +-------------+
     SentenceSplitter                | ReAct Agent |  <-- Claude Sonnet 4.5
     (chunk_size=1024,               | (thinks +   |      with system prompt
      overlap=200)                   |  decides)   |
                |                    +------+------+
     HuggingFace Embedding                 |
     (all-MiniLM-L6-v2)            Can call tools:
                |                          |
     ChromaDB (persist to disk)     +------+------+
     chroma_db/                     |             |
                              search_books   retrieve_passages
                                    |             |
                              Embeds query   Embeds query
                                    |             |
                              ChromaDB        ChromaDB
                              top-K search    top-K search
                                    |             |
                              tree_summarize  Raw text +
                              via Claude      scores
                                    |             |
                                    +------+------+
                                           |
                                    Agent combines
                                    all results
                                           |
                                    Final answer
```

---

## Cost Considerations

Agentic RAG uses more API calls than naive RAG because:

1. **Each tool call** to `search_books` triggers an LLM call internally (to synthesize the retrieved passages)
2. **The agent itself** uses LLM calls for its reasoning steps (Thought -> Action -> Observation loop)
3. A single question might result in **5-10+ LLM calls** depending on complexity

For simple factual questions, the agent typically makes 1-2 searches. For complex "find everything" questions, it may make 5-7+. Budget accordingly with your Anthropic API credits.
