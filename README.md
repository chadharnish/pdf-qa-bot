# PDF Question-Answering Bot

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions about a PDF document, with source page citations. Built as a learning project to understand RAG end-to-end.

## What it does

Loads a PDF, splits it into searchable chunks, embeds them into a local vector database, and uses a language model to answer questions grounded in the document's content. Includes page-level source attribution so every answer is traceable back to where it came from in the original document.

## Architecture
PDF → Pages → Filter index/TOC → Chunks (1000 chars, 200 overlap)
↓
Embedding model (all-MiniLM-L6-v2, 384-dim)
↓
FAISS vector store (cached to disk)
↓
Question → Embed → Top-K similar chunks → Prompt template
↓
LLM (Qwen 2.5 1.5B Instruct, local)
↓
Grounded answer + page citations

## Tech stack

- **Python 3.12** in a virtual environment
- **LangChain** — pipeline orchestration (loaders, splitters, prompts)
- **Hugging Face Transformers** — embedding and language models
- **sentence-transformers/all-MiniLM-L6-v2** — embeddings
- **FAISS** — local vector store with on-disk persistence
- **Qwen 2.5 1.5B Instruct** — local LLM (CPU inference)
- **pypdf** — PDF text extraction

## Setup

```bash
git clone https://github.com/chadharnish/pdf-qa-bot.git
cd pdf-qa-bot

# Create and activate virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install langchain langchain-community langchain-huggingface langchain-text-splitters pypdf sentence-transformers faiss-cpu python-dotenv transformers

# Add a Hugging Face token to .env
echo HUGGINGFACEHUB_API_TOKEN=hf_yourtoken > .env

# Drop a PDF named "document1.pdf" in the project root, then:
python app.py
```

First run downloads ~80MB (embedding model) plus ~3GB (Qwen 1.5B). Subsequent runs load from cache and start in seconds.

## Configuration

The top of `app.py` exposes a few knobs:

```python
LLM_MODE = "local"                         # "local" or "api"
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" # any compatible HF model
API_MODEL   = "HuggingFaceH4/zephyr-7b-beta"
```

## What I learned

- **Data quality dominates model quality.** As with any data project, garbage in = garbage out. It is best to clean the data as much as possible.
- **Test each layer in isolation.** It is best to test retrieval quality before plugging in the LLM saved hours of misdirected debugging.
- **The Hugging Face free Inference API is a moving target.** Free-tier model availability shifts and is unreliable.
- **Cache early, iterate fast.** Persisting the FAISS index turned a 30-second startup into under a second. I wish I would have done this first as it makes it quicker to test and debug.
- **Prompt instructions matter as much as retrieval.** Explicit "use only this context" and "say I don't know if it's not in the context" instructions reduce hallucinations substantially.

## Future improvements

- Hybrid retrieval combining semantic similarity with BM25 keyword search
- Cross-encoder re-ranking of the top-K retrieved chunks
- Cleanup pass for PDF typographic ligatures (e.g. `/T_h` → `Th`, `/f_i` → `fi`)
- Hash-based cache invalidation when chunking parameters change
- Web UI with Streamlit or Gradio
- Switch to a paid LLM endpoint (OpenAI, Anthropic, Together AI) for production-quality answers