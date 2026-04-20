# Load environment variables from .env (so HUGGINGFACEHUB_API_TOKEN
# becomes available to any Hugging Face library that looks for it).
from dotenv import load_dotenv
load_dotenv()

# Import the loader class. PyPDFLoader knows how to read a PDF
# and convert it into LangChain's standard "Document" format.
from langchain_community.document_loaders import PyPDFLoader

# Create a loader instance pointing at our PDF file.
# This does NOT read the file yet — it just sets things up.
loader = PyPDFLoader("document1.pdf")

# Now actually read the file. .load() returns a list of Document
# objects, one per page of the PDF.
documents = loader.load()

# Print how many Documents we got. For most PDFs, this equals page count.
print(f"Loaded {len(documents)} document(s) from the PDF.")

# Each Document has a .page_content (the text) and .metadata (a dict
# with things like the page number and source filename).
# We print the first 500 chars of page 1 to confirm the text looks right.
print("---- First 500 characters of page 1 ----")
#print(documents[0].page_content[:500])

# Also print the metadata for page 1, just to see what's in there.
print("---- Metadata for page 1 ----")
#print(documents[0].metadata)

# Import the splitter. RecursiveCharacterTextSplitter is the
# go-to choice for most prose documents.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Configure the splitter.
# - chunk_size=1000 means "try to keep each chunk under 1000 chars"
# - chunk_overlap=200 means "repeat the last 200 chars of each chunk
#   at the start of the next one" (preserves context across boundaries)




def is_likely_navigation_page(text):
    """
    Detect pages that look like an index, table of contents, or other
    navigational content (rather than real prose).

    Heuristic: navigation pages have an unusual density of digits
    (page numbers) and commas (separator between entries).
    """
    # Very short pages don't give us enough signal to judge — keep them.
    if len(text) < 200:
        return False

    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    comma_ratio = text.count(',') / len(text)

    # These thresholds were eyeballed; tune for your document.
    # Normal prose is roughly <1% digits and <2% commas.
    # Indexes are typically 5%+ digits and 2%+ commas.
    return digit_ratio > 0.05 and comma_ratio > 0.015


# Filter out navigation pages before we chunk anything.
content_pages = [doc for doc in documents if not is_likely_navigation_page(doc.page_content)]
removed_count = len(documents) - len(content_pages)
print(f"\nFiltered out {removed_count} navigation pages (index/TOC).")
print(f"Keeping {len(content_pages)} content pages for chunking.")









text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# split_documents takes our list of page-level Documents and returns
# a new list of smaller chunk-level Documents. Metadata (page numbers etc)
# is preserved on each chunk, which is great — you can later trace any
# answer back to the page it came from.
#chunks = text_splitter.split_documents(documents)
chunks = text_splitter.split_documents(content_pages)
# How many chunks did we end up with? Usually several times the page count.
#print(f"\nSplit the document into {len(chunks)} chunks.")

# Look at the first chunk to confirm it looks reasonable.
#print("---- First chunk content ----")
#print(chunks[0].page_content)
#print("---- First chunk metadata ----")
#print(chunks[0].metadata)

# And the second chunk, so you can SEE the overlap in action —
# notice how the start of chunk 2 repeats the end of chunk 1.
#print("---- Second chunk content ----")
#print(chunks[1].page_content)


# Look at a chunk from the middle of the book to see real prose chunking.
#print("\n---- Chunk 45 ----")
#print(chunks[45].page_content)
#print("\n---- Chunk 46 ----")
#print(chunks[46].page_content)
#print("\n---- Chunk 47 ----")
#print(chunks[47].page_content)
#print("\n---- Chunk 48 ----")
#print(chunks[48].page_content)

# Import the Hugging Face embedding wrapper and the FAISS vector store.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the embedding model. The first time this runs, it downloads
# the model from Hugging Face (~80MB). Subsequent runs use the cache.
# all-MiniLM-L6-v2 is small, fast, and good enough for most use cases.
print("\nLoading the embedding model... (downloads ~80MB the first time)")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build the FAISS index from our chunks.
# This embeds every chunk (slow!) and stores the resulting vectors.
print(f"Embedding {len(chunks)} chunks and building the FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store built successfully.")

# Quick test: do a similarity search to see what comes back.
# We're NOT involving an LLM yet — just retrieving the most relevant chunks.
#query = "How should I pose someone's hands?"
#hyde style
#query = "Fingers should appear relaxed and natural in a portrait"
query = "Avoid stiff or awkward hand positions when posing a subject"
print(f"\n---- Top 3 chunks for query: '{query}' ----")
results = vector_store.similarity_search(query, k=10)
for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} (page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:300])





# Keep this import — the prompt template is used regardless of LLM mode.
from langchain_core.prompts import PromptTemplate

# ============================================================
# LLM CONFIGURATION — change LLM_MODE below to switch backends.
# ============================================================
# "api"   = call a hosted model on Hugging Face's Inference API.
#           Fast, but may fail if the model isn't on the free tier.
# "local" = download a smaller model and run it on your CPU.
#           Slower, but reliable and free.
LLM_MODE = "local"

# If LLM_MODE == "api", which hosted model? Try these in order:
#   "HuggingFaceH4/zephyr-7b-beta"
#   "mistralai/Mistral-7B-Instruct-v0.2"
#   "mistralai/Mistral-7B-Instruct-v0.3"
API_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# If LLM_MODE == "local", which model?

# Causal LMs (decoder-only) — pick by size/quality tradeoff:
#   "Qwen/Qwen2.5-0.5B-Instruct"        (~1GB, tiny and fast)
#   "Qwen/Qwen2.5-1.5B-Instruct"        (~3GB, slower but better)
#   "microsoft/Phi-3-mini-4k-instruct"  (~2.4GB, strong for Q&A)
LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# ============================================================

if LLM_MODE == "api":
    from langchain_huggingface import HuggingFaceEndpoint
    print(f"\nConnecting to {API_MODEL} via Hugging Face Inference API...")
    llm = HuggingFaceEndpoint(
        repo_id=API_MODEL,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )

elif LLM_MODE == "local":
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline
    print(f"\nLoading {LOCAL_MODEL} locally... (downloads on first run)")
    pipe = pipeline(
        "text-generation",          # causal LMs use this task
        model=LOCAL_MODEL,
        max_new_tokens=512,
        return_full_text=False,     # don't echo the prompt back in the output
        do_sample=False,            # deterministic, factual answers
    )
    llm = HuggingFacePipeline(pipeline=pipe)

else:
    raise ValueError(f"Unknown LLM_MODE: {LLM_MODE!r}. Use 'api' or 'local'.")

# The prompt template. {context} and {question} are placeholders that
# get filled in for each query. Notice the explicit "if you don't know"
# instruction — this REDUCES hallucinations dramatically.
prompt = PromptTemplate.from_template("""You are a helpful assistant answering questions about a photography book.
Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't know based on the provided context."
Be concise but thorough.

Context:
{context}

Question: {question}

Answer:""")


def answer_question(question, k=5):
    """Run the full RAG pipeline for a single question."""
    # 1. Retrieve the K most relevant chunks.
    docs = vector_store.similarity_search(question, k=k)

    # 2. Concatenate them into one big context string.
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Fill in the prompt template.
    formatted_prompt = prompt.format(context=context, question=question)

    # 4. Send it to the LLM and get back the answer.
    answer = llm.invoke(formatted_prompt)

    return answer, docs


# Try it!
question = "How should I pose someone's hands to look natural?"
print(f"\nQ: {question}\n")
answer, sources = answer_question(question)
print(f"A: {answer.strip()}\n")
print("Sources used (for attribution):")
for i, doc in enumerate(sources, 1):
    page = doc.metadata.get("page", "?")
    print(f"  {i}. Page {page}")