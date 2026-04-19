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
print(documents[0].page_content[:500])

# Also print the metadata for page 1, just to see what's in there.
print("---- Metadata for page 1 ----")
print(documents[0].metadata)

# Import the splitter. RecursiveCharacterTextSplitter is the
# go-to choice for most prose documents.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Configure the splitter.
# - chunk_size=1000 means "try to keep each chunk under 1000 chars"
# - chunk_overlap=200 means "repeat the last 200 chars of each chunk
#   at the start of the next one" (preserves context across boundaries)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# split_documents takes our list of page-level Documents and returns
# a new list of smaller chunk-level Documents. Metadata (page numbers etc)
# is preserved on each chunk, which is great — you can later trace any
# answer back to the page it came from.
chunks = text_splitter.split_documents(documents)

# How many chunks did we end up with? Usually several times the page count.
print(f"\nSplit the document into {len(chunks)} chunks.")

# Look at the first chunk to confirm it looks reasonable.
print("---- First chunk content ----")
print(chunks[0].page_content)
print("---- First chunk metadata ----")
print(chunks[0].metadata)

# And the second chunk, so you can SEE the overlap in action —
# notice how the start of chunk 2 repeats the end of chunk 1.
print("---- Second chunk content ----")
print(chunks[1].page_content)


# Look at a chunk from the middle of the book to see real prose chunking.
print("\n---- Chunk 45 ----")
print(chunks[45].page_content)
print("\n---- Chunk 46 ----")
print(chunks[46].page_content)
print("\n---- Chunk 47 ----")
print(chunks[47].page_content)
print("\n---- Chunk 48 ----")
print(chunks[48].page_content)