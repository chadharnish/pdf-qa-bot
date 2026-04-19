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