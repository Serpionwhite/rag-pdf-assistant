"""PDF loading: reads a file from disk and returns LangChain Documents."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF file and return one Document per page.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects. Each Document has:
        - ``page_content``: the extracted text for that page
        - ``metadata["source"]``: the file path as a string
        - ``metadata["page"]``: 0-indexed page number

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    # TODO: implement this function.
    #
    # Steps:
    #   1. Convert `path` to a Path object and check it exists.
    #      Raise FileNotFoundError with a descriptive message if not.
    #   2. Create a PyPDFLoader — it takes a string path, not a Path object.
    #      Hint: PyPDFLoader(str(path))
    #   3. Call .load() on the loader — it returns list[Document].
    #   4. Return the documents.
    #
    # Try it in a notebook or REPL first:
    #   from langchain_community.document_loaders import PyPDFLoader
    #   docs = PyPDFLoader("data/sample.pdf").load()
    #   print(len(docs), docs[0].metadata)
    

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Path not found : {path}")
    
    loader = PyPDFLoader(str(path))

    docs = loader.load()

    return docs


