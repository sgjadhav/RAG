
from langchain_community.document_loaders import UnstructuredURLLoader

def load_documents(urls):
    """
    Load documents from URLs
    """

    loader = UnstructuredURLLoader(urls=urls)

    documents = loader.load()

    return documents
