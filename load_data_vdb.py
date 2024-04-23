from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings


DATA_PATH = "data/"
DB_PATH = 'vectorstores/db/'

def create_vector_db():
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    vectorestore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)
    
    #  write the in-memory vector embeddings to disk 
    vectorestore.persist()
    print(f'Processed {len(documents)} is stored.')
    
    
    
if __name__ == "__main__":
    create_vector_db()