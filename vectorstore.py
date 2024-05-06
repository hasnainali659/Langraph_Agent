import os
import pickle
import torch

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader


class VectorEmbeddingsBase:
    """
    Base class for vector embeddings.

    Args:
        file (str): The path to the file.
        upload_dir (str, optional): The directory where the file will be uploaded. Defaults to 'media'.
        document_type (str, optional): The type of the document. Defaults to ''.
    """

    def __init__(self, file, upload_dir="media", document_type=""):
        self.upload_dir = upload_dir
        self.document_type = document_type
        self.file = file

    @staticmethod
    def bge_embedding():
        """
        Returns a HuggingFaceBgeEmbeddings object for BGE embedding.

        This method initializes a HuggingFaceBgeEmbeddings object with the specified model name and model kwargs.
        It also sets the encode kwargs to normalize the embeddings.

        Returns:
            HuggingFaceBgeEmbeddings: A HuggingFaceBgeEmbeddings object for BGE embedding.
        """
        
        try:
            model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
            print("CUDA is available!" if model_kwargs["device"] == "cuda" else "CUDA is not available!")
            print("Loading BGE model...")
            model_name = "BAAI/bge-base-en-v1.5"
            encode_kwargs = {"normalize_embeddings": True}

            return HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            
        except Exception as e:
            return str(e)

    def load_document(self):
        """
        Load and parse the document based on its file extension.

        Returns:
            The loaded document as a string.

        Raises:
            None.
        """
        
        try:
            name, extension = os.path.splitext(self.file)
            loader_mapping = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".txt": TextLoader,
                ".csv": CSVLoader,
            }
            if extension not in loader_mapping:
                print("Document format is not supported!")
                return None

            loader = loader_mapping[extension](self.file)
            print(f"Loading {self.file}")
            return loader.load()
        
        except Exception as e:
            return str(e)
        

    def chunk_data(self, data):
        """
        Chunk the given data into smaller segments based on the file type.

        Args:
            data (str): The data to be chunked.

        Returns:
            list: A list of smaller segments of the data.

        """
        try:
            name, extension = os.path.splitext(self.file)
            if extension == ".pdf" or extension == ".docx" or extension == ".txt":
                chunk_size = 500
                chunk_overlap = 200
            else:
                chunk_size = 200
                chunk_overlap = 100
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            return text_splitter.split_documents(data)
        
        except Exception as e:
            return str(e)

    def base_retriever(self, chunks, username, filename):
        """
        Saves a BM25Retriever object to a pickle file.

        Args:
            chunks (list): List of document chunks.
            username (str): User's username.
            filename (str): Name of the file.

        Returns:
            None
        """
        
        try:
            filename = os.path.splitext(os.path.basename(filename))[0]
            bm25_retriever = BM25Retriever.from_documents(chunks)
            retriever_path = os.path.join(
                self.upload_dir,
                f"{self.document_type}-search",
                username,
                "base_retriever_db",
                f"{filename}_retriever.pkl",
            )

            with open(retriever_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
        
        except Exception as e:
            return str(e)

    def standard_retriever(self, chunks, username):
        """
        Retrieves vectors for the given chunks and persists them in a vector database.

        Args:
            chunks (list): A list of chunks to retrieve vectors for.
            username (str): The username associated with the vector database.

        Returns:
            None
        """
        
        try:
            persist_directory = os.path.join(
                self.upload_dir, f"{self.document_type}-search", username, "standard_db"
            )
            vectordb = Chroma.from_documents(
                chunks,
                VectorEmbeddingsBase.bge_embedding(),
                persist_directory=persist_directory,
            )
            vectordb.persist()
            
        except Exception as e:
            return str(e)

    def hyde_retriever(self, chunks, username):
        """
        Retrieves and persists vector embeddings for the given chunks of documents.

        Args:
            chunks (list): A list of document chunks.
            username (str): The username associated with the document.

        Returns:
            None
        """
        
        try:
            base_embeddings = self.bge_embedding()
            llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
            embeddings = HypotheticalDocumentEmbedder.from_llm(
                llm, base_embeddings, "web_search"
            )
            persist_directory = os.path.join(
                self.upload_dir, f"{self.document_type}-search", username, "hyde_db"
            )
            vectordb = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=persist_directory
            )
            vectordb.persist()
            
        except Exception as e:
            return str(e)

    def add_vectorstore(self, username, filename):
        """
        Add a vector store for the given username and filename.

        Parameters:
        - username (str): The username associated with the vector store.
        - filename (str): The name of the file to be added to the vector store.

        Returns:
        None
        """

        try:
            filedir_path = os.path.join(
                self.upload_dir, f"{self.document_type}-search", username
            )
            standardb_path = os.path.join(filedir_path, "standard_db")
            base_retriever_path = os.path.join(filedir_path, "base_retriever_db")
            hyde_retriever_path = os.path.join(filedir_path, "hyde_db")

            for path in [standardb_path, base_retriever_path, hyde_retriever_path]:
                if not os.path.exists(path):
                    os.makedirs(path)

            data = self.load_document()            
            chunks = self.chunk_data(data)            
            self.base_retriever(chunks, username, filename)            
            self.standard_retriever(chunks, username)            
            self.hyde_retriever(chunks, username)

        except Exception as e:
            return str(e)

    def remove_vectorstore(self, username, file_path):
        """
        Remove vectors from the vector store based on the given file path.

        Args:
            username (str): The username of the user.
            file_path (str): The file path to remove vectors for.

        Returns:
            None
        """
        
        try:
            filedir_path = os.path.join(
                self.upload_dir, f"{self.document_type}-search", username
            )
            retriever_paths = [
                os.path.join(filedir_path, "standard_db"),
                os.path.join(filedir_path, "hyde_db"),
            ]

            for path in retriever_paths:
                vector_store = Chroma(
                    persist_directory=path, embedding_function=self.bge_embedding()
                )
                ids = vector_store.get(where={"source": file_path})["ids"]

                if len(ids) > 0:
                    vector_store.delete(ids=ids)

            base_retriever_path = os.path.join(filedir_path, "base_retriever_db")
            if os.path.exists(base_retriever_path):
                file_without_ext = os.path.splitext(file_path)[0].split("/")[-1]
                os.remove(
                    os.path.join(base_retriever_path, f"{file_without_ext}_retriever.pkl")
                )

        except Exception as e:
            return str(e)
        
if __name__ == "__main__":

    vectorstore = VectorEmbeddingsBase(file="media/semantic-search/hasnain/handbook.pdf", document_type="semantic")
    vectorstore.add_vectorstore(username="hasnain", filename="handbook.pdf")