from dotenv import load_dotenv

from langchain import hub
from react_prompt_template import get_react_prompt_template
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
    AgentType
)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.schema import Document
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List
import torch
import pandas as pd


load_dotenv()

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


def query_strategy_chain(query, query_dict):

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

    template="""
        You are a Strategy Agent You are a master at understanding what a user wants when they write a query and able to provide steps
        to how to process the query.

        You will be given a query {query} and you have to provide a step by step guide on how to process the query.

        These are the list of steps that you can take to process the query:

        1. If there is a pdf/doc/docx/txt document then you will have to perform RAG (Retrieval Augmented Generation) on the document.
        2. If there is a csv/xlsx/xlx document then you will have perform database search.
        3. If there is a website search then you will have to perform web search.

        Here is an example of sample query:

        Example 1: 

        query: How many employees are late this month. You are provided with the handbook in which company policy is mentioned for
        daily attendance and a csv database in which employee attendance is mentioned. Here is the workflow dictionary created by the user
        in which the files attached in which order are mentioned.

        Response by the chain:

        steps to process the query:
        steps_list = [
        Perform RAG on the handbook to find the company policy for daily attendance.
        Perform database search on the csv database to find the employee attendance.
        ]

        step_queries = [
        "find the company policy for daily attendance in the handbook.",
        "find the employee attendance in the csv database."
        ]
        
        Example 2:

        query: You are provided with multiple resume pdf. Use web to find the ideal resume for the job. Then compare the 
        ideal resume with the provided resume and provide the best one. Here is the workflow dictionary created by the user.

        Response by the chain:

        steps to process the query:
        steps_list = [
        Perform web search to find the ideal resume for the job.
        Compare the ideal resume with the provided resume and provide the best one.
        ]

        step_queries = [
        "find the ideal resume for the job.",
        "compare the ideal resume with the provided resume and provide the best one."
        ]
        Return an output json with the "steps_list" and "steps_queries" as key and the steps list and
        stwp queries list as value.
        """

    prompt = PromptTemplate.from_template(template)

    strategy_chain = prompt | llm | JsonOutputParser()

    return strategy_chain

query = """ from the book reasons_for_obesity.pdf find after which bmi the patient has to be considered obese.
and from bmi.csv find the patient with the highest bmi and are considered obese. 
"""
query_dict = {1: 'reasons_for_obesity.pdf', 2: 'bmi.csv'}
formatted_query_dict = str(query_dict)
new_query = query + f'Here is the workflow dictionary created by the user {formatted_query_dict}'

chain = query_strategy_chain(new_query, query_dict)
result = chain.invoke({"query":new_query})

print(result)


def rag_chain(rag_query, document_name):

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

    template = """You are an expert in question answer related task. Given a query
    and context you can answer the question.
    
    query: {query}
    
    context: {context}
    """

    qa_prompt = PromptTemplate.from_template(template)

    file_path = "media/semantic-search/hasnain/handbook.pdf"

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)    

    standard_retriever_path = "media/semantic-search/hasnain/standard_db"

    persist_directory = standard_retriever_path
    vector_store = Chroma(persist_directory=persist_directory,embedding_function=bge_embedding())
    search_args = {"filter": {"source": file_path}, "k": 5}
                
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs=search_args
    )

    relevant_docs = retriever.get_relevant_documents(rag_query)

    setup_and_retrieval = RunnableParallel(
    {"context": retriever, "query": RunnablePassthrough()}
    )

    rag_chain = setup_and_retrieval | qa_prompt | llm

    response = rag_chain.invoke(rag_query)

    return response.content


def database_chain(query, document_name):

    df = pd.read_csv(f"media/database-search/hasnain/{document_name}")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    df_agent = create_pandas_dataframe_agent(llm, df, verbose=True,
                                             agent_type=AgentType.OPENAI_FUNCTIONS,
                                             handle_parsing_errors=True)
    
    response = df_agent.invoke(query)

    return response['output']
