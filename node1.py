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
        step_list = [
        Perform RAG on the handbook to find the company policy for daily attendance.
        Perform database search on the csv database to find the employee attendance.
        ]

        step_queries = [
        "find the company policy for daily attendance in the handbook.",
        "find the employee attendance in the csv database."
        ]

        actions = ["rag", "database"]
        
        Example 2:

        query: You are provided with multiple resume pdf. Use web to find the ideal resume for the job. Then compare the 
        ideal resume with the provided resume and provide the best one. Here is the workflow dictionary created by the user.

        Response by the chain:

        steps to process the query:
        step_list = [
        Perform web search to find the ideal resume for the job.
        Compare the ideal resume with the provided resume and provide the best one.
        ]

        step_queries = [
        "find the ideal resume for the job.",
        "compare the ideal resume with the provided resume and provide the best one."
        ]

        step_actions = ["websearch", "rag"]

        Return an output json with the "step_list", "step_queries" and "step_actions" as key and the step list,
        step queries list and action list as value.
        """

    prompt = PromptTemplate.from_template(template)

    strategy_chain = prompt | llm | JsonOutputParser()

    return strategy_chain

# query = """ from the book reasons_for_obesity.pdf find after which bmi the patient has to be considered obese.
# and from bmi.csv find the patient with the highest bmi and are considered obese. 
# """
# query_dict = {1: 'reasons_for_obesity.pdf', 2: 'bmi.csv'}
# formatted_query_dict = str(query_dict)
# new_query = query + f'Here is the workflow dictionary created by the user {formatted_query_dict}'

# chain = query_strategy_chain(new_query, query_dict)
# result = chain.invoke({"query":new_query})

# print(result)


def rag_chain(rag_query, document_name):

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

    template = """You are an expert in question answer related task. Given a query
    and context you can answer the question.
    
    query: {query}
    
    context: {context}
    """

    qa_prompt = PromptTemplate.from_template(template)

    file_path = f"media/semantic-search/hasnain/{document_name}"

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

#####################################################################

from langchain.schema import Document
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_email: email
        email_category: email category
        draft_email: LLM generation
        final_email: LLM generation
        research_info: list of documents
        info_needed: whether to add search info
        num_steps: number of steps
    """
    step_to_take : int
    initial_query : str
    query_dict : dict
    response_dict : dict 
    step_list : list
    step_queries : list
    step_actions : list
    rag_response : str
    final_response : str
    num_steps : int

######################################################################

def build_strategy(state):
    """take the initial query and return a step wise strategy to process it"""
    initial_query = state['initial_query']
    query_dict = state['query_dict']
    num_steps = int(state['num_steps'])
    num_steps += 1

    chain = query_strategy_chain(initial_query, query_dict)
    response_dict = chain.invoke({"query":initial_query})
    print("\n", response_dict, "\n")

    # state['step_to_take'] = len(response_dict['step_list'])

    return {"step_list": response_dict['step_list'], "step_queries": response_dict["step_queries"],
            "step_actions": response_dict["step_actions"], "num_steps":num_steps, "step_to_take": len(response_dict['step_list'])}

def rag_or_database(state):
    """take the step_list, step_query and action and return a rag or database based answer"""
    if len(state['step_list']) > 1 and type(state['step_list']) == list:
        step_list = state['step_list'][0]
        step_query = state['step_queries'][0]
        action = state['step_actions'][0]

    else:
        state['step_list'] = [state['step_list']]
        state['step_queries'] = [state['step_queries']]
        state['step_actions'] = [state['step_actions']]

        step_list = state['step_list'][0]
        step_query = state['step_queries'][0]
        action = state['step_actions'][0]

    document_name = state['query_dict'][1]

    num_steps = int(state['num_steps'])
    num_steps += 1

    if action == "rag":
        return "rag"
    elif action == "database":
        return "database_search"
    else:
        return "state_printer"

def rag_node(state):
    """take the step_list, step_query and action and return a rag based answer"""

    if len(state['step_list']) > 1 and type(state['step_list']) == list:
        step_list = state['step_list'][0]
        step_query = state['step_queries'][0]
        action = state['step_actions'][0]
    
    else:
        state['step_list'] = [state['step_list']]
        state['step_queries'] = [state['step_queries']]
        state['step_actions'] = [state['step_actions']]

        step_list = state['step_list'][0]
        step_query = state['step_queries'][0]
        action = state['step_actions'][0]

    document_name = "handbook.pdf"

    num_steps = int(state['num_steps'])
    num_steps += 1

    response = rag_chain(step_query, document_name)
    print("\n", response, "\n")

    if state['step_to_take'] > 1:
        state['step_list'] = state['step_list'].pop()
        state['step_queries'] = state['step_queries'].pop()
        state['step_actions'] = state['step_actions'].pop()
   
    else:
        state['step_list'] = []
        state['step_queries'] = []
        state['step_actions'] = []

    # state['step_to_take'] = state['step_to_take'] - 1

    return {"step_list": state['step_list'], "step_queries": state["step_queries"],
            "step_actions": state["step_actions"], "rag_response": response,"num_steps":num_steps,
            "step_to_take": state['step_to_take'] - 1}

def database_search(state):
    """ Perform database search"""
    
    if len(state['step_list']) > 1 and type(state['step_list']) == list:
        step_list = state['step_list'][0]
        step_query = state['step_queries'][0]
        action = state['step_actions'][0]
    
    else:
        state['step_list'] = [state['step_list']]
        state['step_queries'] = [state['step_queries']]
        state['step_actions'] = [state['step_actions']]

        step_list = state['step_list'][0]
        step_query = state['step_queries'][0]
        action = state['step_actions'][0]

    document_name = "Bitcoin_Historical_Data.csv"

    num_steps = int(state['num_steps'])
    num_steps += 1

    response = database_chain(step_query, document_name)

    if state['step_to_take'] > 1:
        state['step_list'] = state['step_list'].pop()
        state['step_queries'] = state['step_queries'].pop()
        state['step_actions'] = state['step_actions'].pop()
    
    else:
        state['step_list'] = []
        state['step_queries'] = []
        state['step_actions'] = []

    # state['step_to_take'] = state['step_to_take'] - 1

    return {"step_list": state['step_list'], "step_queries": state["step_queries"],
            "step_actions": state["step_actions"], "final_response": response, "num_steps":num_steps,
            "step_to_take": state['step_to_take'] - 1}

def state_printer(state):
    """print the state"""
    print("---STATE PRINTER---")
    print(f"Initial query: {state['initial_query']} \n" )


workflow = StateGraph(GraphState)

workflow.add_node("build_strategy", build_strategy)
workflow.add_node("rag_node", rag_node)
workflow.add_node("database_search", database_search)
workflow.add_node("state_printer", state_printer)


workflow.set_entry_point("build_strategy")
workflow.add_conditional_edges(
    "build_strategy",
    rag_or_database,
    {
        "rag": "rag_node",
        "database_search": "database_search",
        "state_printer": "state_printer"
    },
)


# workflow.add_edge("database_search", "state_printer")
# workflow.add_edge("rag_node", "")

workflow.add_conditional_edges(
    "rag_node",
    rag_or_database,

    {
        "rag": "rag_node",
        "database_search": "database_search",
        "state_printer": "state_printer"
    },
)

workflow.add_conditional_edges(
    "database_search",
    rag_or_database,
    {
        "rag": "rag_node",
        "database_search": "database_search",
        "state_printer": "state_printer"
    },
)

# workflow.add_edge("rag_node", "state_printer")
# workflow.add_edge("database_search", "state_printer")
workflow.add_edge("state_printer", END)

app = workflow.compile()


#############################################################################

query1 = """ Who is the chairman of icsarabia? from handbook.pdf. And what is the average of high and low level of bitcoin 
from Bitcoin_Historical_Data.csv?. 
"""

query2 = """ What is the average of high and low level of bitcoin from Bitcoin_Historical_Data.csv?. And what is the 
leave policy for employees in the company from handbook.pdf?
"""
query_dict = {1: 'handbook.pdf', 2: 'Bitcoin_Historical_Data.csv'}
formatted_query_dict = str(query_dict)

inputs = {"initial_query": query2, "query_dict": query_dict, "num_steps":0}

output = app.invoke(inputs)

print(output)