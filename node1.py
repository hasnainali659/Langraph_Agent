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
)

from langchain.schema import Document
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List

load_dotenv()

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
    1. Perform RAG on the handbook to find the company policy for daily attendance.
    2. Perform database search on the csv database to find the employee attendance.

    Example 2:

    query: You are provided with multiple resume pdf. Use web to find the ideal resume for the job. Then compare the 
    ideal resume with the provided resume and provide the best one. Here is the workflow dictionary created by the user.

    Response by the chain:

    steps to process the query:
    1. Perform web search to find the ideal resume for the job.
    2. Compare the ideal resume with the provided resume and provide the best one.

    Return an output json with the "steps" as key and the steps to process the query in a list as value.
    """

prompt = PromptTemplate.from_template(template)

strategy_chain = prompt | llm | JsonOutputParser()

query = """ from the book reasons_for_obesity.pdf find after which bmi the patient has to be considered obese.
and from bmi.csv find the patient with the highest bmi and are considered obese. 
"""
query_dict = {1: 'reasons_for_obesity.pdf', 2: 'bmi.csv'}
formatted_query_dict = str(query_dict)
new_query = query + f'Here is the workflow dictionary created by the user {formatted_query_dict}'

result = strategy_chain.invoke({"query": new_query})

print(result)
