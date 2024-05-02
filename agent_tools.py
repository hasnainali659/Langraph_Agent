import zipfile
import json
from dotenv import load_dotenv

from utilities import extract_summary

from langchain.agents import tool
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI

@tool
def decorated_colored_print(message: str):
    """Prints a message in a decorated and colored format."""
    print(f"{'*' * 10} {message} {'*' * 10}")


@tool
def add(a, b):
    """Adds two numbers together and returns the result."""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtracts the second number from the first number and returns the result."""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers together and returns the result."""
    return a * b

@tool
def divide(a: int, b: int):
    """Divides the first number by the second number and returns the result."""
    return a / b

import datetime
from langchain.agents import tool

@tool
def check_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""

    # get the current date and time
    current_time = datetime.datetime.now()
    
    # format the time as a string in the format "YYYY-MM-DD HH:MM:SS"
    formatted_time = current_time.strftime(format)
    
    # return the formatted time
    return formatted_time

@tool
def extract_zip_file(zip_file_path):
    """Extracts the zip file to the current directory and return extracted file path of the text file"""

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall()
    return zip_ref.namelist()[0]

@tool
def read_and_summarize_text_file(text_file_path):
    """Reads the text file case.txt in the extracted folder from the zip file and summarize the file and output
    the result in correct json format as shown below:"""

    loader = TextLoader(text_file_path)
    documents = loader.load()

    prompt_template = """ Summarize the text below:
    {text}
    
    Provide a detailed summary of the text above.
    
    Detailed Summary:
    
    Also, provide important part summary in the format below:
    
    Example:
    
    Page Line: 10-15
    Topic: Introduction
    Summary: This section introduces the topic of the document.
    """
    
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
    
    summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    summarize_chain = load_summarize_chain(
        llm=llm, chain_type="stuff", prompt=summary_prompt
    )

    result = summarize_chain.invoke(documents)

    output_json = extract_summary(result["output_text"])
