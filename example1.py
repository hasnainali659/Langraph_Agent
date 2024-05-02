from dotenv import load_dotenv

from langchain import hub
from react_prompt_template import get_react_prompt_template
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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

@tool
def decorated_colored_print(message: str):
    """Prints a message in a decorated and colored format."""
    print(f"{'*' * 10} {message} {'*' * 10}")

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

tools = [decorated_colored_print]
tool_names = [tool.name for tool in tools]
prompt_template = get_react_prompt_template()

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
# agent_executor.invoke({"input": "what is langchain? Provide answer in a decorated colored print format."})


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_query: query
    """
    initial_query : str
    response : str
    

def test_query(state):

    initial_query = state['initial_query']

    response = agent_executor.invoke({"input": initial_query})
    print(response)

    return {"response": response}

workflow = StateGraph(GraphState)
workflow.add_node("test_query", test_query)

workflow.set_entry_point("test_query")
workflow.add_edge("test_query", END)

app = workflow.compile()

query = "what is langchain? Provide answer in a decorated colored print format."

output = app.invoke({"initial_query": query})
