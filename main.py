from agent_tools import add, subtract, multiply, divide, check_system_time, decorated_colored_print

from dotenv import load_dotenv

from langchain import hub
from react_prompt_template import get_react_prompt_template
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)

load_dotenv()


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

tools = [decorated_colored_print]
tool_names = [tool.name for tool in tools]
prompt_template = get_react_prompt_template()
# prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is langchain? Provide answer in a decorated colored print format."})

