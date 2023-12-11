from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMMathChain 
from langchain.tools import DuckDuckGoSearchRun
import os

from langchain.memory import ConversationBufferWindowMemory


from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=openai_api_key)




search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, years, dates, issues, etc. Worth using for general topics. Use precise questions.",
)

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
math_tool = Tool.from_function(
    func=llm_math_chain.run,
    name="Calculator",
    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.",
)

tools=[search_tool, math_tool]



prefix = """
I am a highly capable assistant with access to a variety of tools to help with different tasks. 
Whether it's writing code, drafting emails, searching the web, or solving math problems, I can handle it all. 
Below is the list of tools at my disposal and the current conversation history to provide context for my responses.
"""

suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,

    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferWindowMemory(memory_key="chat_history")


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

agent_chain.run(input="How many people live in canada?")


print(" ----- "* 16)

agent_chain.run(input="what is their national anthem called?")
