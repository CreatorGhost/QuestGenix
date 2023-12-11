# from fastapi import FastAPI
# from pydantic import BaseModel
# import os
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import AgentType, initialize_agent
# from langchain.chains import LLMMathChain 
# from langchain.tools import DuckDuckGoSearchRun
# from langchain.utilities import WikipediaAPIWrapper
# from langchain.agents import Tool, initialize_agent
# from fastapi.middleware.cors import CORSMiddleware
# from langchain.memory import ConversationBufferMemory
# from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# app.add_middleware(
#        CORSMiddleware,
#        allow_origins=["*"],  # Allows all origins
#        allow_credentials=True,
#        allow_methods=["*"],  # Allows all methods
#        allow_headers=["*"],  # Allows all headers
#    )

# class Message(BaseModel):
#     message: str

# openai_api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=openai_api_key)


# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

# search = DuckDuckGoSearchRun()
# search_tool = Tool(
#     name="Web Search",
#     func=search.run,
#     description="A useful tool for searching the Internet to find information on world events, years, dates, issues, etc. Worth using for general topics. Use precise questions.",
# )

# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
# math_tool = Tool.from_function(
#     func=llm_math_chain.run,
#     name="Calculator",
#     description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.",
# )

# tools=[search_tool, math_tool]
# from langchain.chains import LLMChain

# prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
# suffix = """Begin!"

# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

# prompt = ZeroShotAgent.create_prompt(
#     tools,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input", "chat_history", "agent_scratchpad"],
# )


# llm_chain = LLMChain(llm=llm, prompt=prompt)
# agent = ZeroShotAgent(llm_chain=llm, tools=tools, verbose=True)


# agent_chain = AgentExecutor.from_agent_and_tools(
#     agent=agent, tools=tools, verbose=True, memory=memory
# )
# input_data = {
#         'input': "Tell about the prime numbers",
#         'chat_history': []  # Retrieve chat history from memory
#     }

# response = agent_chain.run(input_data)
# print(response)


# response = agent_chain.run( "write python code for it")

# # print(response)
# # @app.post("/")
# # async def create_completion(message: Message):
# #     return {"botResponse": agent.run(message.message)}