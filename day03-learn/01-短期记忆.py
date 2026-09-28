from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
llm = ChatOpenAI(model="kimi-k2.6",
                 api_key=os.getenv("MOONSHOT_API_KEY"),
                 base_url=os.getenv("MOONSHOT_BASE_URL"))

checkPointer = InMemorySaver()

# 定义一个线程配置，指定线程id=user_1，通过线程id就能区别不同的会话，同一个线程id之间的内容（记忆）共享
threadConfig = {"configurable": {"thread_id": "user_1"}}

agent = create_agent(model=llm, checkpointer=checkPointer) # 添加短期记忆

result1 = agent.invoke({"messages": [HumanMessage(content="你好，我的名字叫哈喽")]}, config=threadConfig)

result2 = agent.invoke({"messages": [HumanMessage(content="我的名字是什么")]}, config=threadConfig)

print("第一次回复",result1)

print("第二次回复",result2)