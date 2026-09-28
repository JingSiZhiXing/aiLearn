import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

load_dotenv()

llm = ChatOpenAI(model="kimi-k2.6",
           api_key=os.getenv("MOONSHOT_API_KEY"),
           base_url=os.getenv("MOONSHOT_BASE_URL"))

# 创建一个Mysql数据库检查点来进行存储短期记忆
with PyMySQLSaver.from_conn_string("mysql://root:123456@localhost:3307/langchain_agent") as checkpointer:

    checkpointer.setup()# 初始化表

    threadConfig = {"configurable": {"thread_id": "user_2"}}

    agent = create_agent(model=llm, checkpointer=checkpointer)

    # result1 = agent.invoke({"messages": [HumanMessage(content="你好，我的名字叫哈喽")]}, config=threadConfig)

    # 通过数据库持久化短期记忆后，之后的每次问答都会把之前存储的state全部拿过来
    result2 = agent.invoke({"messages": [HumanMessage(content="我的名字是什么")]}, config=threadConfig)

    # print("第一次回复", result1)

    print("第二次回复", result2)
