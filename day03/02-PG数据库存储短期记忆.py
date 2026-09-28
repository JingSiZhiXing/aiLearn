from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  # 短期记忆存储的模块
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 短期记忆指会话记忆

# 初始化模型
llm = ChatOpenAI(model="deepseek-v4-flash",
                 api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"))
# 创建一个PG数据库检查点来进行存储短期记忆
# 创建数据库连接的url，格式为：postgresql://用户名:密码@主机地址:端口号/数据库名称?sslmode=disable关闭 SSL 加密连接
DB_URL = "postgresql://postgres:123456@localhost:5432/langchain_agent?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    checkpointer.setup() # 初始化表
    # 定义一个线程配置，指定线程id=user_1，通过线程id就能区别不同的会话，同一个线程id之间的内容（记忆）共享
    thread_config = {"configurable": {"thread_id": "user_3"}}

    agent = create_agent(
        model=llm,
        checkpointer=checkpointer  # 添加短期记忆
    )

    # res1 = agent.invoke({
    #     "messages": [
    #         HumanMessage(content="你好，我的名字叫初见")
    #     ]
    # }, config=thread_config)

    # 通过PG数据库持久化短期记忆后，之后的每次问答都会把之前存储的state全部拿过来
    res2 = agent.invoke({
        "messages": [
            HumanMessage(content="你好，我的名字叫什么")
        ]
    }, config=thread_config)
    # print("第一次回复：", res1)
    print("第二次回复：", res2)
