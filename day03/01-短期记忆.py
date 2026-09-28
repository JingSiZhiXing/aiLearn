from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  # 短期记忆存储的模块
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = ChatOpenAI(model="deepseek-v4-flash",
                 api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"))
# 创建一个检查点来进行存储短期记忆
checkpointer = InMemorySaver()

# 定义一个线程配置，指定线程id=user_1，通过线程id就能区别不同的会话，同一个线程id之间的内容（记忆）共享
thread_config = {"configurable": {"thread_id": "user_1"}}

agent = create_agent(
    model=llm,
    checkpointer=checkpointer  # 添加短期记忆
)

res1 = agent.invoke({
    "messages": [
        HumanMessage(content="你好，我的名字叫初见")
    ]
}, config=thread_config)

res2 = agent.invoke({
    "messages": [
        HumanMessage(content="你好，我的名字叫什么")
    ]
}, config=thread_config)
print("第一次回复：", res1)
print("第二次回复：", res2)
