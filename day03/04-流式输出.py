from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(model="deepseek-v4-flash", api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))


# 创建一个工具
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 创建自定义流式输出
    stream_writer = get_stream_writer()
    stream_writer(f"正在获取{city}的天气信息...")
    stream_writer(f"{city}的天气信息获取完成！{city}的天气是晴朗的!")
    return f"{city}的天气是晴朗的!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手，能够回答用户关于天气的问题。请使用提供的工具来获取天气信息。",
    checkpointer=InMemorySaver(),  # 使用InMemorySaver来保存对话状态，这样在同一线程中进行的对话可以共享状态
)
config = {"configurable": {"thread_id": str(uuid7())}}

# print("====================== 流式输出：updates ======================")
# for chunk in agent.stream(
#         {"messages": [{"role": "user", "content": "长沙天气怎么样?"}]},
#         config=config,
#         stream_mode="updates",  # 指定流式输出模式
#         version="v2",  # 版本使用v2版本
# ):
#     if chunk["type"] == "updates":  # 过滤updates模式
#         for step, data in chunk["data"].items():
#             print(f"step: {step}")
#             print(f"content: {data['messages'][-1].content_blocks}")

# print("====================== 流式输出：messages ======================")
#
# # 监控llm进行流式输出
# for chunk in agent.stream(
#         {"messages": [{"role": "user", "content": "长沙天气怎么样?"}]},
#         config=config,
#         stream_mode="messages",
#         version="v2",
# ):
#     if chunk["type"] == "messages":
#         # 在"messages"模式下，流式输出的每个chunk包含一个完整的消息对象和元数据，我们可以直接访问消息内容
#         token, metadata = chunk["data"]
#         print(f"node: {metadata['langgraph_node']}")
#         print(f"content: {token.content}")  # 会输出很多空的内容，代表模型在进行思考，直到最后输出完整的消息内容
#         print("\n")

print("====================== 流式输出：custom ======================")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "长沙天气怎么样?"}]},
    config=config,
    stream_mode=["updates", "messages", "custom"],
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])

    if chunk["type"] == "updates":
        pass

    if chunk["type"] == "messages":
        pass