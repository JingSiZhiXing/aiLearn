from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化模型   qwen模型对事件流支持不行,不能生成tool_id,故此换成deepseek
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# 创建一个工具
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}的天气是晴朗的!"

# 创建代理
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手，能够回答用户关于天气的问题。请使用提供的工具来获取天气信息。",
    checkpointer=InMemorySaver(),  # 使用InMemorySaver来保存对话状态，这样在同一线程中进行的对话可以共享状态
)
# 创建线程id
config = {"configurable": {"thread_id": str(uuid7())}}

# 使用事件流获取代理全部执行过程
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "长沙天气怎么样？"}]},
    config=config,
    version="v3",
)

# 获取原始的流事件
# print("获取原始的流事件")
# for event in stream:
#     print(event)

# 获取llm返回的每条消息
for message in stream.messages:
    # print("获取消息的文本增量和最终文本:")
    # for delta in message.text:
    #     print(delta, end="", flush=True)
    # print("获取模型生成工具调用时的内容")
    # print(message.tool_calls.get())
    # print("获取模型本次输出消息:")
    # print(message.output)
    print("获取模型本次推理消息:")
    for delta in message.reasoning:
        print(f"[thinking] {delta}", end="", flush=True)

#
# print()
# print("------获取最终状态-------")
# print(stream.output)