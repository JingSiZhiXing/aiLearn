from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="kimi-k2.6",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)


# 创建一个天气工具
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""  # 工具的描述（工具的使用说明）

    return f"{city} 天气是晴朗的"


# 创建Agent代理
agent = create_agent(
    model=llm,
    tools=[get_weather, ],  # 工具列表
    # 系统提示词
    system_prompt="你是一个智能助手，能够回答用户关于天气的问题，可以使用提供的天气工具来获取天气信息"
)
# 传递的格式要是messages类型
# messages = {
#     "messages": [
#         {"role": "user","content": "你好，我叫初见"},
#     ]
# }
messages = {
    "messages": [
        SystemMessage(content="你是一个助手"),  # 不推荐使用SystemMessage，而是使用system_prompt
        HumanMessage(content="你好，我是初见"),

    ]

}

result = agent.invoke(messages)
print(result)
