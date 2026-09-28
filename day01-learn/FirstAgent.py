import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="kimi-k2.6",
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url=os.getenv("MOONSHOT_BASE_URL")
)

# 定义工具方法
def weather_tool(city: str) -> str:
    """ 获取指定城市的天气"""
    return f"{city}天气是多云的"

# 创建智能体
agent = create_agent(model=llm, tools=[weather_tool, ],
             system_prompt="你是一个智能助手，能够回答用户关于天气的问题，可以使用提供的天气工具来获取天气信息")

# 构造消息体
messages = {
    "messages": [
        {"role": "user", "content": "你好，我叫哈喽，长春天气怎么样"}
    ]
}

# 调用智能体
result = agent.invoke(messages)

print(result)