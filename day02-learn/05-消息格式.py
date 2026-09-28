from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="kimi-k2.6",
                 api_key=os.getenv("MOONSHOT_API_KEY"),
                 base_url=os.getenv("MOONSHOT_BASE_URL"))


def get_weather(city: str) -> str:
    """获取指定城市天气信息"""
    return f"{city} 天气是晴朗的"


agent = create_agent(model=llm, tools=[get_weather],
                     system_prompt="你是一个智能助手，能够回答用户关于天气的问题，可以使用提供的天气工具来获取天气信息")

messages = {
    "messages": [
        SystemMessage(content="你是一个智能助手"),
        HumanMessage(content="北京天气如何")
    ]
}
response = agent.invoke(messages)

print(response)