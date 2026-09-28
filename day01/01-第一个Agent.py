import os
import json

from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="kimi-k2.6",
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url=os.getenv("MOONSHOT_BASE_URL")
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
messages = {
    "messages": [
        {"role": "user","content": "你好，我叫初见，长春天气怎么样"},
    ]
}
result = agent.invoke(messages)

# 消息对象不能直接 JSON 序列化，先用 model_dump() 转成普通字典
json_str = json.dumps(
    [message.model_dump() for message in result["messages"]],
    ensure_ascii=False,
    indent=2
)
print(json_str)
# 用户：长沙天气怎么样
# agent：我看到有一个工具是去查询天气的，就觉得要去使用这个工具（提取参数）
    # 工具->create_agent注册进去    系统提示词会加上：可用工具（名称：get_weather，描述：获取指定城市的天气信息）
# 工具：长沙天气是晴朗的
# agent：根据最新的天气信息，长沙目前的天气是**晴朗的**。这样的好天气适合外出活动，不过建议您根据实际温度适当增减衣物，并做好防晒措施。

# 人机交互


# 将Agent应用封装成一个服务-提供接口
