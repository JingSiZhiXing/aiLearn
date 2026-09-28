from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

model = "qwen3.7-max-2026-05-20"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

llm = ChatQwen(
    model=model,
    api_key=api_key,
    base_url=api_base_url,
)


# 创建自定义的上下文
@dataclass
class Context:
    user_name: str


# 创建动态提示词
@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # 通过request获取runtime中上下文存在的用户名
    user_name = request.runtime.context.user_name
    # 从ES或者redis中获取到指定的少样本示例，动态的添加到系统提示词中
    # skills：有一个后端（系统环境），需要从这个后端中读取SKILL.md中的前置元数据（skill描述、skill工具、skills读取路径），
    # 将元数据封装到提示词中；通过工具，从我们的后端中加载全部的SKILL.md内容
    # deepagents中会详细的去使用skills
    return request.system_message.content + f"当前用户的名称是{user_name}"


# 创建代理
agent = create_agent(
    model=llm,
    middleware=[state_aware_prompt],
    system_prompt="你是一个翻译的专家，能够轻松的中译英",
    context_schema=Context
)
# 运行代理
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好呀,我的名字叫什么，在帮我翻译：今天天气真不错"}]},
    context=Context(user_name="初见")
)
print(response["messages"][-1].content)
