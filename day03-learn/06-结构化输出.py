"""
langchain支持四种方式去支持模型原生结构化输出
1.Pydantic模型  -提供数据校验功能、自动转化等功能
2.Dataclass    -是Python标准库自带的，主要用于简化类的定义
3.Typeddict    -用于约束字典（Dict）的键和值的类型,运行时就是一个普通的字典
4.json Schema  -JSON格式规范
"""
import os

from dotenv import load_dotenv
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen

load_dotenv()
llm = ChatQwen(model="kimi-k2.6",
               api_key=os.getenv("MOONSHOT_API_KEY"),
               base_url=os.getenv("MOONSHOT_BASE_URL"))

# 使用qwen模型的结构化输出需要遵守以下内容
"""
在请求体中设置 response_format 参数即可开启结构化输出，需满足以下两个条件：
1.设置response_format参数：将 response_format 参数设置为 {"type": "json_object"}。
2.提示词中包含JSON关键词：System Message 或 User Message 中必须包含"JSON"关键词（不区分大小写），否则API会返回错误：
    'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.
"""

# ============================================================
# 1. Pydantic模型
# ============================================================
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class UserInfoPydantic(BaseModel):
    """用户信息"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄")
    phone: str = Field(description="手机号")


agent_pydantic = create_agent(model=llm,
                              system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
                              response_format=ToolStrategy(UserInfoPydantic))

message = {"messages": [HumanMessage(content="我的名字叫张三，今年20岁，手机号：18578656489")]}

result_pydantic = agent_pydantic.invoke(message)
print("1. Pydantic模型结果:")
print(result_pydantic["structured_response"])
print("-" * 50)

# ============================================================
# 2. Dataclass (Python标准库)
# ============================================================

from dataclasses import field, dataclass


@dataclass
class UserInfoDataclass:
    """用户信息"""
    name: str = field(metadata={"description": "用户姓名"})
    age: int = field(metadata={"description": "用户年龄"})
    phone: str = field(metadata={"description": "手机号"})


# 创建代理
agent_dataclass = create_agent(
    model=llm,
    system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
    response_format=UserInfoDataclass
)

message = [HumanMessage(content="我的名字叫李四，今年25岁，手机号：13800138000")]

result_dataclass = agent_dataclass.invoke({"messages": message})

print("2. Dataclass结果:")
print(result_dataclass["structured_response"])
print("-" * 50)

# ============================================================
# 3. TypedDict (类型约束字典)
# ============================================================

from typing import TypedDict


class UserInfoTypedDict(TypedDict):
    """用户信息"""
    name: str  # 用户姓名
    age: int  # 用户年龄
    phone: str  # 手机号


# 创建代理
agent_typeddict = create_agent(
    model=llm,
    system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
    response_format=UserInfoTypedDict
)

message = [HumanMessage(content="我的名字叫王五，今年30岁，手机号：13900139000")]
result_typeddict = agent_typeddict.invoke({"messages": message})
print("3. TypedDict结果:")
print(result_typeddict["structured_response"])
print("-" * 50)


# ============================================================
# 4. Json Schema (直接传入JSON Schema字典)
# ============================================================
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# 定义JSON Schema
user_info_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "用户姓名"
        },
        "age": {
            "type": "integer",
            "description": "用户年龄"
        },
        "phone": {
            "type": "string",
            "description": "手机号"
        }
    },
    "required": ["name", "age", "phone"]
}

# 创建代理
agent_json_schema = create_agent(
    model=llm,
    system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
    response_format=user_info_schema
)
message = [HumanMessage(content="我的名字叫赵六，今年28岁，手机号：13700137000")]
result_json_schema = agent_json_schema.invoke({"messages": message})
print("4. Json Schema结果:")
print(result_json_schema["structured_response"])
print("-" * 50)