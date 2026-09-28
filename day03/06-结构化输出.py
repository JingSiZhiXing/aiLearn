# """
# langchain支持四种方式去支持模型原生结构化输出
# 1.Pydantic模型  -提供数据校验功能、自动转化等功能
# 2.Dataclass    -是Python标准库自带的，主要用于简化类的定义
# 3.Typeddict    -用于约束字典（Dict）的键和值的类型,运行时就是一个普通的字典
# 4.json Schema  -JSON格式规范
#
# """
# from langchain_qwq import ChatQwen
# from dotenv import load_dotenv
# import os
#
# # 加载环境变量
# load_dotenv()
#
# # 初始化模型
# llm = ChatQwen(
#     model="deepseek-v4-flash",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=os.getenv("DASHSCOPE_BASE_URL")
# )
#
# # 使用qwen模型的结构化输出需要遵守以下内容
# """
# 在请求体中设置 response_format 参数即可开启结构化输出，需满足以下两个条件：
# 1.设置response_format参数：将 response_format 参数设置为 {"type": "json_object"}。
# 2.提示词中包含JSON关键词：System Message 或 User Message 中必须包含"JSON"关键词（不区分大小写），否则API会返回错误：
#     'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.
# """
#
# # ============================================================
# # 1. Pydantic模型
# # ============================================================
# from pydantic import BaseModel, Field
# from langchain.agents import create_agent
# from langchain.messages import HumanMessage
# from langchain.agents.structured_output import ProviderStrategy
#
# class UserInfoPydantic(BaseModel):
#     """用户信息"""
#     name: str = Field(description="用户姓名")
#     age: int = Field(description="用户年龄")
#     phone: str = Field(description="手机号")
#
# # 创建代理
# agent_pydantic = create_agent(
#     model=llm,
#     system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
#     response_format=ProviderStrategy(UserInfoPydantic)
# )
# message = [HumanMessage(content="我的名字叫张三，今年20岁，手机号：18578656489")]
# result_pydantic = agent_pydantic.invoke({"messages": message})
# print("1. Pydantic模型结果:")
# print(result_pydantic["structured_response"])
# print("-" * 50)
#
#
# # ============================================================
# # 2. Dataclass (Python标准库)
# # ============================================================
# from dataclasses import dataclass, field
# from langchain.agents import create_agent
# from langchain.messages import HumanMessage
#
# @dataclass
# class UserInfoDataclass:
#     """用户信息"""
#     name: str = field(metadata={"description": "用户姓名"})
#     age: int = field(metadata={"description": "用户年龄"})
#     phone: str = field(metadata={"description": "手机号"})
#
# # 创建代理
# agent_dataclass = create_agent(
#     model=llm,
#     system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
#     response_format=UserInfoDataclass
# )
# message = [HumanMessage(content="我的名字叫李四，今年25岁，手机号：13800138000")]
# result_dataclass = agent_dataclass.invoke({"messages": message})
# print("2. Dataclass结果:")
# print(result_dataclass["structured_response"])
# print("-" * 50)
#
#
# # ============================================================
# # 3. TypedDict (类型约束字典)
# # ============================================================
# from typing import TypedDict
# from langchain.agents import create_agent
# from langchain.messages import HumanMessage
#
# class UserInfoTypedDict(TypedDict):
#     """用户信息"""
#     name: str   # 用户姓名
#     age: int    # 用户年龄
#     phone: str  # 手机号
#
# # 创建代理
# agent_typeddict = create_agent(
#     model=llm,
#     system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
#     response_format=UserInfoTypedDict
# )
# message = [HumanMessage(content="我的名字叫王五，今年30岁，手机号：13900139000")]
# result_typeddict = agent_typeddict.invoke({"messages": message})
# print("3. TypedDict结果:")
# print(result_typeddict["structured_response"])
# print("-" * 50)
#
#
# # ============================================================
# # 4. Json Schema (直接传入JSON Schema字典)
# # ============================================================
# from langchain.agents import create_agent
# from langchain.messages import HumanMessage
#
# # 定义JSON Schema
# user_info_schema = {
#     "type": "object",
#     "properties": {
#         "name": {
#             "type": "string",
#             "description": "用户姓名"
#         },
#         "age": {
#             "type": "integer",
#             "description": "用户年龄"
#         },
#         "phone": {
#             "type": "string",
#             "description": "手机号"
#         }
#     },
#     "required": ["name", "age", "phone"]
# }
#
# # 创建代理
# agent_json_schema = create_agent(
#     model=llm,
#     system_prompt="你是一个信息提取助手。请始终以JSON格式输出提取的结构化数据。",
#     response_format=user_info_schema
# )
# message = [HumanMessage(content="我的名字叫赵六，今年28岁，手机号：13700137000")]
# result_json_schema = agent_json_schema.invoke({"messages": message})
# print("4. Json Schema结果:")
# print(result_json_schema["structured_response"])
# print("-" * 50)
#
#
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatQwen(
    model="qwen3.6-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
# ToolStrategy也支持 pydantic、Dataclass、TypedDict、JSON Schema四种方式，使用方式和模型原生策略一致
# 创建模型类
class ProductReview(BaseModel):
    """产品评价分析结果"""
    rating: int | None = Field(description="产品评分", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="评价的情感倾向")
    key_points: list[str] = Field(description="评价的关键要点")

# 创建代理
agent = create_agent(
    model=llm,
    response_format=ToolStrategy(ProductReview)
)

# 接触多智能体就用的多了
# 主智能体  -> {"意图识别": "子智能体1"}   结构化的提取作为一个工具或者子智能体进行使用
# 子智能体1 子智能体2 子智能体3

# 下一个工具的输入参数    是上一个工具的输出内容-> {....}

# 执行代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "分析这条产品评价：'很棒的5星产品。发货很快，但是有点贵'"}]
})
print(result["structured_response"])
# 注意，使用ToolStrategy进行结构化输出，messages中最后一条消息会以ToolMessage结尾
"""
ToolMessage(content="Returning structured response: rating=5 sentiment='positive' key_points=['产品很棒', '发货速度快', '价格偏高']", 
    name='ProductReview', id='475c134e-39c7-44d3-a2ee-563b3c05c069', tool_call_id='call_07384fcbdd3b494ca4b7aeff')]
"""
print(result)