from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any
from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatQwen(
    model="qwen3.7-max-2026-05-20",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

"""
    human.------------   50
    ai.------------      100
    human.------------   150
    ai.------------      200
    human.------------   50
    ai.------------      100

"""
# 不常用   会造成消息的丢失


@before_model
def custom_trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """裁剪消息，只保留最后几条消息，保证上下文窗口长度"""

    messages = state["messages"]

    print("目前消息的token长度-->", count_tokens_approximately(messages, chars_per_token=1.8))

    # 官方提供裁剪消息的工具，会根据配置自动裁剪消息
    trimmed_messages = trim_messages(
        messages=messages,  # 待裁剪的消息列表（通常是对话历史）
        max_tokens=2000,  # 裁剪后的消息总 Token 数上限，超过此值会丢弃旧消息
        token_counter=lambda msgs: count_tokens_approximately( # Token 计数器，使用 LangChain 内置的近似计数函数
            msgs,
            chars_per_token=1.8,  # 中文更接近 1.8 字符/Token
        ),
        # 其他可选值：
        #   - len: 按消息条数计数（此时 max_tokens 代表条数）
        #   - ChatOpenAI(model="gpt-4o"): 使用模型原生分词器（最精准但较慢）
        #   - 自定义函数: 自己实现的计数逻辑
        strategy="last",  # 裁剪策略：从末尾往前保留最新消息
        #   - "last": 保留最新的 N 个 Token（适合保留最近上下文）
        #   - "first": 保留开头的 N 个 Token（适合保留系统提示和早期指令）
        allow_partial=False,  # 是否允许截断单条消息
        #   - False（默认）: 消息要么完整保留，要么整条丢弃，不会出现"半条消息"
        #   - True: 允许截断消息，max_tokens 落在某条消息中间时会保留部分内容
        #     注意：strategy="last" 时保留后半部分，strategy="first" 时保留前半部分
        start_on="human",  # 强制裁剪后的消息以 HumanMessage 开头
        # 会从后往前找到第一个 HumanMessage，丢弃它之前的所有消息
        # 确保对话历史符合模型要求（通常以 Human 或 System+Human 开头）
        # 可选值: "human", "ai", "system", "tool", 或对应的 Message 类
        end_on=("human", "ai"),  # 强制裁剪后的消息以 HumanMessage/AIMessage 结尾 这里需要注意！
        # 会从前往后找到最后一个 HumanMessage/AIMessage，丢弃它之后的所有消息
        include_system=True,  # 是否保留第一条 SystemMessage（系统提示词）
        # 当 strategy="last" 时生效，会特殊保护位于索引 0 的 SystemMessage 不被裁剪
        # 建议设为 True，因为系统提示词包含给模型的核心指令
    )
    print("裁剪过的消息-->", trimmed_messages)
    # 删除所有消息
    RemoveMessage(id=REMOVE_ALL_MESSAGES),  # RemoveMessage删除消息的工具
    return {
        "messages": [
            *trimmed_messages  # 把列表拆成每一个元素填充到新的列表中
        ]
    }

agent = create_agent(
    model=llm,
    middleware=[custom_trim_messages],
    system_prompt="你是一个智能助手",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "你好，我叫初见"}, config)
agent.invoke({"messages": "请写一首关于月亮的诗句"}, config)
agent.invoke({"messages": "在写一首关于太阳的诗句"}, config)
final_response = agent.invoke({"messages": "我叫什么名字？"}, config)

for res in final_response["messages"]:
    res.pretty_print()