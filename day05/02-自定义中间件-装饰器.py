from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
import json
from langgraph.runtime import Runtime
from typing import Callable
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

# 1.在进入智能体之前先进行敏感词过滤
# 2.在模型执行之后校验返回的格式

# 创建json格式验证
def repair_json_string(raw_str: str) -> str:
    # 1. 去掉 Markdown 的代码块标签
    raw_str = re.sub(r"```json\s*|```", "", raw_str).strip()

    # 2. 修复最常见的：对象或数组末尾多余的逗号
    # 匹配: , 后面跟着 } 或 ]
    raw_str = re.sub(r",\s*([}\]])", r"\1", raw_str)

    # 3. 简单的引号补全（针对属性名漏掉引号的情况）
    # 匹配: {后面或逗号后面 没写引号的 key
    raw_str = re.sub(r"([{,]\s*)([a-zA-Z0-9_]+)(\s*:)", r'\1"\2"\3', raw_str)

    return raw_str


# 创建上下文  frozen保证对象初始化后不能在被修改
@dataclass(frozen=True)
class Context:
    user_id: int
    user_permissions: str


# 导入所有的钩子装饰器
from langchain.agents.middleware import (
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call,
    ModelRequest,
    ModelResponse,
)


@before_agent(can_jump_to="end")  # 提供当前可以调整的地方（提供选择）
def manage_human_message_before_agent(state: AgentState, runtime: Runtime[Context]):
    """
    在Agent启动之前调用
    can_jump_to：可以提前结束中间件
        'end': 跳转到代理执行的结尾（或第一个 after_agent 钩子）
        'tools': 跳转到工具节点
        'model': 跳转到模型节点（或第一个 before_model 钩子）

    Runtime:用来传递全局变量（只是用来读取的内容-上下文的常量），还经常用来传递数据库连接池、日志对象一些配置信息
    """

    # 从后往前找第一个类别为 HumanMessage 的消息
    user_content = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_content = message.content
            break
    # 打印用户最新的问题
    print(f"在before_agent中，用户最新问题：{user_content}")
    # 1.处理用户敏感用词
    sensitive_words = ["TM", "TMD", "CNM", "挂了", "垃圾"]
    # 可以通过模型判断
    if any(word in user_content.upper() for word in sensitive_words):
        # 发现敏感词，直接构造一个 AI 响应，不再交给模型思考
        return {
            "messages": [AIMessage(content="检测到不当言论，请文明交流。")],
            "jump_to": "end"  # 直接结束这次对话
        }
    # 2.vip用户特殊处理
    # 获取用户权限
    user_permissions = runtime.context.user_permissions
    if user_permissions == "vip":
        # vip权限能够进行所有知识库的访问
        print("我是VIP用户，能够查看全部的内容")
    else:
        print("我是普通用户，能够查看部分的内容")

    return None


@after_model
def fix_json_structure(state: AgentState, runtime: Runtime[Context]):
    # 1. 获取模型最后一条回复
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return
    # 这里我们模拟模型返回错误的json
    # raw_content = last_message.content
    raw_content = """
        ```json
        {
          "user_id": "123",
          "action": "send_package",
          "items": ["book", "pen"],
        }
    """
    print(f"开始进行json格式修复:{raw_content}")
    try:
        # 尝试直接解析，如果成功说明不需要修复
        json.loads(raw_content)
    except json.JSONDecodeError:
        # 2. 如果解析失败，执行修复逻辑
        fixed_content = repair_json_string(raw_content)
        print(f"json格式修复完成：{fixed_content}")

        try:
            # 再次验证修复结果
            json.loads(fixed_content)
            # 3. 【关键】写回消息对象
            # last_message.content = fixed_content
            # 也可以记录一个标记位说明发生过修正
            # last_message.additional_kwargs["is_fixed"] = True
        except Exception:
            # 如果修复后还是不行，可以抛出异常触发重试，或记录错误
            pass
    # 通过return 返回字典可以修改状态中的内容
    return {"messages": state["messages"]}


@wrap_model_call
def smart_model_wrapper(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]  # 代表handler是一个函数
) -> ModelResponse:
    user_preference = "用户喜欢二次元"
    context_msg = SystemMessage(content=f"用户偏好：{user_preference}。请根据此偏好回答问题。")
    # 通过override去覆盖一个新请求(只是一个临时指令，并不会添加到历史对话中)
    # 临时换一些配置，只针对本次调用模型
    new_request = request.override(
        system_message=context_msg,
        tools=[]
    )

    # 调用真正执行模型的内容
    response = handler(new_request)
    # 模拟 after_model 的逻辑：结构化修正 (可选)

    return response

# 创建智能体
agent = create_agent(model=llm,
                     middleware=[manage_human_message_before_agent,
                                 fix_json_structure])
result = agent.invoke({"messages": [HumanMessage("你好，我是初见")]},
                      context=Context(user_permissions="vip", user_id=1))
# 获取AI回复
# print(result["messages"][-1].content)
print(result)
