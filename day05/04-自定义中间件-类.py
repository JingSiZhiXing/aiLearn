from typing import Callable, Optional, Dict, Any
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.runtime import Runtime
from dataclasses import dataclass
import re
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

"""
在什么时候使用类去自定义中间件：
1.为同一个钩子定义同步和异步的实现
2.在单个中间件中需要多个钩子
3.需要复杂的配置
4.在初始化时配置，实现项目重用
"""


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


# dataclass会自动创建init、repr、eq方法，frozen能够保证对象初始化之后不能修改
@dataclass(frozen=True)
class Context:
    user_id: int
    user_permissions: str


class UnifiedAgentMiddleware(AgentMiddleware):
    def __init__(self, sensitive_words: list = None):
        # 可以在构造函数中传入配置，如敏感词库、数据库连接等
        self.sensitive_words = sensitive_words or ["TM", "TMD", "CNM", "挂了", "垃圾"]

    # --- Agent 级别钩子 ---
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
        """在 Agent 逻辑开始前执行（敏感词检查 & 权限验证）"""
        user_content = ""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                user_content = message.content
                break

        print(f"[Before Agent] 检查内容: {user_content}")

        # 1. 敏感词拦截
        if any(word in user_content.upper() for word in self.sensitive_words):
            print(12)
            return {
                "messages": [AIMessage(content="检测到不当言论，请文明交流。")],
                "jump_to": "end"
            }

        # 2. 权限处理
        user_permissions = runtime.context.user_permissions
        status = "VIP" if user_permissions == "vip" else "普通"
        print(f"[Before Agent] {status}用户访问")

        return None

    def after_agent(self, state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
        """在 Agent 逻辑结束后执行"""
        print("[After Agent] Agent 执行完毕，准备返回。")
        return None

    # --- Model 级别钩子 ---
    def before_model(self, state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
        pass

    def after_model(self, state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
        """在模型调用后修复数据格式"""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # 模拟格式修复
        raw_content = last_message.content
        # 假设这里触发了修复逻辑（示例中写死一段错误内容演示）
        if "{" in raw_content:
            print(f"[After Model] 尝试修复 JSON...")
            fixed_content = repair_json_string(raw_content)
            # 更新消息内容
            last_message.content = fixed_content

        return {"messages": state["messages"]}

    # --- 包装器钩子 (高级拦截) ---

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """深度干预模型请求与响应"""
        print("[Wrap Model] 动态覆盖系统提示词词")
        # 这里的 override 不会改变 state 里的历史记录，仅对本次请求有效
        new_request = request.override(
            system_message=SystemMessage(content="用户偏好：二次元")
        )

        # 执行实际的模型调用
        response = handler(new_request)
        return response



# 1. 实例化中间件对象
my_middleware = UnifiedAgentMiddleware()

# 2. 传入 create_agent
agent = create_agent(
    model=llm,
    middleware=[my_middleware], # 直接传入实例
)

# 3. 调用
result = agent.invoke(
    {"messages": [HumanMessage("你好，我是初见TM")]},
    context=Context(user_permissions="vip", user_id=1)
)

print(result["messages"][-1].content)