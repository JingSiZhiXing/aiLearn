from typing import Callable
from typing_extensions import NotRequired

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
import os
from dotenv import load_dotenv
from langchain_qwq import ChatQwen

load_dotenv()

model = ChatQwen(
    model="qwen3.7-max-2026-05-20",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    thinking_budget=16,
)


# ============================================================================
# 1. State：active_agent 就是"现在该谁接待"的指针
# ============================================================================
class TutorState(AgentState):
    active_agent: NotRequired[str]  # "triage" / "math" / "chinese" / "english"


# ============================================================================
# 2. 转交工具
# ============================================================================

def make_handoff_tool(target: str, description: str):
    @tool(f"transfer_to_{target}", description=description)
    def _handoff(runtime: ToolRuntime[None, TutorState]) -> Command:
        print(f"开始交接, 转交给{target}")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"已转交给 {target} 专家处理",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "active_agent": target,  # <-- 只是改了"指针"，没有真正跳转到别的节点
            }
        )

    return _handoff


transfer_to_math = make_handoff_tool("math", "当问题是数学题时调用")
transfer_to_chinese = make_handoff_tool("chinese", "当问题是语文题（古诗文/阅读/写作）时调用")
transfer_to_english = make_handoff_tool("english", "当问题是英语题（语法/单词/翻译）时调用")
transfer_to_triage = make_handoff_tool("triage", "当问题不属于你的学科范围时，转回分诊台")

# ============================================================================
# 3. 配置表：每个"角色"对应的人设 + 能用的工具
#    跟官方案例的 STEP_CONFIG 写法 1:1 对应
# ============================================================================
AGENT_CONFIG = {
    "triage": {
        "prompt": "你是学科答疑助手的分诊台。判断学生问题属于数学、语文还是英语，"
                  "调用对应的 transfer_to_xxx 工具转交，不要自己回答问题。",
        "tools": [transfer_to_math, transfer_to_chinese, transfer_to_english],
    },
    "math": {
        "prompt": "你是数学专家，分步骤讲解数学题。"
                  "禁止回答除数学以外的任何问题"
                  "如果学生问题不是数学，调用 transfer_to_triage 转回分诊台，不要勉强回答。",
        "tools": [transfer_to_triage],
    },
    "chinese": {
        "prompt": "你是语文专家，讲解古诗文、阅读理解、写作问题。"
                  "禁止回答除语文以外的任何问题"
                  "如果学生问题不是语文，调用 transfer_to_triage 转回分诊台。",
        "tools": [transfer_to_triage],
    },
    "english": {
        "prompt": "你是英语专家，讲解语法、单词、翻译问题。"
                  "禁止回答除英语以外的任何问题"
                  "如果学生问题不是英语，调用 transfer_to_triage 转回分诊台。",
        "tools": [transfer_to_triage],
    },
}


# ============================================================================
# 4. 中间件：每次调用模型前，读 active_agent，去配置表查对应人设，换装
# ============================================================================
@wrap_model_call
def apply_active_agent(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    active_agent = request.state.get("active_agent", "triage")  # 默认从分诊台开始
    config = AGENT_CONFIG[active_agent]

    request = request.override(
        system_prompt=config["prompt"],
        tools=config["tools"],
    )
    return handler(request)


# ============================================================================
# 5. 组装 agent —— 只有一个 create_agent
# ============================================================================
all_tools = [transfer_to_math, transfer_to_chinese, transfer_to_english, transfer_to_triage]

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=TutorState,
    middleware=[apply_active_agent],
    checkpointer=InMemorySaver(),
)

# ============================================================================
# 6. 跑一遍流程，观察 active_agent 怎么来回切换
# ============================================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-tutor-mw"}}

    turns = [
        "帮我解一下这道题：x + 5 = 12，x等于多少？",  # triage -> math
        "顺便问下，'明月几时有'诗句的下一句是什么？",  # math 发现不对口 -> 转回triage -> chinese
        "apple的复数形式怎么写？",  # chinese 发现不对口 -> 转回triage -> english
    ]

    for user_input in turns:
        result = agent.invoke({"messages": [HumanMessage(user_input)]}, config)
        last_msg = result["messages"][-1]
        print(f"\n学生: {user_input}")
        print(f"回复: {last_msg.content}")
        print(f"当前 active_agent: {result.get('active_agent')}")