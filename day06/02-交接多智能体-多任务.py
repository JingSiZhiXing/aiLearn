from typing import Callable
from typing_extensions import NotRequired

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)

from langchain.messages import (
    HumanMessage,
    ToolMessage,
)

from langchain.tools import (
    tool,
    ToolRuntime
)

import os
from dotenv import load_dotenv

from langchain_qwq import ChatQwen

load_dotenv()

# ======================================================
# 1. Model
# ======================================================

model = ChatQwen(
    model="qwen3.7-max-2026-05-20",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    thinking_budget=16,
)


# ======================================================
# 2. State
# ======================================================

class TutorState(AgentState):
    # 当前执行Agent
    current_agent: NotRequired[str]
    # 剩余任务
    pending_tasks: NotRequired[list[dict]]
    # 已完成结果
    answers: NotRequired[list[dict]]


# ======================================================
# 3. Handoff交接
# ======================================================

def make_handoff_tool(target: str, description: str, ):
    @tool(
        f"transfer_to_{target}",
        description=description
    )
    def handoff(
            runtime: ToolRuntime[
                None,
                TutorState
            ]
    ) -> Command:
        print(f"开始交接, 转交给{target}")
        return Command(
            update={
                "current_agent": target,
                "messages": [
                    ToolMessage(
                        content=f"转交给{target}",
                        tool_call_id=
                        runtime.tool_call_id
                    )
                ]
            }
        )

    return handoff


transfer_to_math = make_handoff_tool("math", "当问题是数学题时调用")
transfer_to_chinese = make_handoff_tool("chinese", "当问题是语文题（古诗文/阅读/写作）时调用")
transfer_to_english = make_handoff_tool("english", "当问题是英语题（语法/单词/翻译）时调用")
transfer_to_triage = make_handoff_tool("triage", "当问题不属于你的学科范围时，转回分诊台")


# ======================================================
# 4. 分诊工具
# ======================================================

@tool
def create_tasks(
        tasks: list[dict],
        runtime: ToolRuntime[
            None,
            TutorState
        ]
):
    """
    创建任务列表

    示例:

    [
      {
        "agent":"math",
        "task":"解x+5=12"
      },
      {
        "agent":"chinese",
        "task":"写李白诗句"
      }
    ]

    """
    # 取出第一个任务
    first = tasks[0]

    return Command(
        update={
            "pending_tasks": tasks[1:], # 去除第一个任务之后的所有任务进行更新
            "current_agent":
                first["agent"],
            "messages": [
                ToolMessage(
                    content=
                    f"任务拆分完成，进入{first['agent']}",
                    tool_call_id=
                    runtime.tool_call_id
                )
            ]
        }
    )


# ======================================================
# 5. 任务完成工具
# ======================================================

@tool
def finish_task(
        answer: str,
        runtime: ToolRuntime[
            None,
            TutorState
        ]
):
    """
    完成当前任务并保存处理结果。

    当你作为专业Agent完成自己的子任务后，
    必须调用此工具提交最终答案。

    参数:
        answer:
            当前Agent生成的任务结果。

    示例:
        数学Agent:
        answer="x+5=12，所以x=7"

        语文Agent:
        answer="《静夜思》的作者是李白"

    调用后系统会:
    1. 保存当前Agent的回答
    2. 判断是否还有未完成任务
    3. 如果存在剩余任务，将控制权交给下一个Agent
    4. 如果所有任务完成，则结束流程
    """
    pending = runtime.state.get(
        "pending_tasks",
        []
    )
    answers = runtime.state.get(
        "answers",
        []
    )
    current = runtime.state.get(
        "current_agent"
    )

    answers.append({
        "agent": current,
        "answer": answer
    })
    # 还有任务

    if pending:
        # 取出下一个任务，将待执行的任务列表中的下一个任务删除掉
        next_task = pending[0]
        return Command(
            update={
                "answers": answers,
                "pending_tasks": pending[1:],
                "current_agent": next_task["agent"],
                "messages": [
                    ToolMessage(
                        content=f"任务完成，答案已保存：{answer}",
                        tool_call_id=runtime.tool_call_id
                    )
                ],
            }
        )
    # 全部完成
    return Command(
        update={
            "answers": answers,
            "current_agent": "triage",
            "messages": [
                ToolMessage(
                    content=f"全部任务完成",
                    tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


# ======================================================
# 6. Agent配置
# ======================================================

AGENT_CONFIG = {
    "triage": {
        "prompt":
            """
            你是分诊Agent。
            
            你的任务：
            1. 分析用户问题。
            2. 如果包含多个领域[math, chinese, english]，拆成多个任务。
            3. 当所有任务都完成之后，进行总结回复
            
            例如：
            用户：
            解数学题，并写李白诗句
            调用create_tasks：
            [
                {
                    "agent":"math",
                    "task":"解数学题"
                },
                {
                    "agent":"chinese",
                    "task":"李白诗句"
                }
            ]
            """,
        "tools": [
            create_tasks
        ]
    },
    "math": {
        "prompt":
            """
            你是数学专家。
            只处理数学任务。
            完成后调用finish_task保存答案。
            """,
        "tools": [
            finish_task
        ]
    },
    "chinese": {
        "prompt":
            """
            你是语文专家。
            只处理语文任务。
            完成后调用finish_task保存答案。
            """,
        "tools": [finish_task]
    },
    "english": {
        "prompt":
            """
            你是英语专家。
            只处理英语相关任务：
            - 单词
            - 翻译
            - 语法
            完成后调用 finish_task 保存答案。
            """,

        "tools": [
            finish_task
        ]
    }
}


# ======================================================
# 7. Middleware
# ======================================================

@wrap_model_call
def dynamic_agent(
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse
        ]
):
    agent_name = request.state.get("current_agent", "triage")
    print("当前人设：", agent_name)
    config = AGENT_CONFIG[agent_name]

    request = request.override(
        system_prompt=
        config["prompt"],
        tools=config["tools"]
    )

    return handler(request)


# ======================================================
# 8. Create Agent
# ======================================================

agent = create_agent(
    model,
    tools=[
        create_tasks,
        finish_task,
        transfer_to_math,
        transfer_to_chinese,
        transfer_to_english,
        transfer_to_triage
    ],
    state_schema=TutorState,
    middleware=[
        dynamic_agent
    ],
    checkpointer=InMemorySaver()
)

# ======================================================
# 9. Test
# ======================================================

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id":
                "demo"
        }
    }

    q = """
    帮我解一下 x+5=26，并帮我翻译apple翻译成中文，在帮我写一个50字的中文座右铭
    """

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(q)
            ]
        },
        config
    )
    print("================")
    print(result)
    print()
    print(
        "最终答案:",
        result.get(
            "answers"
        )
    )

    """
        流程：
        1. 定义 任务列表状态、任务完成的答案列表状态
        2. 通过triage分诊台智能体，进行任务的规划，通过create_tasks工具进行将任务列表存储到自定义的状态中（会取出第一个工具作为本次执行的 prompt+tool）
        3. 通过 任务列表状态，依次的去执行任务，每执行完成一个任务就将答案进行存储
        4. 当任务完成之后，回到分诊台，读取答案列表状态进行总结回复。
        
        注意：交接模式原则-> 当前的Agent不适合做这个任务，需要找到匹配的Agent，将当前的状态+工具等内容传递给匹配的Agent（完全将控制权交给这个Agent）
    
    """
