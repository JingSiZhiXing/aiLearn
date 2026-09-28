import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_qwq import ChatQwen
from langchain_tavily import TavilySearch

# =====================================================
# 初始化模型
# =====================================================
load_dotenv()

model = ChatQwen(
    model="qwen3.7-max-2026-05-20",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    thinking_budget=16,
)


# =====================================================
# 数学工具
# =====================================================

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。
    示例：
        "(52+18)*2"
    返回：
        140
    """
    import numexpr
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"计算失败：{e}"


# =====================================================
# 语文工具 - 搜索
# =====================================================

search_tool = TavilySearch()

# =====================================================
# 数学专家
# =====================================================

math_agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt="""
你是数学专家。
职责：
    - 数学计算
    - 数值推导
    - 公式计算
要求：
    遇到计算优先调用 calculator。
禁止：
    翻译。
    写作文。
"""
)

# =====================================================
# 英语专家
# =====================================================

english_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
你是英语专家。
职责：
    - 中译英
    - 英译中
    - 英文润色
要求：
    保持准确。
禁止：
    做数学计算。
    写文章。
"""
)

# =====================================================
# 语文专家
# =====================================================

chinese_agent = create_agent(
    model=model,
    tools=[search_tool],
    system_prompt="""
你是语文专家。
职责：
    - 写作
    - 扩写
    - 总结
    - 搜集素材
要求：
    如果需要背景知识：
    优先搜索。
禁止：
    修改数学结果。
"""
)


# =====================================================
# Agent 包装成 Tool
# =====================================================

def agent_as_tool(
        name: str,
        title: str,
        description: str,
        agent,
):
    """
    将 Agent 包装成 Tool

    Supervisor：
    调用 Tool

    实际：
    Tool → 调 Agent
    """

    @tool(name, description=description)
    def execute(
            request: str,
    ) -> str:
        """
        调用子智能体。
        """

        print(f"\n开始执行：{name}")

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request
                    }
                ]
            }
        )

        print(f"完成执行：{name}， 结果：{result["messages"][-1].content}")

        return f"""
        【{title}】已完成任务：

        执行结果：{result["messages"][-1].content}
        """

    return execute


math_tool = agent_as_tool(
    name="math_expert",
    title="数学专家",
    description="数学专家智能体，负责数学计算",
    agent=math_agent,
)

english_tool = agent_as_tool(
    name="english_expert",
    title="英语专家",
    description="英语专家智能体，负责翻译",
    agent=english_agent,
)

chinese_tool = agent_as_tool(
    name="chinese_expert",
    title="语文专家",
    description="语文专家智能体，负责写作",
    agent=chinese_agent,
)

# =====================================================
# Supervisor
# =====================================================

supervisor = create_agent(
    model=model,
    tools=[
        math_tool,
        english_tool,
        chinese_tool,
    ],
    system_prompt="""
你是教育领域 Supervisor。
职责：
    分析用户需求。
    决定调用哪个专家。
可调用：
    1.数学专家  math_tool
    2.英语专家  english_tool
    3.语文专家  chinese_tool
禁止：
    自己直接回答问题。
    最终汇总结果。
"""
)

# =====================================================
# 运行
# =====================================================

if __name__ == "__main__":
    user_input = """
            帮我写一个猫咪的故事，字数控制在100字，并把故事翻译成英文
        """

    result = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )
    """
        main Agent -> chinese_expert[1.网页搜索 5条数据【5000 tokens】 2. 根据搜索的内容进行文章编写 3. 将编写好的文章返回主智能体]  -> main Agent
        上下文总结解决  模型上下文溢出
        因为主智能体只接收子智能体的结果，并不会接收子智能体的思考过程（执行的所有步骤）
        
        
        总结：subAgents -》 将子智能体封装成主智能体的工具，主智能体根据任务来进行工具调用
        
    """

    print("\n")
    print("=" * 80)
    print(result["messages"][-1].content)
