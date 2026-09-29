import os

import numexpr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, ToolCallLimitMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# 加载环境变量
load_dotenv()

# 构建模型
llm = ChatOpenAI(model="kimi-k2.6",
                 api_key=os.getenv("MOONSHOT_API_KEY"),
                 base_url=os.getenv("MOONSHOT_BASE_URL"))


# 创建工具
@tool
def calculator(expression: str):
    """
    一个数学计算工具
    """
    return f"计算结果:{numexpr.evaluate(expression).item()}"


# 定义对应的工具列表
tools = [calculator, TavilySearch(max_results=1)]

# 创建Agent
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[TodoListMiddleware(),
                # 限制工具调用中间件，限制tavily_search工具在一次任务中只能被调用两次
                ToolCallLimitMiddleware(tool_name="tavily_search", run_limit=2)
                ],
)

result = agent.invoke(
    {"messages":
        {
            "role": "user",
            "content": "请帮我查询一下目前最新的小米su7的最低价格，在对比尚界z7的最低价格，他们的价格相差多少？"
                       "请使用待办事项"
        }
    }
)
print(result)
print(result["todos"])
