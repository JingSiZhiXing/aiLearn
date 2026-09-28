from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import TodoListMiddleware, ToolCallLimitMiddleware
import numexpr
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

model = "qwen3.7-max-2026-05-20"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")
# 初始化qwen模型
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=api_base_url,
    temperature=0.1
)


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
