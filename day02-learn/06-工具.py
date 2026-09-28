from langchain.tools import tool
import numexpr


# 函数名称会被当作工具名称，函数的参数会当作工具的参数，函数返回值会当作工具的输出结果
# 函数的文档字符串会把当作工具的描述
# 函数名称也需要有意义

@tool
def get_weather(city: str) -> str:
    """获取指定城市天气信息
    Args:
        city(str):城市名称
    Returns:
        str:城市的天气信息
    """
    return f"{city}的天气是晴朗的"


@tool("calculator", description="执行算术计算。用这个来解数学题。输入应该是一个数学表达式，例如 '2 + 2' 或 'sqrt(16)'。")
def calc(exepression: str) -> str:
    """计算数学表达式
    Args:
        exepression(str):数学表达式
    Returns:
        str:数学表达式的结果
    """
    return str(numexpr.evaluate(exepression).item())


from pydantic import BaseModel, Field

# Pydantic模型就是在进入函数的时候，先去判断参数是否符合要求
class CalculateInput(BaseModel):
    operation: str = Field(description="要执行的运算类型，如 'add' 或 'multiply'"),
    a: float = Field(description="第一个数字"),
    b: float = Field(description="第二个数字")


@tool("calculator", args_schema=CalculateInput)
def calculate(operation: str, a: float, b: float) -> str:
    """当你需要进行数学计算时使用此工具"""
    if operation == "add":
        return str(a + b)
    elif operation == "multiply":
        return str(a * b)
    else:
        return "Invalid operation"


import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(model="kimi-k2.6",
                 api_key=os.getenv("MOONSHOT_API_KEY"),
                 base_url=os.getenv("MOONSHOT_BASE_URL"))

agent = create_agent(model=llm, tools=[get_weather,calc,calculate],
                     system_prompt="你是一个智能助手，能够根据用户的输入调用对应工具处理用户请求")

messages = {
    "messages": [
        SystemMessage(content="你是一个智能助手，能够根据用户的输入调用对应工具处理用户请求"),
        HumanMessage(content=" add 2 2等于几")
    ]
}
response = agent.invoke(messages)
print(response)