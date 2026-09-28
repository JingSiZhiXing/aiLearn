from langchain.tools import tool
import numexpr


# 函数名称会被当作工具名称，函数的参数会当作工具的参数，函数返回值会当作工具的输出结果
# 函数的文档字符串会把当作工具的描述
# 函数名称也需要有意义

# 文档字符串需要清晰的去描述工具的功能和输入输出参数的含义，保证Agent能够正确的识别和调用工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息
    Args:
        city(str): 城市名称
    Returns:
        str: 城市的天气信息
    """
    return f"{city}的天气是晴朗的!"

# 如果不想使用文档字符串当作工具的描述信息

@tool("calculator", description="执行算术计算。用这个来解数学题。输入应该是一个数学表达式，例如 '2 + 2' 或 'sqrt(16)'。")
def calc(expression: str) -> str:
    """计算数学表达式

    Args:
        expression (str): 数学表达式

    Returns:
        str: 计算结果
    """
    return str(numexpr.evaluate(expression).item())

# 还可以使用Pydantic模型来定义工具的输入参数，这样可以更清晰地描述工具的输入结构和类型，并且在Agent调用工具时能够进行参数验证和自动补全
from pydantic import BaseModel, Field


# Pydantic模型就是在进入函数的时候，先去判断参数是否符合要求
class CalculateInput(BaseModel):
    operation: str = Field(description="要执行的运算类型，如 'add' 或 'multiply'")
    a: float = Field(description="第一个操作数")
    b: float = Field(description="第二个操作数")

@tool("calculator", args_schema=CalculateInput)
def calculate(operation: str, a: float, b: float) -> str:
    """当你需要进行数学计算时使用此工具"""
    if operation == "add":
        return str(a + b)
    elif operation == "multiply":
        return str(a * b)
    return "Unknown operation"

