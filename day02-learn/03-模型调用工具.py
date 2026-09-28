from langchain_qwq import ChatQwen
from langchain.tools import tool
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

@tool
def get_user_info(user_name: str)-> str:
    """根据用户名获取用户详情信息
    Args:
        user_name(str):用户名

    Returns: 用户详细信息
    """
    user_info = {
        "user_name": user_name,
        "age": 20
    }
    return f"用户信息如下：{user_info}"


llm = ChatQwen(
    model="kimi-k2.6",
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url=os.getenv("MOONSHOT_BASE_URL")
)

# 模型绑定工具，返回一个新的模型对象
llm_with_tools = llm.bind_tools([get_user_info])

result = llm_with_tools.invoke("获取用户信息：张三")
print(result)

for tool_call in result.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")


