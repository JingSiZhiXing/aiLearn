# 模型进行工具调用
from langchain_qwq import ChatQwen
from langchain.tools import tool
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()


# 创建一个工具
@tool
def get_user_info(user_name: str) -> str:
    """根据用户名获取用户详细信息

    Args:
        user_name (str): 用户吗

    Returns: 用户详细信息
    """
    user_info = {
        "user_name": user_name,
        "age": 20
    }
    return f"用户信息如下：{user_info}"

# 初始化模型
llm = ChatQwen(
    model="kimi-k2.6",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
# 模型绑定工具，返回一个新的模型对象
model_with_tools  = llm.bind_tools([get_user_info])
# 模型并不会自己调用工具，只会返回我要调用哪个工具

response = model_with_tools.invoke("我的名字叫张三，帮我查询我的详细信息")
for tool_call in response.tool_calls:
    # 查看模型所做的工具调用
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")



# function call  函数调用功能
