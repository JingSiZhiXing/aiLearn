import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain_core.messages import ToolMessage
from langchain_qwq import ChatQwen
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

llm = ChatQwen(model="kimi-k2.6",
               api_key=os.getenv("MOONSHOT_API_KEY"),
               base_url=os.getenv("MOONSHOT_BASE_URL"))


# 自定义状态
class CustomAgent(AgentState):
    user_hobby: list[str]


# 自定义上下文的属性
@dataclass
class Context:
    user_id: str
    user_name: str


# 创建工具
# 在工具中使用runtime，runtime只存在一次任务中
@tool
def get_user_name(runtime: ToolRuntime[Context]):
    """获取用户姓名"""
    print(runtime.state["messages"])  # 获取对于的状态（短期记忆）
    return runtime.context.user_name  # 工具返回值会自动处理转成ToolMessage


# 创建一个内存的长期记忆
store = InMemoryStore()  # 存储容器-用户的长期记忆

# 创建一个内存的短期记忆（state）
checkpointer = InMemorySaver()


@tool
def set_user_hobby(hobby: list[str], runtime: ToolRuntime[Context]):
    """设置用户爱好"""
    print(f"提取的用户爱好：{hobby}")
    # state是短期记忆，要想长期保存需要使用长期记忆
    print(f"{runtime.config.get("configurable").get("thread_id")}", runtime.context.user_name)

    namespace = (runtime.context.user_id, "memories")  # 对应：(用户ID, 记忆类别)
    key = "user_profile"
    store.put(namespace=namespace,
              key=key,
              value={
                  "user_name": runtime.context.user_name,
                  "hobby": hobby
              })
    # 可以使用command在工具中更新状态，Command会在langgraph详细介绍
    return Command(
        update={  # 更新Agent的state状态
            "user_hobby": hobby,
            "messages": [
                ToolMessage(
                    content=f"已更新用户爱好：{hobby}",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


# 创建代理
agent = create_agent(
    model=llm,  # 模型
    tools=[get_user_name, set_user_hobby],
    state_schema=CustomAgent,  # 自定义state
    store=store,  # 长期记忆
    checkpointer=checkpointer,  # 短期记忆
    context_schema=Context,
)

config = {"configurable": {"thread_id": "user_1"}}

# Context上下文只存在一次任务中 ， runtime的生命周期就是Agent执行一次任务
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "我的名字叫什么"}]},
#     config=config,
#     context=Context(user_id="user1", user_name="哈喽")
# )
# print(response["messages"][-1].content)


print("------更新用户爱好------")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我目前的爱好喜欢编程、看电视、看球赛等"}]},
    config=config,
    context=Context(user_id="user1", user_name="哈喽")
)

print("长期记忆中存储的内容：", store.get(("user1", "memories"), "user_profile"))

"""
梳理流程：
    1. runtime 
        -> Agent在执行任务的时候，可以通过runtime在工具中获取（Context上下文【静态的配置信息】、Store长期记忆、State状态）
    2.自定义Context
        @dataclass  #
        class Context:   # 名字随便取   UserContext
            user_id: str  # 用户id
            user_name: str  # 用户名称
    3.在执行任务开始的时候将Context进行填充
        agent.invoke(
            {"messages": [{"role": "user", "content": "我目前的爱好喜欢编程、看电视、看球赛等"}]},
            config=config,
            context=Context(user_id="user1", user_name="哈喽")
        )
    4.在工具中通过ToolRuntime进行获取
        @tool
        def get_user_name(runtime: ToolRuntime[Context]):  # 在工具中使用runtime，runtime只存在一次任务中
            '''获取用户姓名'''
            print(runtime.state["messages"])  # 获取对于的状态（短期记忆）
            return runtime.context.user_name  # 工具返回值会自动处理转成ToolMessage
    5.想在工具中修改state，需要在工具函数返回Command对象
       return  Command(
            update={  # 更新Agent的state状态
                "user_hobby": hobby,
                "messages": [
                    ToolMessage(
                        content=f"已更新用户爱好：{hobby}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )

长期记忆：存储用户的一些行为习惯、偏好内容、用户画像（用户的购买力）；根据业务需求来。   存储用户喜欢的商品，，后续做商品推荐的时候，Agent会参考存储的长期记忆进行推荐

短期记忆：就是一个会话中的state，只要这个会话一直存在，state就是永久的

Context：就是一次任务所需要的配置信息内容

"""