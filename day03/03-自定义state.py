from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = ChatOpenAI(model="deepseek-v4-flash",
                 api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"))

# 自定义状态state
class CustomAgent(AgentState):
    # 添加额外的state属性
    user_name: str
    hobby: dict

agent = create_agent(
    model=llm,
    state_schema=CustomAgent,
    checkpointer=InMemorySaver()  # 添加短期记忆
)

# 定义一个线程配置，指定线程ID为"user_1"，这样在同一线程中进行的对话可以共享状态
thread_config = {"configurable": {"thread_id": "user_1"}}

agent.invoke(
    {
        "messages": [{"role": "user", "content": "你好，我的名字叫做初见！"}],
        # 在输入参数中直接传递用户的ID、名字和爱好等信息，这些信息会被存储在Agent的State中，并且在同一线程中进行的对话可以共享这些状态信息？
        "user_name": "初见",
        "hobby": {"sport": "basketball", "music": "pop"}
    },
    thread_config,
)

# 可以思考下：用户问题 -> 我的名字叫初见，我的爱好有：唱歌、打篮球。  |  帮我查询一下用户id为user_1的用户的爱好是什么？
# 用户在通过自然语言说他的名字和爱好时，Agent能够正确地将这些信息存储在State中，
# 并且在后续的对话中能够根据用户的ID来查询和使用这些信息，从而实现个性化的对话体验.

# 通过工具
