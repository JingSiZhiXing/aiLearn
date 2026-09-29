import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(model="kimi-k2.6",
                 api_key=os.getenv("MOONSHOT_API_KEY"),
                 base_url=os.getenv("MOONSHOT_BASE_URL"))


# 自定义状态state
class CustomAgent(AgentState):
    # 添加额外的state属性
    userName: str
    hobby: dict


agent = create_agent(
    model=llm,
    state_schema=CustomAgent,
    checkpointer=InMemorySaver())  # 添加短期记忆

threadConfig = {"configurable": {"thread_id": "user_3"}}

messages = {
    "messages": [
        {"role": "user", "content": "你好，我的名字叫哈喽"}
    ],
    # 在输入参数中直接传递用户的ID、名字和爱好等信息，
    # 这些信息会被存储在Agent的State中，并且在同一线程中进行的对话可以共享这些状态信息？
    "userName": "哈喽",
    "hobby": {"爱好": "coding"}
}

result1 = agent.invoke(messages, threadConfig)

print(result1)

result2 = agent.invoke({"messages": [HumanMessage(content="我的名字和爱好是什么")]}, threadConfig)

print(result2)

# 可以思考下：用户问题 -> 我的名字叫初见，我的爱好有：唱歌、打篮球。  |  帮我查询一下用户id为user_1的用户的爱好是什么？
# 用户在通过自然语言说他的名字和爱好时，Agent能够正确地将这些信息存储在State中，
# 并且在后续的对话中能够根据用户的ID来查询和使用这些信息，从而实现个性化的对话体验.
