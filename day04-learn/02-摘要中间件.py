import os


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()

# 构建模型
llm = ChatOpenAI(model="kimi-k2.6",
                 api_key=os.getenv("MOONSHOT_API_KEY"),
                 base_url=os.getenv("MOONSHOT_BASE_URL"))

# 配置工具
tools = [TavilySearch(max_results=1)]

# 摘要说明
"""
 SummarizationMiddleware(
            model=llm,  # 进行摘要的模型
            trigger=("tokens", 4000),  # 触发的阈值
            # 三个选项：
                ("messages", 50)  当消息条数达到50条就开始进行总结
                ("tokens", 3000)  当tokens长度达到3000就开始总结
                ("fraction", 0.8) 当模型的最大输入标记达到80%时触发总结
                他是有 AND  OR
                中括号[] 代表的OR
                花括号{} 代表的是AND
            keep=("messages", 20), # 条件控制
             # 要保留多少上下文信息 只能选择一个
                1.fraction- 要保留的模型上下文大小的比例
                2.tokens- 要保留的绝对令牌数量
                3.messages- 要保留的最近消息数量
        ),
"""

# 不用提示词，默认总结后是英文的摘要
SHORT_SUMMARY_PROMPT = """你是一个记忆压缩专家。
请将下方的对话历史压缩成一段简洁的背景摘要，保留以下核心：
1. 用户最终想要解决的问题是什么？
2. 已经执行了哪些关键步骤或得到了哪些结论？
3. 还有哪些待办事项？
4. 使用中文进行摘要

请直接输出摘要内容，不要包含任何开场白。

待压缩的对话：
{messages}
"""

# 构建智能体
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[SummarizationMiddleware(
        model=llm,
        SHORT_SUMMARY_PROMPT=SHORT_SUMMARY_PROMPT,
        trigger=[("tokens", 4000), ("messages", 10)],  # 触发条件 中括号[] 代表的OR
        keep=("messages", 2),  # 摘要后要保留多少上下文信息
    )],
    checkpointer=InMemorySaver()  # 短期记忆
)


def run_test():
    print("=== 开始 Agent 自动化测试 ===")
    config = {"configurable": {"thread_id": "1"}}

    # 场景 1：基础问答 + 工具调用（验证 Tavily 搜索是否正常）
    print("\n[测试点 1: 工具调用]")
    query_1 = "2026年3月最新的AI大模型技术趋势是什么？请列出3-4点 简单总结内容"
    print(f"用户: {query_1}")
    response_1 = agent.invoke({"messages": [{"role": "user", "content": query_1}]}, config)
    print(f"Agent 响应: {response_1}")

    # 场景 2：连续对话（验证上下文保留与中间件触发）
    # 我们故意发送一些长文本，模拟达到 4000 tokens 或 3 条消息的触发条件
    print("\n[测试点 2: 多轮对话与摘要中间件验证]")
    test_conversations = [
        "请记住我的名字叫‘浩英’，我是一名AI架构师。",
        "刚才我问的技术趋势中，哪个对医疗行业影响最大？",
        "请基于我们刚才聊到的所有内容，给我写一份200字的行业简报。"
    ]
    for i, user_input in enumerate(test_conversations):
        print(f"\n第 {i + 2} 轮对话输入: {user_input}")
        # 执行对话
        res = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config)
        print(f"Agent 响应: {res}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
