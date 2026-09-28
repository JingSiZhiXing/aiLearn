from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
# 加载环境变量
load_dotenv()

# 初始化模型
llm = init_chat_model(model="kimi-k2.6",
                      api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url=os.getenv("DASHSCOPE_BASE_URL"),
                      model_provider="openai")  # 指定模型提供商为OpenAI，千问在langchain中并没有原生支持，可以使用openai兼容模式

# 普通对话
result = llm.invoke("你好！")
print(result.content)
# 流式输出
# for chunk in llm.stream("请告诉我一个笑话！"):
#     print(chunk.text, end="", flush=True)
# 批次对话-并行调用Agent完成任务
# responses = llm.batch([
#     "为什么天空是蓝色的？",
#     "为什么鸡有翅膀却不能飞？",
#     "什么是langchain ？"
# ])
# for response in responses:
#     print(response.content)
