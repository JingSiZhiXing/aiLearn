# 使用init_chat_model调用qwen模型，模型不会进行思考
# 通过ChatQwen进行调用qwen模型，就能开启模型的思考功能

# 和智能体没关系，，，，，通过langchain框架去使用模型


from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatQwen(
    model="kimi-k2.6",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    enable_thinking=False
)

print(llm.invoke("你好，我是初见"))




