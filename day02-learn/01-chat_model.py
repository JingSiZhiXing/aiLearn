from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os


load_dotenv()


llm = init_chat_model(model="kimi-k2.6",
                api_key = os.getenv("MOONSHOT_API_KEY"),
                base_url = os.getenv("MOONSHOT_BASE_URL"),
                model_provider="openai") # 指定模型提供商为OpenAI，千问在langchain中并没有原生支持，可以使用openai兼容模式

result = llm.invoke("哈喽啊")

print(result.content)

