from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatQwen(
    model="kimi-k2.6",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
# 在线的图片
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个图像的内容"},
        {"type": "image", "url": "https://bkimg.cdn.bcebos.com/pic/2e2eb9389b504fc2ef3eb893ebdde71191ef6dc6?x-bce-process=image/format,f_auto/quality,Q_70/resize,m_lfit,limit_1,w_536"},
    ]
}
response = llm.invoke([message])
print(response.content_blocks)
