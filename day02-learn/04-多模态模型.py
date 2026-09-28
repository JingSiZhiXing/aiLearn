import os

from langchain_qwq import ChatQwen
from dotenv import load_dotenv

load_dotenv()

llm = ChatQwen(model="kimi-k2.6",
         api_key=os.getenv("MOONSHOT_API_KEY"),
         base_url=os.getenv("MOONSHOT_BASE_URL"))

message = {
    "role":"user",
    "content":[
        {"type":"text","text":"描述这个图像的内容"},
        {"type":"image","url":"https://bkimg.cdn.bcebos.com/pic/2e2eb9389b504fc2ef3eb893ebdde71191ef6dc6?x-bce-process=image/format,f_auto/quality,Q_70/resize,m_lfit,limit_1,w_536"}
    ]
}
# {'error': {'message': 'Invalid request: unsupported image url
response = llm.invoke([message])

print(response)