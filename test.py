# 临时跑一行，确认能拿到向量
from adapter import OpenAICompatibleEmbedding
import os
emb = OpenAICompatibleEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("EMBEDDING_MODEL"),
)
print(len(emb.get_text_embedding("你好")))  # 能正常输出维度，如 1024/1536/3072