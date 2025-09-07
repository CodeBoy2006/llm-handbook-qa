# adapter.py
from __future__ import annotations
from typing import List, Optional

from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI, AsyncOpenAI
try:
    # pydantic v2/v1 都有 PrivateAttr
    from pydantic import PrivateAttr
except Exception:  # 极少数环境
    from pydantic.fields import PrivateAttr  # type: ignore


class OpenAICompatibleEmbedding(BaseEmbedding):
    """OpenAI-Compatible 嵌入适配器。
    直接调用 /embeddings，允许任意 model 字符串（Qwen/...、bge/...等）。
    """

    # 这些是 Pydantic 字段（允许通过 super().__init__ 传参）
    model_name: str
    timeout: int = 30
    max_batch: int = 256

    # 这些是 Pydantic 私有属性（实例化后赋值，不做校验/序列化）
    _client: OpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        timeout: int = 30,
        max_batch: int = 256,
    ) -> None:
        # 先初始化 Pydantic 字段
        super().__init__(model_name=model, timeout=timeout, max_batch=max_batch)
        # 再设置私有属性（OpenAI 客户端）
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ---------------- 必需的同步方法 ----------------
    def _get_text_embedding(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.model_name, input=text)
        return resp.data[0].embedding

    def _get_query_embedding(self, query: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.model_name, input=query)
        return resp.data[0].embedding

    def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch):
            chunk = texts[i : i + self.max_batch]
            resp = self._client.embeddings.create(model=self.model_name, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return out

    # ---------------- 必需的异步方法 ----------------
    async def _aget_text_embedding(self, text: str) -> List[float]:
        resp = await self._aclient.embeddings.create(model=self.model_name, input=text)
        return resp.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        resp = await self._aclient.embeddings.create(model=self.model_name, input=query)
        return resp.data[0].embedding

    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch):
            chunk = texts[i : i + self.max_batch]
            resp = await self._aclient.embeddings.create(model=self.model_name, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return out