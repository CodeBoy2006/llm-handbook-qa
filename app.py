#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from adapter import OpenAICompatibleEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError


# -------------------- 环境 --------------------
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
COLLECTION = os.getenv("COLLECTION_NAME", "handbook")

# Embedding（检索阶段用，同构建阶段一致）
EMBED_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
if not EMBED_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 未设置")

Settings.embed_model = OpenAICompatibleEmbedding(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL, model=EMBED_MODEL)

# Chat（生成式回答）
CHAT_API_KEY = EMBED_API_KEY
CHAT_BASE_URL = EMBED_BASE_URL
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "30"))

oai_client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_BASE_URL, timeout=TIMEOUT_SEC)

# -------------------- 向量检索初始化 --------------------
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=collection)

index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=5)

# -------------------- Web --------------------
app = FastAPI(title="Handbook QA (Chroma + Retriever + Generator)")

# 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QAHit(BaseModel):
    score: float | None = None
    text: str
    metadata: Dict[str, Any]

class QAResponse(BaseModel):
    query: str
    results: List[QAHit]

class GenResponse(BaseModel):
    """
    统一 JSON 输出，便于前端做“每句带来源上标”的渲染：
    - answer: LLM 严格 JSON（见下方 schema）
    - chunk_map: 发送给 LLM 的 chunk_id -> 内容（已截断）
    - evidence: 原始节点元数据
    """
    query: str
    answer: Dict[str, Any]
    chunk_map: Dict[str, str]
    evidence: List[Dict[str, Any]]

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/answer", response_model=GenResponse)
def answer(
    q: str = Query(..., description="用户问题"),
    k: int = 5,
    max_tokens: int | None = None,
    temperature: float | None = None,
):
    """
    先检索，再把片段交给对话模型进行“基于资料的回答”。
    返回结构化 JSON（包含 conclusion 与 answer_list），并附上 chunk_map（chunk_id: 片段内容）供前端标注引用。
    """
    retriever.similarity_top_k = k
    nodes = retriever.retrieve(q)
    if not nodes:
        # 404：与旧行为一致。也可以改为返回空 JSON，由前端决定提示。
        raise HTTPException(status_code=404, detail="未检索到相关内容")

    # 组装上下文与 chunk_map：与 LLM 实际看到的内容保持一致
    chunk_map: Dict[str, str] = {}
    ordered_ids: List[str] = []
    context_lines: List[str] = []

    for i, n in enumerate(nodes):
        meta = n.node.metadata or {}
        # 优先使用已有 chunk_id；若无，则生成稳定占位 id
        cid = str(meta.get("chunk_id") or meta.get("id") or f"chunk_{i+1}")
        text = n.node.get_content() or ""
        # 控制每段长度，避免超上下文（可按需调整）
        if len(text) > 2000:
            text = text[:2000] + " ..."
        chunk_map[cid] = text
        ordered_ids.append(cid)
        # 以“[cid] 内容”形式提供给 LLM，便于引用
        context_lines.append(f"[{cid}]\n{text}")

    # ====== 提示词（严格 JSON 输出 + 结论优先 + 每句带 citations）======
    system_prompt = (
        "你是一个严格的《学生手册》助手。"
        "只依据给定的片段作答；不得编造或引入外部知识。"
        "请按指定 JSON schema 输出，且仅输出 JSON（不包含任何多余文本、Markdown 或解释）。"
    )

    # 新的 schema（给 LLM）
    # - conclusion: 直接答案（选择题给“A/B/C/D”，判断题给“正确/错误”）；一般性问题可为 null 或核心观点摘要。
    # - answer_list: 推理/依据，逐句原子化，句句给 citations（若“手册未明确规定”，则该句 citations 为空数组）。
    output_schema_hint = """
{
  "conclusion": "string | null",
  "answer_list": [
    {
      "sentence": "一句完整且尽量短的原子句。",
      "citations": ["chunk_1", "chunk_3"]
    }
  ]
}
""".strip()

    sources_block = "\n\n".join(context_lines)

    user_prompt = (
        "任务：基于《学生手册》片段，分析并解决用户问题。问题类型可能包括：信息查询、选择题、判断题等。"
        "输出严格 JSON，便于前端做逐句上标引用。\n\n"
        f"用户问题：{q}\n\n"
        "SOURCES（仅可引用以下 chunk_id）：\n\n"
        f"{sources_block}\n\n"
        "输出要求：\n"
        "1) 仅输出 JSON，schema 为：\n"
        f"{output_schema_hint}\n"
        "2) 核心要求：若为选择题或判断题，必须在 `conclusion` 字段给出明确最终答案（如“B”或“正确”）；"
        "一般性提问可为 `null` 或核心观点摘要。\n"
        "3) `answer_list`：将推理或回答过程切分为若干“原子句”，每句意义完整且尽量短。\n"
        "4) `citations`：每句必须给出相关的 chunk_id；若该句为“手册未明确规定”或常识性总结，则为 []。\n"
        "5) 严格性：不得出现 sources 之外的 chunk_id；不得出现空对象或额外字段；不得输出 Markdown 代码块标记。\n"
    )

    # 发起请求（优先 JSON mode；若不支持则回退）
    try:
        resp = oai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=(TEMPERATURE if temperature is None else temperature),
            max_tokens=(MAX_TOKENS if max_tokens is None else max_tokens),
            response_format={"type": "json_object"},
        )
        raw_answer = (resp.choices[0].message.content or "").strip()
    except (APIConnectionError, RateLimitError, APIStatusError) as e:
        raise HTTPException(status_code=502, detail=f"生成失败：{e}")
    except Exception:
        # 某些服务端不支持 response_format，降级重试一次
        try:
            resp = oai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=(TEMPERATURE if temperature is None else temperature),
                max_tokens=(MAX_TOKENS if max_tokens is None else max_tokens),
            )
            raw_answer = (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            raise HTTPException(status_code=502, detail=f"生成失败：{e}")

    # 解析与规范化 JSON；失败则兜底
    normalized_answer: Dict[str, Any]
    try:
        parsed = json.loads(raw_answer)

        # 兼容旧版 {"sentences": [...]}：自动转换为新 schema
        if isinstance(parsed, dict) and "sentences" in parsed and "answer_list" not in parsed:
            answer_list = []
            for s in parsed.get("sentences", []):
                if not isinstance(s, dict):
                    continue
                # 旧字段名 text -> sentence
                sentence = str(s.get("text", "")).strip()
                cits = s.get("citations", [])
                if not isinstance(cits, list):
                    cits = []
                # 仅允许已发送的 chunk_id
                cits = [str(x) for x in cits if str(x) in chunk_map]
                if sentence:
                    answer_list.append({"sentence": sentence, "citations": cits})
            normalized_answer = {"conclusion": None, "answer_list": answer_list}
        else:
            # 新版 schema 校验
            if not isinstance(parsed, dict):
                raise ValueError("JSON 根必须为对象")

            conclusion = parsed.get("conclusion", None)
            if conclusion is None:
                pass
            else:
                # 统一转为字符串
                conclusion = str(conclusion).strip()
                if conclusion == "":
                    conclusion = None

            alist = parsed.get("answer_list", [])
            if not isinstance(alist, list):
                raise ValueError("answer_list 必须是数组")

            normalized_list = []
            for item in alist:
                if not isinstance(item, dict):
                    continue
                # 支持容错（sentence / text）
                sentence = str(
                    item.get("sentence", item.get("text", ""))
                ).strip()
                cits = item.get("citations", [])
                if not isinstance(cits, list):
                    cits = []
                # 只允许已发送的 chunk_id，并去重保序
                filtered = []
                seen = set()
                for x in cits:
                    sx = str(x)
                    if sx in chunk_map and sx not in seen:
                        filtered.append(sx)
                        seen.add(sx)
                if sentence:
                    normalized_list.append({"sentence": sentence, "citations": filtered})

            normalized_answer = {"conclusion": conclusion, "answer_list": normalized_list}

    except Exception:
        # 兜底：将原始文本包一层；无法解析时不强行给结论
        all_ids = ordered_ids
        normalized_answer = {
            "conclusion": None,
            "answer_list": [
                {
                    "sentence": (raw_answer if raw_answer else "手册未明确规定"),
                    "citations": (all_ids if raw_answer else []),
                }
            ],
        }

    evidence = [n.node.metadata or {} for n in nodes]

    return {
        "query": q,
        "answer": normalized_answer,
        "chunk_map": chunk_map,
        "evidence": evidence,
    }


# 托管 / 直接返回 index.html
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")