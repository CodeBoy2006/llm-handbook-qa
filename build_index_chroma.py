#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from adapter import OpenAICompatibleEmbedding

import chromadb
from chromadb.api.types import Documents, Embeddings, IDs

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    简体中文友好的分块：基于句/段边界 + 适度重叠，避免CJK语义断裂。
    需要更智能的自适应切分时，可换用 LlamaIndex Semantic Chunker。
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator="\n\n"
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, ch in enumerate(chunks):
        docs.append(Document(text=ch, metadata={"source": "handbook", "chunk_id": i}))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/handbook.txt")
    parser.add_argument("--chroma_dir", default=os.getenv("CHROMA_DIR", "./chroma"))
    parser.add_argument("--collection", default=os.getenv("COLLECTION_NAME", "handbook"))
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # 可为空（直连官方时）
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    assert api_key, "OPENAI_API_KEY 未设置"

    os.makedirs(args.chroma_dir, exist_ok=True)

    # 1) 读取与分块
    text = load_text(args.input)
    docs = chunk_text(text, args.chunk_size, args.chunk_overlap)
    print(f"[INFO] 总长度={len(text)}，分块数={len(docs)}")

    # 2) 配置 OpenAI-Compatible Embedding
    Settings.embed_model = OpenAICompatibleEmbedding(api_key=api_key, base_url=base_url, model=model)

    # 3) 初始化 Chroma 持久化客户端 & 集合
    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_or_create_collection(name=args.collection)

    # 4) 用 LlamaIndex 封装 Chroma 向量库
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5) 建索引（批量调用在线 Embedding）
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)

    # 6) 导出 chunk 审计文件（可选）
    audit_path = os.path.join(args.chroma_dir, "chunks.jsonl")
    with open(audit_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.text, "metadata": d.metadata}, ensure_ascii=False) + "\n")
    print(f"[OK] 完成。Chroma 持久化目录：{os.path.abspath(args.chroma_dir)}")


if __name__ == "__main__":
    main()