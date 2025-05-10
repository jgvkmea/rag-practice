import os
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def lang_smith_setup():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "agent-book"
    print("LangSmith setup complete.")


lang_smith_setup()

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)
documents = loader.load()
print(len(documents))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma")
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。
文脈: """
{context}
"""
質問: {question}
''')

# 6.4
# # 仮説的な回答を生成(HyDE)
# hypothetical_prompt = ChatPromptTemplate.from_template("""\
# 次の質問に回答する一文を書いてください。
# 質問: {question}
# """)
# hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

# hyde_rag_chain = (
#     {
#         "question": RunnablePassthrough(),
#         "context": hypothetical_chain | retriever,
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )
# hyde_rag_chain.invoke("LangChainの概要を教えて")


# # 複数のクエリでリトリーブ
# class QueryGenerationOutput(BaseModel):
#     queries: list[str] = Field(..., description="検索クエリのリスト")


# query_generation_prompt = ChatPromptTemplate.from_template("""\
# 質問に対してベクターデータベースから関連文書を検索するために、
# 3つの異なる検索クエリを生成してください。
# 距離ベースの類似性検索の限界を克服するために、
# ユーザーの質問に対して複数の視点を提供することが目標です。

# 質問: {question}
# """)

# query_generation_chain = (
#     query_generation_prompt
#     | model.with_structured_output(QueryGenerationOutput)
#     | (lambda x: x.queries)
# )

# multi_query_rag_chain = (
#     {
#         "question": RunnablePassthrough(),
#         "context": query_generation_chain | retriever.map(),
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[str]:
    # 各ドキュメントのコンテンツ (文字列) とそのスコアの対応を保持する辞書を準備
    content_score_mapping = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content

            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            # (1 / (順位 + k)) のスコアを加算
            content_score_mapping[content] += 1 / (rank + k)

    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]


# rag_fusion_chain = (
#     {
#         "question": RunnablePassthrough(),
#         "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )


# # Cohereによるリランク
# def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
#     question = inp["question"]
#     documents = inp["documents"]

#     cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
#     return cohere_reranker.compress_documents(documents=documents, query=question)


# rerank_rag_chain = (
#     {
#         "question": RunnablePassthrough(),
#         "documents": retriever,
#     }
#     | RunnablePassthrough.assign(context=rerank)
#     | prompt
#     | model
#     | StrOutputParser()
# )


# 6.5 複数のRetriever
from langchain_community.retrievers import TavilySearchAPIRetriever

langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain_document_retriever"},
)

web_retriever = TavilySearchAPIRetriever(k=3).with_config(
    {"run_name": "web_retriever"},
)

from enum import Enum


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


route_prompt = ChatPromptTemplate.from_template("""\
質問に回答するために適切なRetrieverを選択してください。

質問: {question}
""")

route_chain = (
    route_prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)
)


def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    if route == Route.web:
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown route: {route}")


route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt
    | model
    | StrOutputParser()
)


# ハイブリッド検索
from langchain_community.retrievers import BM25Retriever

chroma_retriever = retriever.with_config(
    {"run_name": "chroma_retriever"},
)

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retriever"},
)

from langchain_core.runnables import RunnableParallel

hybrid_retriever = (
    RunnableParallel(
        {
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        },
    )
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
)

hybrid_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

hybrid_rag_chain.invoke("LangChainの概要を教えて")
