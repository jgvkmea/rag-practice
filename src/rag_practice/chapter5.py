"""5章"""

import os
from operator import itemgetter

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent-book"

open_ai_model = "gpt-4.1-nano"
model = ChatOpenAI(model=open_ai_model, temperature=0)
output_parser = StrOutputParser()


def chapter5_1() -> None:
    """5.1章"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーが入力した料理のレシピを考えてください。"),
            ("human", "{dish}"),
        ],
    )

    chain = prompt | model | output_parser
    output = chain.invoke({"dish": "カレー"})
    print(output)


@chain
def upper(text: str) -> str:
    """文字列を大文字にする"""
    return text.upper()


def chapter5_2() -> None:
    """5.2章"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ],
    )
    chain = prompt | model | output_parser | RunnableLambda(upper)
    output = chain.invoke({"input": "Hello!"})
    print(output)


def chapter5_3() -> None:
    """5.3章"""
    optimistic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください。",
            ),
            ("human", "{topic}"),
        ],
    )
    optimistic_chain = optimistic_prompt | model | output_parser

    pessimistic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。",
            ),
            ("human", "{topic}"),
        ],
    )
    pessimistic_chain = pessimistic_prompt | model | output_parser

    synthesize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは客観的AIです。{topic}について2つの意見をまとめてください。",
            ),
            (
                "human",
                "楽観的意見: {optimistic_opinion}\n悲観的意見: {pessimistic_opinion}",
            ),
        ],
    )
    synthesize_chain = (
        RunnableParallel(
            {
                "optimistic_opinion": optimistic_chain,
                "pessimistic_opinion": pessimistic_chain,
                "topic": itemgetter("topic"),
            },
        )
        | synthesize_prompt
        | model
        | output_parser
    )

    output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
    print(output)


def chapter5_4() -> None:
    """5.4章"""
    prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。
    文脈: """
    {context}
    """
    質問: {question}
    ''')

    retriever = TavilySearchAPIRetriever(k=3)

    chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()},
    ) | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())
    output = chain.invoke("東京の今日の天気は?")
    print(output)
