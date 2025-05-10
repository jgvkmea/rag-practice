from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

# ファイルの読み込み

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = PyPDFLoader("pdf/r6_guidebook_sogyo1.pdf")
raw_docs = loader.load()
print(len(raw_docs))

# テキストをチャンクに分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(raw_docs)
print(len(docs))

# テキストをベクトル化して保存
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(docs, embeddings)


# リトリーブ
retriever = db.as_retriever()

query = "持続化補助金は創業後何年までが対象ですか？"

context_docs = retriever.invoke(query)
print(f"len = {len(context_docs)}")

first_doc = context_docs[0]
print(f"metadata = {first_doc.metadata}")
print(first_doc.page_content)


# def practice_chat_model():
#     model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
#     messages = [
#         SystemMessage(content="You are a helpful assistant."),  # role="system"
#         HumanMessage("こんにちは！私はジョンと言います！"),  # role="user"
#         AIMessage(
#             content="こんにちは、ジョンさん！どのようにお手伝いできますか？",  # role="assistant"
#         ),
#         HumanMessage(content="私の名前がわかりますか？"),
#     ]
#     ai_message = model.invoke(messages)
#     print(ai_message.content)


# def practice_call_open_api():
#     client = OpenAI()

#     response = client.chat.completions.create(
#         model="gpt-4.1-nano",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "こんにちは！私はジョンと言います！"},
#         ],
#         stream=True,
#     )

#     for chunk in response:
#         content = chunk.choices[0].delta.content
#         if content is not None:
#             print(content, end="", flush=True)


