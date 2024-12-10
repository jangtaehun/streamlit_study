import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os


st.set_page_config(page_title="CodeChallengeGPT", page_icon="📚")

st.markdown(
    """
        # DocumentGPT
    """
)

with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

    file = st.file_uploader(
        """
    챗봇을 사용하고 싶다면 파일을 업로드해주세요!! \n
    <.txt .pdf or .docx file 가능>
                            """,
        type=["pdf", "txt", "docx"],
    )

    c = st.container()
    c.link_button("git hub", url="https://github.com/jangtaehun/DocumentGPT")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def embed_file(file):
    file_content = file.read()
    file_path = f"files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"files/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(f"files/{file.name}")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    print(retriever)
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# message, role 저장
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# message 기록 보이기, 저장X
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    {context}만을 사용해 질문에 답변을 해주세요. 모른다면 모른다고 하세요. 꾸며내지 마세요.
    """,
        ),
        ("human", "{question}"),
    ]
)
if openaikey:
    # file이 존재하면 실행
    if file:
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
            # openai_api_key=openai_api_key,
        )

        retriever = embed_file(file)
        send_message("나는 준비됐어 물어봐!!", "ai", save=False)
        paint_history()
        message = st.chat_input("첨부한 파일에 대해 어떤 것이든 물어봐!!")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | template
                | llm
            )

            with st.chat_message("ai"):
                response = chain.invoke(message)

    else:
        st.session_state["messages"] = []
else:
    st.markdown(
        """
        Ask questions about the file
                
        Start by writing OpenAI API key and URL of the website on the sidebar.

    """
    )
