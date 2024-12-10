import streamlit as st
import os
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pyperclip
import time

# https://openai.com/sitemap.xml


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    # session_state 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


#############

answers_prompt = ChatPromptTemplate.from_template(
    """
    답변할 수 없다면 아무말이나 지어내지말고 모른다고 하세요.
    그리고 각 답변을 0부터 5까지의 점수로 평가해주세요.
    0점은 사용자에게 쓸모없음, 5점은 사용자에게 매우 유용함을 의미합니다.
    사용자 question에 대한 예제입니다.
    
    그리고 가장 높은 점수 하나만 출력해주세요.
    
    Make sure to inclide the answer's score.
    
    Context: {context}
    
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    
    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            
            반드시 점수가 높고 유저에게 도움이 되는 질문을 하나만 골라서 답해주세요.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


# @st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/blog\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="Site GPT",
    page_icon="👀",
)

st.markdown(
    """
        # SiteGPT
    """
)

with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Write Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

if openaikey:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )

    with st.sidebar:
        # if st.button("링크 복사"):
        #     pyperclip.copy("https://openai.com/sitemap.xml")
        #     a = st.success("링크 복사")
        #     time.sleep(2)
        #     a.empty()
        st.text("example api key: https://openai.com/sitemap.xml")
        url = st.text_input(
            "Write down a URL",
            placeholder="https://example.xml",
        )

    if url:

        if ".xml" not in url:
            with st.sidebar:
                st.error("Please write down a Sitemap url")
        else:
            retriever = load_website(url)
            query = st.text_input("Ask a question to the website.")
            if query:
                # chain 2개: 1. 모든 개별 document에 대한 답변 생성 및 채점, 2. 모든 답변을 가진 마지막 시점에 실행 -> 점수가 제일 높고, 가장 최신 정보를 담고 있는 답변 고르기
                chain = (
                    {"docs": retriever, "question": RunnablePassthrough()}
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                # get_answer이 무엇을 반환하든 choose_answer에 전달된다. / get_answer의 출력 -> choose_answer의 입력값

                result = chain.invoke(query)
                # 질문이 retriever로 전달 -> docs 반환 -> RunnablePassthrough는 question 값으로 교체된다.
                # dictionary가 get_answers function의 입력값으로 전달되고, 그 function은 dictionary에서 documents와 question 값을 추출
                st.write(result.content.replace("$", "\$"))
else:
    st.markdown(
        """
        Ask questions about the content of a website.
                
        Start by writing OpenAI API key and URL of the website on the sidebar.

    """
    )
