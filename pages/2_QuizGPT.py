import streamlit as st
import json
import os
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI

# from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from prompt import quiz_prompt

st.set_page_config(
    page_title="QuizGPT",
    page_icon="💷",
)

st.markdown(
    """
        # QuizGPT
    """
)


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

if openaikey:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    #########
    # easy
    easy_questions_chain = (
        {"context": format_docs} | quiz_prompt.easy_question_prompt | llm
    )
    easy_formatting_chain = quiz_prompt.easy_formatting_prompt | llm
    # print(easy_formatting_chain)
    # hard
    hard_questions_chain = (
        {"context": format_docs} | quiz_prompt.hard_question_prompt | llm
    )
    hard_formatting_chain = quiz_prompt.hard_formatting_prompt | llm
    #########

    # embed (X)
    @st.cache_data(show_spinner="Loading file...")
    def split_file(file):
        file_content = file.read()
        file_path = f"files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader("./files/chapter_one.txt")
        docs = loader.load_and_split(text_splitter=splitter)
        return docs

    with st.sidebar:
        select_custom = st.selectbox(
            "난이도",
            (
                "쉬움",
                "어려움",
            ),
            key="1",
        )
    if select_custom == "쉬움":

        @st.cache_data(show_spinner="Making quiz...")
        def run_quiz_chain(_docs, topic):
            chain = (
                {"context": easy_questions_chain}
                | easy_formatting_chain
                | output_parser
            )
            check = {"context": easy_questions_chain} | easy_formatting_chain
            print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(check.invoke(_docs))
            print("\n\n###########################################")
            print(chain.invoke(_docs))
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return chain.invoke(_docs)

        @st.cache_data(show_spinner="Searching Wikipedia...")
        def wiki_search(term):
            retriever = WikipediaRetriever(top_k_results=3)
            docs = retriever.get_relevant_documents(term)
            return docs

    else:

        @st.cache_data(show_spinner="Making quiz...")
        def run_quiz_chain(_docs, topic):
            chain = (
                {"context": hard_questions_chain}
                | hard_formatting_chain
                | output_parser
            )
            return chain.invoke(_docs)

        @st.cache_data(show_spinner="Searching Wikipedia...")
        def wiki_search(term):
            retriever = WikipediaRetriever(top_k_results=3)  # lang="ko"
            docs = retriever.get_relevant_documents(term)
            return docs

    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["docx", "txt", "pdf"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

    if not docs:
        st.markdown(
            """
        QuizeGPT는 Wikipedia와 사용자가 제공한 파일을 이용해 퀴즈를 만들어 제공합니다.
        
        사용을 원하신다면 파일을 업로드하거나 Wikipedia을 선택해 검색어를 입력해주세요.
        """
        )
    else:
        response = run_quiz_chain(docs, topic if topic else file.name)
        # json 출력
        count = 23  # key가 겹치는 것을 방지하기 위해 중복되지 않는 범위에서 설정
        count_score = 0

        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["질문"])

                value = st.radio(
                    "Select an option.",
                    [선택지["선택지"] for 선택지 in question["선택지s"]],
                    index=None,
                    key=count,
                )
                count += 1
                if {"선택지": value, "정답": True} in question["선택지s"]:
                    st.success("정답")
                    count_score += 1
                elif value is not None:
                    st.error("오답")
            button = st.form_submit_button()

        if count_score == count:
            st.balloons()
else:
    st.markdown(
        """
        Making a quiz about file and Wikipedia
        
        Start by writing OpenAI API key and upload a file on the sidebar.
    """
    )


with st.sidebar:
    c = st.container()
    c.link_button("git hub", url="https://github.com/jangtaehun/DocumentGPT")
