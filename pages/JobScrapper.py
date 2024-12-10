import streamlit as st
import csv
import json
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseOutputParser
from scrapper.total import Result_scrapper
from prompt import scrapper


# from scrapper.file import save_job
import sys
import os

sys.path.append("scrapper/total.py")

st.set_page_config(page_title="CodeChallengeGPT", page_icon="📚")


@st.cache_data(show_spinner="Searching Job...")
def start_total(keyword):
    total_scrapper = Result_scrapper(keyword)
    jobs = total_scrapper.scrap()
    total_db[keyword] = jobs
    return total_db


def save_job(keyword, jobs_db):
    file = open(f"Job_{keyword}.csv", "w", encoding="CP949")
    writter = csv.writer(file)
    writter.writerow(["Title", "Company", "Position", "url"])
    for job in jobs_db:
        writter.writerow(job.values())
    file.close()


###


# class ChatCallbackHandler(BaseCallbackHandler):
#     message = ""

#     def on_llm_start(self, *arg, **kwargs):
#         self.message_box = st.empty()

#     def on_llm_end(self, *arg, **kwargs):
#         save_message(self.message, "ai")

#     def on_llm_new_token(self, token: str, *args, **kwargs):
#         self.message += token
#         self.message_box.markdown(self.message)


def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})


# message, role 저장
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


def format_docs(docs):
    print(docs)
    return "\n\n".join(docs)


output_parser = JsonOutputParser()

st.markdown(
    """
        # Job Scrapper
    """
)

with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Write Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey


if openaikey:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=150, memory_key="chat_history", return_messages=True
    )

    total_db = {}
    keyword = st.text_input("Write Language")
    if keyword:
        total_job = start_total(keyword)
        with st.container():
            col1, col2 = st.columns(2)

            btn1 = col1.button("Job List")
            btn2 = col2.button(f"What is {keyword}")

        if btn1:
            total_job = start_total(keyword)
            term = total_job[keyword]
            with st.container():
                count = 1

                if keyword:
                    st.write(f"Searched Job: {len(term)}")

                for job in term:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.write(f":blue[{count}. {job['title']}]")
                    col2.write(f"{job['company']}")
                    col3.write(f"{job['position']}")
                    col4.write(f"{job['link']}")
                    count += 1

        if btn2:
            send_message(f"{keyword} 언에 대해 알려줘!!", "human", save=True)

            if keyword:

                @st.cache_data(show_spinner=f"Searching {keyword}...")
                def invoke_chain(keyword):
                    chain = scrapper.doc_prompt | llm
                    q_chain = chain.invoke({"term": keyword})
                    return q_chain

                q_chain = invoke_chain(keyword)
                send_message(q_chain.content, "ai", save=True)

                # def load_memory(input):
                #     return memory.load_memory_variables({})["chat_history"]

                # def invoke_chain(question):
                #     result = chain.invoke({"term": question})
                #     memory.save_context({"input": question}, {"output": result.content})
                #     return result.content

                # chain = (
                #     RunnablePassthrough.assign(chat_history=load_memory)
                #     | scrapper.doc_prompt
                #     | llm
                # )
                # a = invoke_chain(keyword)

                # json으로 형성
                # chain = scrapper.question_prompt | llm
                # q_chain = chain.invoke({"context": keyword})
                # formatting_chain = scrapper.formatting_prompt | llm | output_parser
                # j_chain = formatting_chain.invoke({"context": q_chain})
                # with st.chat_message("ai"):
                #     for question in j_chain["questions"]:
                #         head = question["프로그래밍언어"]
                #         body = [방법["방법"] for 방법 in question["공부방법"]]
else:
    st.markdown(
        """
        Search the list of jobs you want
        
        Start by writing OpenAI API key on the sidebar.
    """
    )

with st.sidebar:
    c = st.container()
    c.link_button("git hub", url="https://github.com/jangtaehun/DocumentGPT")
