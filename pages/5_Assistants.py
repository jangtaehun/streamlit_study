import streamlit as st
import openai as client
import os
import time

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="Assistans",
    page_icon="🔥",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def load_memory(input):
    return memory.load_memory_variables({})["history"]


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# # message, role 저장
# def send_message(message, role, save=True):
#     with st.chat_message(role):
#         st.markdown(message)
#     if save:
#         save_message(message, role)


# message 기록 보이기, 저장X
# def paint_history():
#     for message in st.session_state["messages"]:
#         send_message(message["message"], message["role"], save=False)

##############


# @st.cache_data(show_spinner="Create assistant_id...")
def create_id():
    assistant = client.beta.assistants.create(
        name="Book Assistant",
        instructions="You help users with their question on the files they upload.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
    )
    return assistant.id


# @st.cache_data(show_spinner="Create thread_id...")
def thread_id():
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "I want you to help me with this file",
            }
        ]
    )
    return thread.id


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


#############


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        if len(message.content) == 0:
            continue
        return f"{message.role}: {message.content[0].text.value}"
        # for annotation in message.content[0].text.annotations:
        #     print(f"Source: {annotation.file_citation}")


def run(thread_id, assistant_id):
    runing = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return runing.id


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


###################

st.markdown(
    """
        # Assistants
    """
)

with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Write Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

if openaikey:

    llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=150, memory_key="history"
    )

    assistant_id = create_id()
    thread = thread_id()

    with st.sidebar:
        file = st.file_uploader(
            """
    You can upload a .txt .pdf or .docx file
                            """,
            type=["pdf", "txt", "docx"],
        )

    if file:

        @st.cache_data(show_spinner="Create file_id...")
        def file_id():
            file_doc = client.files.create(
                file=client.file_from_path(f"files/{file.name}"), purpose="assistants"
            )
            return file_doc.id

        @st.cache_data(show_spinner="Create client Threads...")
        def create_client(thread_id, file_id):
            client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content="", file_ids=[file_id]
            )

        file_id = file_id()
        create_client(thread, file_id)

        text = st.chat_input("첨부한 파일에 대해 어떤 것이든 물어봐!!")

        a = get_messages(thread)
        with st.chat_message("human"):
            st.markdown(a)

        if text:
            with st.chat_message("human"):
                st.markdown(text)

            send_message(thread, text)

            run(thread, assistant_id)
            time.sleep(10)

            b = get_messages(thread)
            st.write(b)

            # with st.chat_message("ai"):
            #     st.markdown(a)
        # send_message("나는 준비됐어 물어봐!!", "ai", save=False)
else:
    st.markdown(
        """
        Ask questions about the content of a file
                
        Start by writing OpenAI API key and upload a file on the sidebar.
    """
    )
