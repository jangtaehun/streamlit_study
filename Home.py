import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json

st.set_page_config(
    page_title="zzangtae's Portfolio",
    page_icon="🐈",
    initial_sidebar_state="expanded",
    layout="wide",
)


streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Tilt+Neon&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Roboto', sans-serif;
			}
            a[href] {
            text-decoration: none;
            color: #393E46;
            }
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)


@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# Options Menu
with st.sidebar:
    selected = option_menu(
        "Jang taehun",
        ["Intro", "About"],
        icons=["play-btn", "info-circle"],
        menu_icon="intersect",
        default_index=0,
    )


# Intro Page
if selected == "Intro":
    # Header
    st.title("Welcome to zzangtae's Portfolio!")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Here are the apps I made:")
            st.markdown(
                """
        - [ ] [DocumentGPT](/DocumentGPT)
        - [ ] [QuizGPT](/QuizGPT)
        - [ ] [SiteGPT](/SiteGPT)
        - [ ] [SummaryGPT](/SummaryGPT)
        - [ ] [Assistants](/Assistants)
        - [ ] [JobScrapper](/JobScrapper)
        """
            )

    st.divider()

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Here are the Data Analysis:")
            st.markdown(
                """
        - [ ] [MovieRecommend](/MovieRecommend)
        """
            )

        with col2:
            lottie1 = load_lottiefile("media/cat.json")
            st_lottie(lottie1, key="place", height=200, width=200)


if selected == "About":
    st.title("Data")
    col1, col2, col3 = st.columns(3)
    col1.subheader("Source")
    col2.subheader("Description")
    col3.subheader("Link")

    with st.container():
        col1, col2, col3 = st.columns(3)

        col1.write(":blue[Nomad coder]")
        col2.write("기초 지식 제공 - Nomad coder")
        # col2.write('American Community Survey, 5-Year Profiles, 2021, datasets DP02 - DP05')
        col3.write("https://nomadcoders.co/")

    with st.container():
        col1, col2, col3 = st.columns(3)
        # col1.image('census_graphic.png',width=150)
        col1.write(":blue[BERLIN STARTUP JOBS]")
        col2.write("직업 정보 제공 - BERLIN STARTUP JOBS")
        # col2.write('American Community Survey, 5-Year Profiles, 2021, datasets DP02 - DP05')
        col3.write("https://berlinstartupjobs.com/")

    with st.container():
        col1, col2, col3 = st.columns(3)
        # col1.image('cdc.png',width=150)
        col1.write(":blue[WEB3]")
        col2.write("직업 정보 제공 - WEB3")
        col3.write("https://web3.career/")

    with st.container():
        col1, col2, col3 = st.columns(3)
        # col1.image('ods.png',width=150)
        col1.write(":blue[TMDB]")
        col2.write("영화 데이터 제공 - TMDB")
        col3.write("https://developer.themoviedb.org/docs/getting-started")

    st.divider()

    st.title("Creator")
    with st.container():
        col1, col2 = st.columns(2)
        col1.write("**Name:**    Jang TaeHun")
        # col1.write(
        #     "**Experience:**    ~~~"
        # )
        col1.write("**Contact:**    jangth0056@gmail.com or jangth0056@naver.com")
        col1.write("**Github :**    https://github.com/jangtaehun")
        col1.write("**Thanks for stopping by!**")

with st.sidebar:
    st.link_button("git hub", url="https://github.com/jangtaehun/My_GPT")
