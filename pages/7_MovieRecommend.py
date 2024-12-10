import streamlit as st
from streamlit_option_menu import option_menu

import pickle
from tmdbv3api import Movie, TMDb
import os

from analisis import recommend_code, how_recommend

movie_code = Movie()
tmdb = TMDb()
tmdb.language = "ko-KR"


# 영화의 제목을 입력받으면 코사인 유사도를 통해 가장 유사도가 높은 상위 10개의 영화 목록 반환
def get_recommendations(title):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = movies[movies["title"] == title].index[0]

    # 코사인 유사도와 매트릭스(cosine_sim)에서 idx에 해당하는 데이터를 (idx, 유사도)) 형태로 얻는다
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬
    # key 를 통하여 정렬할 기준을 정할 수 있다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]

    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indicies = [i[0] for i in sim_scores]

    # 인덱스 정보를 통해 영화 제목 추출
    movie_images = []
    movie_titles = []
    for i in movie_indicies:
        movie_id = movies["id"].iloc[i]
        details = movie_code.details(movie_id)

        image_path = details["poster_path"]
        if image_path:
            image_path = "https://image.tmdb.org/t/p/w500" + image_path
        else:
            image_path = "media/no_image.jpg"

        movie_images.append(image_path)
        movie_titles.append(details["title"])
    return movie_images, movie_titles


movies = pickle.load(open("movie_recommend/movies.pickle", "rb"))
cosine_sim = pickle.load(open("movie_recommend/cosine_sim2.pickle", "rb"))


st.set_page_config(
    page_title="movie recommendation",
    page_icon="🎬",
)

with st.sidebar:
    # tmdbkey = None
    # tmdbkey = st.text_input("Write Your TMDB API key: ", type="password")
    # os.environ["TMDB_API_KEY"] = tmdbkey
    # st.text("example: 8d9d408138b462042279dd8d0f3ef955")

    selected = option_menu(
        "구현 방법",
        ["영화 추천", "방법", "코드"],
        icons=["camera-reels", "play-btn", "file-code"],
        menu_icon="intersect",
        default_index=0,
    )

if selected == "영화 추천":

    tmdbkey = None
    tmdbkey = st.text_input("Write Your TMDB API key: ", type="password")
    os.environ["TMDB_API_KEY"] = tmdbkey
    st.text("example: 8d9d408138b462042279dd8d0f3ef955")

    if tmdbkey:
        movie_list = movies["title"].values

        st.markdown(
            """
        # Movie Recommend
    """
        )
        st.text("컨텐츠 기반 필터링(Content Based Filtering): 다양한 요소 기반")
        title = st.selectbox(
            """
    영화를 선택해주세요. 선택한 영화와 비슷한 영화를 추천해드립니다.

    Avatar는 해당 정보가 TMDb 데이터베이스에 존재하지 않아 오류가 발생합니다.
            """,
            movie_list,
        )

        if st.button("Recommend"):
            with st.spinner("잠시만 기다려주세요"):
                images, titles = get_recommendations(title)

                idx = 0
                for i in range(0, 2):
                    cols = st.columns(5)
                    for col in cols:
                        col.image(images[idx])
                        col.write(titles[idx])
                        idx += 1
    else:
        st.markdown(
            """
                    # Movie Recommend
                    
                    Welcome to Movie Recommend, choose or write movie name.

                    Start by writing TNDB API key in the sidebar.
                    """
        )

if selected == "방법":
    how_recommend.recommend_1()


if selected == "코드":
    recommend_code.making_code()
