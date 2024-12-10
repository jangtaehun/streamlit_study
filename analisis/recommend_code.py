import streamlit as st


def making_code():

    st.title("코드")

    with st.container():
        col1, col2 = st.columns(2)

        col1.download_button(
            label="Download tmdb_5000_credits.csv",
            data="movie_recommend/tmdb_5000_credits.csv",
            file_name="tmdb_5000_credits.csv",
        )
        col2.download_button(
            label="Download tmdb_5000_movies.csv",
            data="movie_recommend/tmdb_5000_movies.csv",
            file_name="tmdb_5000_movies.csv",
        )

    # 1. 인구 통계확적 필터링
    st.subheader("인구 통계학적 필터링(Demographic Filtering)")
    code_demographic = """
    import pandas as pd
    import numpy as np
    
    tmdb_credits = pd.read_csv('tmdb_5000_credits.csv')
    tmdb_movies = pd.read_csv('tmdb_5000_movies.csv')
    
    tmdb_movies = tmdb_movies.merge(tmdb_credits [['id', 'cast', 'crew']], on= 'id')
    
    C = tmdb_movies['vote_average'].mean()
    
    m = tmdb_movies['vote_count'].quantile(0.9)
    
    q_movies = tmdb_movies.copy().loc[tmdb_movies['vote_count'] >= m]
    
    
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v+m) * R) + (m / (m+v) * C)
        
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    
    q_movies = q_movies.sort_values('score', ascending=False)
    """
    st.code(code_demographic, language="python")

    # 2. 컨텐츠 기반 필터링 - 줄거리 기반
    st.subheader(
        """
    컨텐츠 기반 필터링(Content Based Filtering)
    : 줄거리 기반(overview)
    """
    )

    code_content = """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    tfidf = TfidfVectorizer(stop_words='english')

    tmdb_movies['overview'].isnull().values.any()

    tmdb_movies['overview'] = tmdb_movies['overview'].fillna('')
    tmdb_movies['overview'].isnull().values.any()
    
    tfidf_matrix = tfidf.fit_transform(tmdb_movies['overview'])
    
    from sklearn.metrics.pairwise import linear_kernel

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df2.index, index=tmdb_movies['title']).drop_duplicates()
    
    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indicies = [i[0] for i in sim_scores]
        return df2['title'].iloc[movie_indicies]
    """
    st.code(code_content, language="python")

    # 3. 컨텐츠 기반 필터링 - 다양한 요소 기반 추천
    st.subheader(
        """
    컨텐츠 기반 필터링(Content Based Filtering)
    : 다양한 요소 기반 추천(장르, 감독, 키워드 등)
    """
    )

    code_content2 = """
    from ast import literal_eval
    
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        tmdb_movies[feature] = tmdb_movies[feature].apply(literal_eval)
        
        
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i["name"]
        return np.nan
        
    tmdb_movies['director'] = tmdb_movies['crew'].apply(get_director)
    
    
    def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 5:
            names = names[:5]
        return names
    return []
    
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        tmdb_movies[feature] = tmdb_movies[feature].apply(get_list)
        
        
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else: 
                return ""
                
    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        tmdb_movies[feature] = tmdb_movies[feature].apply(clean_data)
        
    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    
    tmdb_movies['soup'] = tmdb_movies.apply(create_soup, axis=1)
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(tmdb_movies['soup'])
    
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df2.index, index=df2['title'])
    
    import pickle
    pickle.dump(movies, open('movies.pickle', 'wb'))
    pickle.dump(cosine_sim2, open('cosine_sim2.pickle', 'wb'))
    """
    st.code(code_content2, language="python")
