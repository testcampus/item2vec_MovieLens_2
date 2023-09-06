import streamlit as st
import pandas as pd
import numpy as np
import gensim

st.title('映画レコメンド')

# 映画情報の読み込み
movies = pd.read_csv("data/movies.tsv", sep="\t")

# 学習済みのitem2vecモデルの読み込み
model = gensim.models.word2vec.Word2Vec.load("data/item2vec.model")

# 映画IDとタイトルを辞書型に変換
movie_titles = movies["title"].tolist()
movie_ids = movies["movie_id"].tolist()
movie_genres = movies["genre"].tolist()
movie_tags = movies["tag"].tolist()
movie_id_to_title = dict(zip(movie_ids, movie_titles))
movie_title_to_id = dict(zip(movie_titles, movie_ids))
movie_id_to_genre = dict(zip(movie_ids, movie_genres))

st.markdown("## 1本の映画に対して似ている映画を表示する")
selected_movie = st.selectbox("映画を選んでください", movie_titles)
selected_num = st.number_input("表示する件数を指定してください", 1, 100, 10)
selected_minscore = st.number_input("基準スコア(0.0-1.0)を指定してください", 0.0, 1.0, 0.6)
selected_movie_id = movie_title_to_id[selected_movie]
st.write(f"あなたが選択した映画は {selected_movie} です")
st.write(f"{selected_num} 件の結果を表示")
st.write("基準スコア {:.2f}".format(selected_minscore))

# 似ている映画を表示
st.markdown(f"### {selected_movie} に似ている映画")
results = []
for movie_id, score in model.wv.most_similar(positive=selected_movie_id, topn=selected_num): #上位topn件だけ表示
    if score < selected_minscore:
        continue
    title = movie_id_to_title[movie_id]
    genre = movie_id_to_genre[movie_id]
    results.append({"タイトル": title, "スコア": score, "ジャンル": eval(genre)})
results = pd.DataFrame(results)
st.write(results)

st.markdown("## 複数の映画を選んでおすすめの映画を表示する")

selected_movies = st.multiselect("映画を複数選んでください", movie_titles)
selected_movie_ids = [movie_title_to_id[movie] for movie in selected_movies]
vectors = [model.wv.get_vector(movie_id) for movie_id in selected_movie_ids] #映画のベクトルを取得
if len(selected_movies) > 0:
    user_vector = np.mean(vectors, axis=0) #平均ベクトルをユーザーのベクトルとする
    st.markdown(f"### おすすめの映画")
    recommend_results = []
    for movie_id, score in model.wv.most_similar(positive=user_vector):
        if movie_id in selected_movie_ids:
            continue
        title = movie_id_to_title[movie_id]
        genre = movie_id_to_genre[movie_id]
        recommend_results.append({"タイトル": title, "スコア": score, "ジャンル": eval(genre)})
    recommend_results = pd.DataFrame(recommend_results)
    st.write(recommend_results)
