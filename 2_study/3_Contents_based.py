# 장르의 유사성으로 contents_based filtering
# CountVectorizer 사용

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 불러 오기
movies = pd.read_csv('../dataset/TMDB5000movie/tmdb_5000_movies.csv')
print('movies shape = ', movies.shape)
print('movies = \n', movies.head())
print(movies.columns)

# null값 확인
print(movies.isnull().sum())

# 사용할 컬럼 추출
movies_df = movies[['id', 'title','genres','vote_average', 'vote_count']]
print('movies_df\n', movies_df)


# genres 컬럼 따로 뽑아내기
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
print(movies_df['genres'][1])
# print(movies_df[['genres']][1]) # 이렇게 하니까 오류가 나네여
print(movies_df[['genres']][:2])
# {'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}
# name만 사용할 거야

movies_df['genres'] = movies_df['genres'].apply(lambda x : [y['name'] for y in x])
print(movies_df[['genres']][:5])



# 장르 콘텐츠 유사도 측정
print("\n##### 장르 콘텐츠 유사도 측정 #####")
print(movies_df[['genres']])
# [Action, Adventure, Fantasy, Science Fiction]

# CounterVectorizer 사용하려면 공백으로 이루어져 있어야 함
# 단위 구분을 공백으로 변환
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
print("\nmovies_df['genres_literal']= \n", movies_df['genres_literal'])

count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print("\ngenre_mat = \n", genre_mat)
print("\ngenre_mat shape = ", genre_mat.shape)


# 유사도 측정
print("##### 유사도 측정 #####")
genre_sim = cosine_similarity(genre_mat, genre_mat)
print("\ngenre_sim = \n", genre_sim[:1])
print("genre_sim shape = ", genre_sim.shape)
# 4803 4803

# 유사도 높은 순으로 한번 뒤집어어 봐 = argsort 써보기
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:1])
# 유사도 제일 높은건 0번 인덱스(나 자신) 그다음은 3494번 인덱스

# 장르 콘텐츠 필터링 영화 추천
print(movies_df[['title', 'vote_average','vote_count']].sort_values('vote_average', ascending=False)[:10])
# count 가 불균형함, 보정해야 됨!


# 가중치가 부여된 평점 계산 방식 사용
print("\n##### 가중치가 부여된 평점 방식 사용하기 #####\n")

percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)
c = movies_df['vote_average'].mean()

def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']

    return ((v/(v+m))*R + ((m/(v+m)))*c)

movies_df['weighted_vote'] = movies.apply(weighted_vote_average, axis = 1)
print("\n 보정 결과 \n",movies_df[['title', 'weighted_vote', 'vote_count']].sort_values('weighted_vote', ascending=False)[:10])



# 이번에는 장르 유사도까지 합쳐서
# 장르 코사인 유사도 인덱스를 가지고 있는 객체. 추천 대상의 기준이 되는 아이템 이름(영화 제목)
# 영화 제목을 입력하면 추천 영화 정보를 dataframe 으로 반환해보기
# 기준 영화  = title_movie, title_index
def find_sim_movie(df, sorted_ind,title_name, top_n =10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values

    similar_indexs = sorted_ind[title_index, :(top_n*2)]

    similar_indexs = similar_indexs.reshape(-1)

    similar_indexs = similar_indexs[similar_indexs != title_index]

    return df.iloc[similar_indexs].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather', 10)
print(similar_movies.columns)
print("\nThe Godfather의 추천 최종 결과\n", similar_movies[['title','vote_average', 'weighted_vote']])

# 장르의 유사성 확인 ~~~~
print("\nThe Godfather의 추천 최종 결과\n", similar_movies[['title', 'genres','vote_average', 'weighted_vote']])
